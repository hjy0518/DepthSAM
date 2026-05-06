import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn.functional as F
import datetime
from segment_anything_training.build_DepthSAM import build_sam_DepthSAM
from utils.dataset_rgb_strategy2 import SalObjDataset
from utils.utils import adjust_lr, AvgMeter
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as data
import math
import random
import numpy as np
from data_cod import test_dataset
import cv2



def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, distributed=False):
    dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=4,
                                  pin_memory=True, drop_last=True, sampler=sampler)
    return data_loader, sampler


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )
        torch.cuda.set_device(args.gpu)
        dist.barrier()  # 等待所有进程初始化完成
        distributed = True

        print(f"Distributed training initialized: rank {args.rank}/{args.world_size}, gpu {args.gpu}")

    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        distributed = True

        print(f"SLURM distributed training: rank {args.rank}, gpu {args.gpu}")

    else:
        print("Not using distributed mode")
        distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    return distributed, args.gpu


class MOEAdapter(nn.Module):
    def __init__(self, blk, num_experts=8, top_k=4) -> None:
        super(MOEAdapter, self).__init__()
        self.block = blk
        self.num_experts = num_experts
        self.top_k = top_k

        self.register_buffer("current_aux_loss", torch.tensor(0.0))

        dim = blk.attn.qkv.in_features
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 32),
                nn.GELU(),
                nn.Linear(32, dim),
                nn.GELU(),
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: (B, H, W, C)
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)

        gate_logits = self.gate(x_flat)
        gate_weights = F.softmax(gate_logits, dim=-1)

        if self.training:
            avg_prob_per_expert = gate_weights.mean(dim=0)
            _, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
            expert_mask = F.one_hot(top_k_indices, self.num_experts).sum(dim=1)
            tokens_per_expert = expert_mask.float().sum(dim=0)
            frac_tokens_per_expert = tokens_per_expert / (B * N)
            load_balance_loss = (avg_prob_per_expert * frac_tokens_per_expert).sum()
            self.current_aux_loss = load_balance_loss * self.num_experts
        else:
            self.current_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # 归一化

        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_weights[:, i].unsqueeze(-1)

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += expert_weight[mask] * expert_output

        output = output.reshape(B, N, C)
        prompted = x + output
        net = self.block(prompted)

        return net


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        print("nominator:", loss_partial)

        return loss


Contrastive_Loss = ContrastiveLoss(batch_size=1)


class MOELossCollector:
    def __init__(self, alpha=0.01):

        self.alpha = alpha

    def get_total_loss(self, model: nn.Module) -> torch.Tensor:

        total_aux_loss = 0.0

        for module in model.modules():
            if isinstance(module, MOEAdapter):
                total_aux_loss += module.current_aux_loss

        total_loss = self.alpha * total_aux_loss

        return total_loss


moe_loss_collector = MOELossCollector(alpha=0.01)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=200, help='every n epochs decay learning rate')
    parser.add_argument('-beta1_gen', type=float, default=0.5, help='beta of Adam for generator')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')
    parser.add_argument('--gpu', type=int, default='0', help='reduced channel of saliency feat')
    return parser.parse_args()


def train():
    opt = get_args()

    image_cod_root = "./Data_all/COD-D/Train_depth/Imgs/"
    gt_cod_root = "./Data_all/COD-D/Train_depth/GT/"
    depth_cod_root = "./ECCV-1/Data_all/COD-D/Train_depth/depth/"

    train_loader, cod_sampler = get_loader(image_cod_root, gt_cod_root, depth_cod_root, batchsize=opt.batchsize,
                                           trainsize=opt.trainsize, distributed=False)

    save_path = './cpts/'

    print("开始初始化模型，优化器...")
    generator = build_sam_DepthSAM()
    generator.cuda()
    generator_optimizer = torch.optim.Adam(generator.parameters(), opt.lr_gen)

    total_step = len(train_loader)
    print("Start Training...")
    for epoch in range(1, opt.epoch + 1):
        generator.train()
        loss_record = AvgMeter()
        print('Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

        train_loader_iter = iter(train_loader)

        for i, (images, gts, depth) in enumerate(train_loader_iter):

            images = images.cuda()
            gts = gts.cuda()

            imgs = images.permute(0, 2, 3, 1).cpu().numpy()
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = (torch.as_tensor((imgs[b_i]).astype(dtype=np.uint8), device=generator.device)
                               .permute(2, 0, 1).contiguous())
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            s1 = generator(batched_input, images)
            total_loss = moe_loss_collector.get_total_loss(generator)
            loss1 = structure_loss(s1, gts)
            loss = loss1 + total_loss

            loss.backward()
            generator_optimizer.step()
            generator_optimizer.zero_grad()

            loss_record.update(loss.data, opt.batchsize)
            if i % 200 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Pre Loss: {:.4f}, Pre1 Loss: {:.4f}'.
                      format(datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(), loss1.data))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if epoch >= 10 or epoch % opt.epoch == 0:
            torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
            w_path = save_path + 'Model_' + str(epoch) + '_gen.pth'
            test_cod(w_path)

best_mae = 10000
best_epoch = 0

def test_cod(w_path):
    opt = get_args()
    global best_mae, best_epoch

    test_path = './Data_all/COD-D/Test_depth/'
    test_datasets = ['CAMO']


    save_path = './cpts/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generator = build_sam_DepthSAM()

    data = torch.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)

    generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = './test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = test_path + dataset + '/Imgs/'
        gt_root = test_path + dataset + '/GT/'
        d_root = test_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, d_root, opt.trainsize)

        mae_sum = 0

        for i in range(test_loader.size):  # 250
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            imgs = image.permute(0, 2, 3, 1).cpu().numpy()
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = (torch.as_tensor((imgs[b_i]).astype(dtype=np.uint8), device=generator.device)
                               .permute(2, 0, 1).contiguous())
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            res = generator(batched_input, image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().detach().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size

        print(dataset, 'Res mae is : ', mae)

if __name__ == '__main__':
    train()
