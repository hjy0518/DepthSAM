# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import cv2
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as ops
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Union
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .MyNet import Decode


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class MOEAdapter(nn.Module):
    def __init__(self, blk, num_experts=8, top_k=2) -> None:
        super(MOEAdapter, self).__init__()
        self.block = blk
        self.num_experts = num_experts
        self.top_k = top_k

        dim = blk.attn.qkv.in_features

        # 门控网络
        self.gate = nn.Linear(dim, num_experts)

        # 多个专家网络
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
        x_flat = x.reshape(B * N, C)  # 展平空间维度

        # 计算门控权重
        gate_logits = self.gate(x_flat)  # (B*H*W, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # 归一化

        # 混合专家输出
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_weights[:, i].unsqueeze(-1)

            # 对每个专家进行前向传播
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += expert_weight[mask] * expert_output

        # 恢复原始形状
        output = output.reshape(B, N, C)

        # 残差连接
        prompted = x + output

        # 通过原始block
        net = self.block(prompted)
        return net

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        model_path = f'checkpoints/depth_anything_v2_vitl.pth'
        self.image_encoder.load_state_dict(torch.load(model_path))
        for param in self.image_encoder.parameters():
            param.requires_grad = False


        blocks = []
        for block in self.image_encoder.pretrained.blocks:
            blocks.append(
                MOEAdapter(block)
            )
        self.image_encoder.pretrained.blocks = nn.Sequential(
            *blocks
        )

        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.decoder = Decode(256,256,256,256)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],x
    ) -> [Tensor, Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """

        x = F.interpolate(x, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        depth,features = self.image_encoder(x)

        out1,out_1 = self.decoder(features[3], features[2], features[1], features[0])
        outputs = []
        for image_record, curr_embedding,out11 in zip(batched_input, out_1,out1):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=image_record.get("boxes", None),
                masks=out11.unsqueeze(0),
            )

            low_res_mask, iou = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            outputs.append(
                {
                    "mask": low_res_mask,
                    "low_res_logits": low_res_mask,
                }
            )
        masks = torch.cat([x["mask"] for x in outputs], dim=0)
        return masks


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(masks, original_size, mode="bilinear")

        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        x = x.unsqueeze(0)
        return x
