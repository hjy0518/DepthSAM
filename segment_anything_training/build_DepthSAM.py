import torch
import torch.nn as nn
from functools import partial
from .modeling import PromptEncoder, TwoWayTransformer
from .modeling.DepthSAM_decoder import MaskDecoder as EdgeDecoder
from .modeling.DepthSAM_edge import Sam as EdgeDepthSAM
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def build_sam_DepthSAM():
    prompt_embed_dim = 256
    image_size = 512
    vit_patch_size = 2
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = EdgeDepthSAM(
            image_encoder=DepthAnythingV2(**model_configs['vitl']),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim, #通道
                image_embedding_size=(image_embedding_size, image_embedding_size), # 尺寸
                input_image_size=(image_size, image_size), # 输入尺寸
                mask_in_chans=16,
            ),
            mask_decoder= EdgeDecoder(
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
    return mobile_sam