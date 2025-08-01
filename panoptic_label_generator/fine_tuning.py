from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from models.dino_v2 import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)
from models.segment_anything import sam_vit_b, sam_vit_h, sam_vit_l
from models.eva02 import eva02_vitb14, eva02_vitl14
from torch import nn


def interpolate_eva_pos_embed(model, new_hw):
    # Extract original pos_embed
    pos_embed = model.pos_embed  # shape [1, N+1, D]
    cls_token = pos_embed[:, :1]  # [1, 1, D]
    patch_pos_embed = pos_embed[:, 1:]  # [1, N, D]
    
    N = patch_pos_embed.shape[1]
    D = patch_pos_embed.shape[2]
    old_hw = int(N ** 0.5)
    
    # Reshape and interpolate
    patch_pos_embed = patch_pos_embed.reshape(1, old_hw, old_hw, D).permute(0, 3, 1, 2)  # [1, D, H, W]
    patch_pos_embed = F.interpolate(patch_pos_embed, size=new_hw, mode='bicubic', align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_hw[0]*new_hw[1], D)  # [1, new_N, D]
    
    new_pos_embed = torch.cat((cls_token, patch_pos_embed), dim=1)  # [1, new_N+1, D]
    model.pos_embed = torch.nn.Parameter(new_pos_embed)


def interpolate_sam_pos_embed(image_encoder, new_hw):
    """
    Interpolates SAM ViT absolute positional embeddings for new input resolution.

    Args:
        image_encoder: The SAM image encoder module (ImageEncoderViT)
        new_hw: Tuple (H, W) in patch units, not pixels
    """
    if image_encoder.pos_embed is None:
        return

    pos_embed = image_encoder.pos_embed  # shape [1, H_old, W_old, D]
    old_h, old_w = pos_embed.shape[1:3]
    D = pos_embed.shape[3]

    pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, D, H_old, W_old]
    pos_embed = F.interpolate(pos_embed, size=new_hw, mode='bicubic', align_corners=False)
    pos_embed = pos_embed.permute(0, 2, 3, 1)  # [1, H_new, W_new, D]

    image_encoder.pos_embed = nn.Parameter(pos_embed)


class FineTuner(pl.LightningModule):
    def __init__(self, vit_model: str, backbone: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None):
        super().__init__()
        self.backbone = backbone
        self.vit_model = vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor

        if self.backbone == 'dino':
            if vit_model == 'vits14':
                self.encoder = dinov2_vits14(pretrained=True)
            elif vit_model == 'vitb14':
                self.encoder = dinov2_vitb14(pretrained=True)
            elif vit_model == 'vitl14':
                self.encoder = dinov2_vitl14(pretrained=True)
            elif vit_model == 'vitg14':
                self.encoder = dinov2_vitg14(pretrained=True)
            else:
                raise ValueError(f'Unknown vit model {vit_model}')
            self.feat_dim = self.encoder.num_features
            self.patch_size = self.encoder.patch_size
        elif self.backbone == "sam":
            if vit_model == 'vitb16':
                self.encoder = sam_vit_b(pretrained=True)
            elif vit_model == 'vitl16':
                self.encoder = sam_vit_l(pretrained=True)
            elif vit_model == 'vith16':
                self.encoder = sam_vit_h(pretrained=True)
            else:
                raise ValueError(f'Unknown vit model {vit_model}')
            self.feat_dim = self.encoder.patch_embed.proj.out_channels  # 768
            self.patch_size = self.encoder.patch_embed.proj.kernel_size[0]  # 16

            interpolate_sam_pos_embed(self.encoder, (448 // self.patch_size, 896 // self.patch_size))
        elif self.backbone == 'eva':
            if vit_model == 'vitb14':
                self.encoder = eva02_vitb14(pretrained=True)
            elif vit_model == 'vitl14':
                self.encoder = eva02_vitl14(pretrained=True)
            else:
                raise ValueError(f'Unknown vit model {vit_model}')
            self.feat_dim = self.encoder.embed_dim  # 768
            self.patch_size = self.encoder.patch_embed.proj.kernel_size[0]  # 14
            self.encoder.patch_embed.img_size = (448, 896)
            interpolate_eva_pos_embed(self.encoder, (448 // self.patch_size, 896 // self.patch_size))
        else:
            raise ValueError(f'Unknown backbone {backbone}')

        self.encoder.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise

        for param in self.encoder.parameters():  # freeze backbone
                param.requires_grad = False

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size
        requires_grad = any(param.requires_grad for param in self.encoder.parameters())

        if self.backbone == 'dino':
            with torch.set_grad_enabled(requires_grad):
                return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])
                
                block_outputs = self.encoder.forward_features(
                    img,
                    return_attention_features=return_attention_features,
                    return_blocks=self.blocks)

                if self.blocks is None:
                    block_outputs = [block_outputs]
                outs = []
                for x in block_outputs:
                    x = x[feature_key]
                    if feature_key == 'attn':
                        return x  # (B, num_heads, Patches+1, Patches+1)
                    if feature_key in ['q', 'k', 'v']:
                        # (B, Patches+1, num_heads, feat_dim // num_heads)
                        x = x.permute((0, 2, 1, 3)).contiguous()
                        x = x.reshape((x.shape[0], -1, self.feat_dim))  # (B, Patches+1, feat_dim)
                    outs.append(x)
                x = torch.cat(outs, dim=2)  # (B, Patches+1, feat_dim * self.num_blocks)

                x = x[:, 1:, :]  # (B, Patches, feat_dim)
                x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim, H*W)
                x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h,
                                patches_w))  # (B, feat_dim, H, W)
                if self.upsample_factor is not None:
                    x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                                    align_corners=False)  # (B, feat_dim, H, W)
                return x
        elif self.backbone == 'sam':
            with torch.set_grad_enabled(requires_grad):
                x = self.encoder.patch_embed(img)  # (B, H', W', C)

                if self.encoder.pos_embed is not None:
                    x = x + self.encoder.pos_embed

                block_outputs = []
                for blk in self.encoder.blocks:
                    x = blk(x)
                    block_outputs.append(x)

                if self.blocks is not None:
                    selected_blocks = [block_outputs[i] for i in self.blocks]
                else:
                    selected_blocks = [block_outputs[-1]]  # Last layer

                outs = []
                for x in selected_blocks:
                    # (B, H', W', C) => reshape to (B, C, H', W')
                    x = x.permute(0, 3, 1, 2).contiguous()
                    outs.append(x)

                # Concatenate along channel dim if multiple blocks
                x = torch.cat(outs, dim=1)  # (B, C * num_blocks, H, W)

                if self.upsample_factor is not None:
                    x = F.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
                return x
        elif self.backbone == 'eva':
            with torch.set_grad_enabled(requires_grad):
                B, C = img.shape[:2]
                x = self.encoder.patch_embed(img)  # shape: [B, num_patches, C]
                N = patches_h * patches_w

                x = torch.cat([self.encoder.cls_token.expand(B, -1, -1), x], dim=1)  # [B, N+1, C]
                x = x + self.encoder.pos_embed
                x = self.encoder.pos_drop(x)

                # Pos embedding done, now optionally normalize
                if hasattr(self.encoder, 'norm_pre') and self.encoder.norm_pre is not None:
                    x = self.encoder.norm_pre(x)

                # Interpolate rotary pos embed (not handled by interpolate_pos_embed())
                if hasattr(self.encoder, 'rotary_pos_emb') and self.encoder.rotary_pos_emb is not None:
                    orig_hw = int((self.encoder.rotary_pos_emb.shape[-2]) ** 0.5)
                    rope = self.encoder.rotary_pos_emb  # [depth, num_heads, H*W, dim]
                    print(rope.size())
                    if rope.shape[2] != N:
                        rope = F.interpolate(
                            rope.permute(0, 1, 3, 2).reshape(-1, orig_hw, orig_hw),
                            size=(patches_h, patches_w),
                            mode='bicubic',
                            align_corners=False
                        ).reshape(rope.shape[0], rope.shape[1], rope.shape[3], N).permute(0, 1, 3, 2)
                    print(rope.size())
                else:
                    rope = None

                # Forward through transformer blocks
                for i, blk in enumerate(self.encoder.blocks):
                    if rope is not None:
                        x = blk(x, rope=rope[i])
                    else:
                        x = blk(x)

                x = self.encoder.norm(x) # [B, N+1, C]
                x = x[:, 1:]  # remove CLS -> [B, N, C]
                x = x.transpose(1, 2).reshape(B, -1, patches_h, patches_w)  # [B, C, H, W]

                if self.upsample_factor is not None:
                    x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                                    align_corners=False)  # (B, C, H, W)
                return x
        else:
            raise ValueError(f'Unknown backbone {self.backbone}')
