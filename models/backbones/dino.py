import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from functools import partial
from peft import LoraConfig, get_peft_model
from models.svf import *
from modules.module1.freq_masker import MaskModule
from modules.module1.phase_attn import PhaseAttention


class FrequencyEnhancer(nn.Module):
    """Wrapper that applies MaskModule + PhaseAttention to encoder features."""

    def __init__(self, channels, num_branches, mask_mode="per_layer"):
        super().__init__()
        self.mask_mode = mask_mode
        self.expected_branches = num_branches
        num_modules = 1 if mask_mode == "shared" else num_branches
        self.masks = nn.ModuleList([MaskModule() for _ in range(num_modules)])
        self.attentions = nn.ModuleList([PhaseAttention(channels) for _ in range(num_modules)])

    def _select_module(self, collection, index):
        if self.mask_mode == "shared":
            return collection[0]
        return collection[index]

    def forward(self, features):
        if len(features) != self.expected_branches:
            raise ValueError(
                f"FrequencyEnhancer expected {self.expected_branches} feature maps, got {len(features)}"
            )
        enhanced = []
        for idx, feature in enumerate(features):
            mask_module = self._select_module(self.masks, idx)
            att_module = self._select_module(self.attentions, idx)
            x = mask_module(feature)
            x = att_module(x)
            enhanced.append(x)
        return tuple(enhanced)

def create_backbone_dino(method, model_repo_path, pretrain_dir) : 
    sys.path.insert(0,os.path.join(model_repo_path, "dino"))
    from vision_transformer import vit_base, vit_small
    dino_backbone = vit_small(patch_size = 16)
    dino_backbone.load_state_dict(torch.load(os.path.join(pretrain_dir, "dino_vitbase16_pretrain.pth")))
    if method == "multilayer" :  
        n = 4
    else : 
        n = 1
    dino_backbone.forward = partial(
            dino_backbone.get_intermediate_layers,
            n=n,
        )
    return dino_backbone

def create_backbone_dinov2(method, model_repo_path, pretrain_dir, dinov2_size="base") : 
    sys.path.insert(0,os.path.join(model_repo_path, "dinov2"))
    from dinov2.models.vision_transformer import vit_base, vit_large, vit_small
    
    if dinov2_size == 'base':
        dino_backbone = vit_base(patch_size = 14, img_size = 524, init_values=1.0, block_chunks=0, num_register_tokens=0)
        dino_backbone.load_state_dict(torch.load(os.path.join(pretrain_dir,"dinov2_vitb14_pretrain.pth")))
    elif dinov2_size == 'small':
        dino_backbone = vit_small(patch_size = 14, img_size = 524, init_values=1.0, block_chunks=0, num_register_tokens=0)
        dino_backbone.load_state_dict(torch.load(os.path.join(pretrain_dir,"dinov2_vits14_pretrain.pth")))
    elif dinov2_size == 'large':
        dino_backbone = vit_large(patch_size = 14, img_size = 524, init_values=1.0, block_chunks=0, num_register_tokens=0)
        dino_backbone.load_state_dict(torch.load(os.path.join(pretrain_dir,"dinov2_vitl14_pretrain.pth")))
    else:
        raise ValueError(f"Unsupported DINOv2 size: '{dinov2_size}'. Please choose 'small', 'base' or 'large'.")

    if method == "multilayer" :
        n = [8,9,10,11]
    else : 
        n = [11]
    dino_backbone.forward = partial(
            dino_backbone.get_intermediate_layers,
            n=n,
            reshape=True,
        )
    return dino_backbone

class DINO_linear(nn.Module):
    def __init__(
        self,
        version,
        method,
        num_classes,
        input_size,
        model_repo_path,
        pretrain_dir,
        dinov2_size="base",
        enable_frequency_adapter=True,
        freq_mask_mode="per_layer",
    ):
        super().__init__()
        self.method = method
        self.version = version
        self.input_size = input_size
        self.patch_size = 14 if self.version == 2 else 16
        self.feature_branches = 4 if self.method == "multilayer" else 1
        if self.version == 2 : 
            self.encoder = create_backbone_dinov2(method, model_repo_path, pretrain_dir, dinov2_size)
            if dinov2_size == 'base':
                base_channels = 768
            elif dinov2_size == 'small':
                base_channels = 384
            elif dinov2_size == 'large':
                base_channels = 1024
            else:
                raise ValueError(f"Unsupported DINOv2 size: '{dinov2_size}'. Please choose 'small', 'base' or 'large'.")
        else : # DINOv1
            self.encoder = create_backbone_dino(method, model_repo_path, pretrain_dir)
            base_channels = 384 # DINOv1 vit_small has 384

        if self.method == "vpt" : 
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.register_tokens.requires_grad = True
        if method == "svf" : 
            self.encoder = resolver(self.encoder)
        if method == "lora" : 
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["qkv"],
                lora_dropout=0.1,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, config)
        
        self.base_channels = base_channels
        self.in_channels = base_channels * self.feature_branches

        self.use_frequency_adapter = enable_frequency_adapter
        if self.use_frequency_adapter:
            self.frequency_adapter = FrequencyEnhancer(
                channels=self.base_channels,
                num_branches=self.feature_branches,
                mask_mode=freq_mask_mode,
            )
        else:
            self.frequency_adapter = None

        self.decoder = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.bn = nn.SyncBatchNorm(self.in_channels)

    def forward(self, x): 
        input_dim = int(self.input_size / self.patch_size) * self.patch_size
        x = F.interpolate(x, size=[input_dim, input_dim], mode='bilinear', align_corners=False)

        grad_enabled = self.method not in ["linear", "multilayer"]
        with torch.set_grad_enabled(grad_enabled):
            features = self.encoder(x)

        if not isinstance(features, (list, tuple)):
            features = (features,)

        if self.frequency_adapter is not None:
            features = self.frequency_adapter(features)

        x = torch.cat(features, dim=1)
        x = self.bn(x)
        return self.decoder(x)
