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


def _load_state_dict_safely(module: nn.Module, ckpt_path: str, strict: bool = True):
    """Load a state dict from a checkpoint path with common-key handling.

    This helper increases robustness across differently packed checkpoints
    by probing common keys (e.g., 'state_dict', 'model').
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        # Try common containers first
        for key in ["state_dict", "model", "module"]:
            if key in state and isinstance(state[key], dict):
                try:
                    module.load_state_dict(state[key], strict=strict)
                    return
                except Exception:
                    pass
        # Fall back to assuming it's already a flat state_dict
        try:
            module.load_state_dict(state, strict=strict)
            return
        except Exception:
            # As a last resort, try non-strict to maximize compatibility while warning
            missing, unexpected = module.load_state_dict(state, strict=False)
            warn_msg = [
                f"Non-strict load used for '{ckpt_path}'.",
                f"Missing keys: {list(missing)[:10]}{' ...' if len(missing) > 10 else ''}",
                f"Unexpected keys: {list(unexpected)[:10]}{' ...' if len(unexpected) > 10 else ''}",
            ]
            print("\n".join(warn_msg))
            return
    else:
        # Unknown format; let it raise for visibility
        module.load_state_dict(state, strict=strict)


def create_backbone_dinov3(method, pretrain_dir, dinov3_size: str = "base"):
    """Build a DINOv3 ViT backbone and adapt its forward for feature extraction.

    - method 'multilayer': return the last 4 layers; otherwise return the last layer.
    - reshape=True so outputs are in [B, C, H, W] for downstream decoders.
    """
    from dinov3.models.vision_transformer import vit_base, vit_large

    # Resolve checkpoint path and peek into its keys to configure the model accordingly
    if dinov3_size == "base":
        ckpt_name = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        base_channels = 768
    elif dinov3_size == "large":
        ckpt_name = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        base_channels = 1024
    else:
        raise ValueError(
            f"Unsupported DINOv3 size: '{dinov3_size}'. Please choose 'base' or 'large'."
        )

    ckpt_path = os.path.join(pretrain_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"DINOv3 weights not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    # Unwrap state dict if needed
    if isinstance(raw, dict):
        for k in ["state_dict", "model", "module"]:
            if k in raw and isinstance(raw[k], dict):
                sd = raw[k]
                break
        else:
            sd = raw if all(isinstance(v, torch.Tensor) for v in raw.values()) else raw
    else:
        sd = raw

    # Infer architecture flags from checkpoint keys
    def has_key_suffix(suffix: str) -> bool:
        return any(k.endswith(suffix) for k in sd.keys())

    n_storage_tokens = 0
    if "storage_tokens" in sd and isinstance(sd["storage_tokens"], torch.Tensor):
        # shape = [1, N, C]
        n_storage_tokens = int(sd["storage_tokens"].shape[1])

    mask_k_bias = has_key_suffix("attn.qkv.bias_mask")
    use_layerscale = has_key_suffix("ls1.gamma") or has_key_suffix("ls2.gamma")
    layerscale_init = 1e-6 if use_layerscale else None
    untie_cls_and_patch_norms = any(k.startswith("cls_norm.") for k in sd.keys())

    common_kwargs = dict(
        patch_size=16,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=mask_k_bias,
        layerscale_init=layerscale_init,
        untie_cls_and_patch_norms=untie_cls_and_patch_norms,
    )

    backbone = vit_base(**common_kwargs) if dinov3_size == "base" else vit_large(**common_kwargs)

    # Now strictly load with the tuned flags
    _load_state_dict_safely(backbone, ckpt_path, strict=True)

    # Select last layers based on depth to avoid hard-coding indices
    depth = getattr(backbone, "n_blocks", None) or 12
    if method == "multilayer":
        n = list(range(depth - 4, depth))
    else:
        n = [depth - 1]

    backbone.forward = partial(
        backbone.get_intermediate_layers,
        n=n,
        reshape=True,
    )
    return backbone, base_channels

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
        dinov3_size="base",
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
        elif self.version == 3:
            self.encoder, base_channels = create_backbone_dinov3(method, pretrain_dir, dinov3_size)
        else : # DINOv1
            self.encoder = create_backbone_dino(method, model_repo_path, pretrain_dir)
            base_channels = 384 # DINOv1 vit_small has 384

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
