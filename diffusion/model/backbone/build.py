from fvcore.common.registry import Registry
from omegaconf import DictConfig

import torch

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbone, whose input is noisy samples and output is the something related.
"""

def build_backbone(cfg: DictConfig) -> torch.nn.Module:
    """
    Build the backbone defined by `cfg.MODEL.BACKBONE.NAME`.
    It does not load checkpoints from `cfg`.
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
    return backbone
