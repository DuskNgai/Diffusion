from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from coach_pl.model import CRITERION_REGISTRY
from sampler import SAMPLER_FORMULATION_TABLE
from .base import DiffusionCriterion

__all__ = ["RectifiedFlowCriterion"]


@CRITERION_REGISTRY.register()
class RectifiedFlowCriterion(DiffusionCriterion):
    """
    Criterion for Rectified Flow Diffusion model.
    """
    @configurable
    def __init__(self,
        prediction_type: str,
    ) -> None:
        super().__init__(
            prediction_type=prediction_type,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "prediction_type": cfg.MODEL.PREDICTION_TYPE,
        }

    def forward(self,
        input: torch.Tensor,
        target: torch.Tensor,
        scale: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        return (input - target).square().mean()
