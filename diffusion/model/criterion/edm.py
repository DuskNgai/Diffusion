from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from coach_pl.model import CRITERION_REGISTRY
from .base import DiffusionCriterion

__all__ = ["EDMCriterion"]


@CRITERION_REGISTRY.register()
class EDMCriterion(DiffusionCriterion):
    """
    Criterion for EDM Diffusion model.
    """
    @configurable
    def __init__(self,
        prediction_type: str,
        sigma_data: float,
    ) -> None:
        super().__init__(
            prediction_type=prediction_type,
        )

        self.sigma_data = sigma_data

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "prediction_type": cfg.MODEL.PREDICTION_TYPE,
            "sigma_data": cfg.MODEL.SIGMA_DATA,
        }

    def forward(self,
        input: torch.Tensor,
        target: torch.Tensor,
        scale: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        if self.prediction_type == "sample":
            weight = ((scale * self.sigma_data) ** 2 + sigma ** 2) / ((sigma * self.sigma_data) ** 2)
            return (weight * (input - target).square()).mean()
        elif self.prediction_type == "epsilon":
            weight = ((scale * self.sigma_data) ** 2 + sigma ** 2) / ((scale * self.sigma_data) ** 2)
            return (weight * (input - target).square()).mean()
        elif self.prediction_type == "velocity":
            raise NotImplementedError
        else:
            raise KeyError(f"Unknown prediction type: {self.prediction_type}")
