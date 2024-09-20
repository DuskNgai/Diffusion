from typing import Any, Callable

import torch

from omegaconf import DictConfig

from coach_pl.configuration import configurable
from coach_pl.model import CRITERION_REGISTRY
from scheduling import SCHEDULER_FORMULATION_TABLE
from .base import DiffusionCriterion

__all__ = ["RectifiedFlowCriterion"]


@CRITERION_REGISTRY.register()
class RectifiedFlowCriterion(DiffusionCriterion):
    """
    Criterion for Rectified Flow model.
    """

    @configurable
    def __init__(self,
        timestep_mean: float,
        timestep_std: float,
        sigma_data: float,
        scale: Callable[[float | torch.Tensor], float | torch.Tensor] = SCHEDULER_FORMULATION_TABLE["Rectified Flow"]["scale"],
        scale_deriv: Callable[[float | torch.Tensor], float | torch.Tensor] = SCHEDULER_FORMULATION_TABLE["Rectified Flow"]["scale_deriv"],
        sigma: Callable[[float | torch.Tensor], float | torch.Tensor] = SCHEDULER_FORMULATION_TABLE["Rectified Flow"]["sigma"],
        sigma_deriv: Callable[[float | torch.Tensor], float | torch.Tensor] = SCHEDULER_FORMULATION_TABLE["Rectified Flow"]["sigma_deriv"],
        prediction_type: str = "sample",
    ) -> None:
        super().__init__(
            sigma_data=sigma_data,
            scale=scale,
            scale_deriv=scale_deriv,
            sigma=sigma,
            sigma_deriv=sigma_deriv,
            prediction_type=prediction_type,
        )

        self.timestep_mean = timestep_mean
        self.timestep_std = timestep_std

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "timestep_mean": cfg.MODEL.CRITERION.TIMESTEP_MEAN,
            "timestep_std": cfg.MODEL.CRITERION.TIMESTEP_STD,
            "sigma_data": cfg.MODEL.SIGMA_DATA,
            "prediction_type": cfg.MODEL.CRITERION.PREDICTION_TYPE,
        }

    def sample_timesteps(self, samples: torch.Tensor) -> torch.Tensor:
        timesteps = torch.sigmoid(torch.randn(samples.shape[0], device=samples.device) * self.timestep_std + self.timestep_mean)
        while timesteps.dim() < samples.dim():
            timesteps = timesteps.unsqueeze(-1)
        return timesteps

    def forward(self,
        predictions: torch.Tensor,
        samples: torch.Tensor,
        noises: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        if self.prediction_type == "sample":
            weights = (self.scale(timesteps) * (self.scale_deriv(timesteps) / self.scale(timesteps) - self.sigma_deriv(timesteps) / self.sigma(timesteps))) ** 2
            return (weights * (predictions - samples).square()).mean()
        elif self.prediction_type == "epsilon":
            weights = (self.sigma(timesteps) * (self.scale_deriv(timesteps) / self.scale(timesteps) - self.sigma_deriv(timesteps) / self.sigma(timesteps))) ** 2
            return (weights * (predictions - noises).square()).mean()
        elif self.prediction_type == "velocity":
            weights = 1.0
            velocities = self.scale_deriv(timesteps) * samples + self.sigma_deriv(timesteps) * noises
            return (weights * (predictions - velocities).square()).mean()
