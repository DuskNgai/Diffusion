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
        sigma_data: float,
        prediction_type: str = "sample",
    ) -> None:
        super().__init__(
            sigma_data=sigma_data,
            scale_fn=SAMPLER_FORMULATION_TABLE["Rectified Flow"]["scale_fn"],
            scale_deriv_fn=SAMPLER_FORMULATION_TABLE["Rectified Flow"]["scale_deriv_fn"],
            sigma_fn=SAMPLER_FORMULATION_TABLE["Rectified Flow"]["sigma_fn"],
            sigma_deriv_fn=SAMPLER_FORMULATION_TABLE["Rectified Flow"]["sigma_deriv_fn"],
            prediction_type=prediction_type,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "sigma_data": cfg.MODEL.SIGMA_DATA,
            "prediction_type": cfg.MODEL.CRITERION.PREDICTION_TYPE,
        }

    def forward_sample(self,
        prediction: torch.Tensor,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        return (prediction - sample).square().mean()

    def forward_epsilon(self,
        prediction: torch.Tensor,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        return (prediction - noise).square().mean()

    def forward_velocity(self,
        prediction: torch.Tensor,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        velocity = self.scale_deriv_fn(timestep) * sample + self.sigma_deriv_fn(timestep) * noise
        return (prediction - velocity).square().mean()
