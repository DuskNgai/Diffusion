from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from sampler import SAMPLER_FORMULATION_TABLE
from .build import NOISE_SCHEDULER_REGISTRY
from .base import NoiseScheduler

__all__ = ["EDMNoiseScheduler"]


@NOISE_SCHEDULER_REGISTRY.register()
class EDMNoiseScheduler(NoiseScheduler):
    """
    Noise scheduler for EDM Diffusion model.
    """
    @configurable
    def __init__(self,
        timestep_mean: float,
        timestep_std: float,
        prediction_type: str,
        sigma_data: float,
    ) -> None:
        super().__init__(
            prediction_type=prediction_type,
            sigma_data=sigma_data,
            scale_fn=SAMPLER_FORMULATION_TABLE["EDM"]["scale_fn"],
            scale_deriv_fn=SAMPLER_FORMULATION_TABLE["EDM"]["scale_deriv_fn"],
            sigma_fn=SAMPLER_FORMULATION_TABLE["EDM"]["sigma_fn"],
            sigma_deriv_fn=SAMPLER_FORMULATION_TABLE["EDM"]["sigma_deriv_fn"],
        )

        self.timestep_mean = timestep_mean
        self.timestep_std = timestep_std

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "timestep_mean": cfg.MODEL.NOISE_SCHEDULER.TIMESTEP_MEAN,
            "timestep_std": cfg.MODEL.NOISE_SCHEDULER.TIMESTEP_STD,
            "prediction_type": cfg.MODEL.PREDICTION_TYPE,
            "sigma_data": cfg.MODEL.SIGMA_DATA,
        }

    def sample_timestep(self, sample: torch.Tensor) -> torch.Tensor | torch.LongTensor:
        timestep = torch.exp(torch.randn(sample.shape[0], device=sample.device) * self.timestep_std + self.timestep_mean)
        while timestep.dim() < sample.dim():
            timestep = timestep.unsqueeze(-1)
        return timestep
