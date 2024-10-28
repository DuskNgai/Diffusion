from typing import Any

from diffusers.configuration_utils import register_to_config
from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from .build import NOISE_SCHEDULER_REGISTRY
from sampler import (
    SAMPLER_FORMULATION_TABLE,
    BaseContinuousTimeNoiseScheduler,
    ContinuousTimeTrainingNoiseScheduler,
    ContinuousTimeNoiseScheduler,
)

__all__ = [
    "EDMTrainingNoiseScheduler",
    "EDMNoiseScheduler"
]


class BaseEDMNoiseScheduler(BaseContinuousTimeNoiseScheduler):
    def preprocess(self, noisy: torch.Tensor, scale: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        c_in = 1.0 / ((scale * self.config.sigma_data) ** 2 + sigma ** 2).sqrt()
        c_noise = 0.25 * sigma.log()
        return c_in * noisy, scale, c_noise

    def postprocess(self, noisy: torch.Tensor, prediction: torch.Tensor, scale: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.config.prediction_type == "sample":
            c_out = (sigma * self.config.sigma_data) / ((scale * self.config.sigma_data) ** 2 + sigma ** 2).sqrt()
            c_skip = scale * self.config.sigma_data ** 2 / ((scale * self.config.sigma_data) ** 2 + sigma ** 2)
        elif self.config.prediction_type == "epsilon":
            c_out = (scale * self.config.sigma_data) / ((scale * self.config.sigma_data) ** 2 + sigma ** 2).sqrt()
            c_skip = sigma / ((scale * self.config.sigma_data) ** 2 + sigma ** 2)
        elif self.config.prediction_type == "velocity":
            raise NotImplementedError
        else:
            raise KeyError(f"Unknown prediction type: {self.config.prediction_type}")

        return c_skip * noisy + c_out * prediction


@NOISE_SCHEDULER_REGISTRY.register()
class EDMTrainingNoiseScheduler(ContinuousTimeTrainingNoiseScheduler, BaseEDMNoiseScheduler):
    """
    Noise scheduler for EDM Diffusion model.
    """
    @configurable
    @register_to_config
    def __init__(self,
        timestep_mean: float,
        timestep_std: float,
        prediction_type: str,
        sigma_data: float,
    ) -> None:
        FORMULATION = SAMPLER_FORMULATION_TABLE["EDM"]
        super().__init__(
            prediction_type=prediction_type,
            sigma_data=sigma_data,
            scale_fn=FORMULATION["scale_fn"],
            scale_deriv_fn=FORMULATION["scale_deriv_fn"],
            sigma_fn=FORMULATION["sigma_fn"],
            sigma_deriv_fn=FORMULATION["sigma_deriv_fn"],
            nsr_inv_fn=FORMULATION["nsr_inv_fn"],
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "timestep_mean": cfg.MODEL.NOISE_SCHEDULER.TIMESTEP_MEAN,
            "timestep_std": cfg.MODEL.NOISE_SCHEDULER.TIMESTEP_STD,
            "prediction_type": cfg.MODEL.PREDICTION_TYPE,
            "sigma_data": cfg.MODEL.SIGMA_DATA,
        }

    def sample_timestep(self, sample: torch.Tensor) -> torch.Tensor | torch.LongTensor:
        timestep = torch.exp(torch.randn(sample.shape[0], device=sample.device) * self.config.timestep_std + self.config.timestep_mean)
        while timestep.dim() < sample.dim():
            timestep = timestep.unsqueeze(-1)
        return timestep


class EDMNoiseScheduler(ContinuousTimeNoiseScheduler, BaseEDMNoiseScheduler):
    def __init__(self,
        t_min: float = 0.002,
        t_max: float = 80.0,
        sigma_data: float = 1.0,
        prediction_type: str = "sample",
        algorithm_type: str = "ode",
        timestep_schedule: str = "linear_lognsr",
    ) -> None:
        FORMULATION = SAMPLER_FORMULATION_TABLE["EDM"]
        super().__init__(
            t_min=t_min,
            t_max=t_max,
            sigma_data=sigma_data,
            scale_fn=FORMULATION["scale_fn"],
            scale_deriv_fn=FORMULATION["scale_deriv_fn"],
            sigma_fn=FORMULATION["sigma_fn"],
            sigma_deriv_fn=FORMULATION["sigma_deriv_fn"],
            nsr_inv_fn=FORMULATION["nsr_inv_fn"],
            prediction_type=prediction_type,
            algorithm_type=algorithm_type,
            timestep_schedule=timestep_schedule,
        )
