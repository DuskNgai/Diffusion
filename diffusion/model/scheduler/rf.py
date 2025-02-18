from typing import Any

from diffusers.configuration_utils import register_to_config
from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from .build import NOISE_SCHEDULER_REGISTRY
from sampler import (
    SAMPLER_FORMULATION_TABLE,
    ContinuousTimeTrainingNoiseScheduler,
    ContinuousTimeNoiseScheduler,
)

__all__ = [
    "RectifiedFlowTrainingNoiseScheduler",
    "RectifiedFlowNoiseScheduler",
]


@NOISE_SCHEDULER_REGISTRY.register()
class RectifiedFlowTrainingNoiseScheduler(ContinuousTimeTrainingNoiseScheduler):
    """
    Noise scheduler for training Rectified Flow model.
    """

    @configurable
    @register_to_config
    def __init__(
        self,
        timestep_mean: float,
        timestep_std: float,
        prediction_type: str,
        sigma_data: float,
    ) -> None:
        FORMULATION = SAMPLER_FORMULATION_TABLE["Rectified Flow"]
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
        timestep = torch.sigmoid(
            torch.randn(sample.shape[0], device=sample.device) * self.config.timestep_std + self.config.timestep_mean
        )
        while timestep.dim() < sample.dim():
            timestep = timestep.unsqueeze(-1)
        return timestep


class RectifiedFlowNoiseScheduler(ContinuousTimeNoiseScheduler):

    def __init__(
        self,
        t_min: float = 0.0001,
        t_max: float = 0.9999,
        sigma_data: float = 1.0,
        prediction_type: str = "velocity",
        algorithm_type: str = "ode",
        timestep_schedule: str = "linear_lognsr",
    ) -> None:
        FORMULATION = SAMPLER_FORMULATION_TABLE["Rectified Flow"]
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
