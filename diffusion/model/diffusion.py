from typing import Any

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY
from diffusion.model.backbone import build_backbone

__all__ = ["GeneralContinuousTimeDiffusion"]


@MODEL_REGISTRY.register()
class GeneralContinuousTimeDiffusion(ModelMixin, ConfigMixin):
    @configurable
    def __init__(self,
        backbone: torch.nn.Module,
        prediction_type: str,
        sigma_data: float,
        require_pre_and_post_processing: bool
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.prediction_type = prediction_type
        self.sigma_data = sigma_data
        self.require_pre_and_post_processing = require_pre_and_post_processing

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "backbone": build_backbone(cfg),
            "prediction_type": cfg.MODEL.PREDICTION_TYPE,
            "sigma_data": cfg.MODEL.SIGMA_DATA,
            "require_pre_and_post_processing": cfg.MODEL.REQUIRE_PRE_AND_POST_PROCESSING,
        }

    def forward(self,
        x: torch.Tensor,
        scale: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        if self.require_pre_and_post_processing:
            c_in, c_out, c_skip, c_noise = self.pre_and_post_processing(scale, sigma)
            return c_skip * x + c_out * self.backbone(c_in * x, scale, c_noise * sigma, condition)
        else:
            return self.backbone(x, scale, sigma, condition)

    def pre_and_post_processing(self, scale: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_in = 1.0 / ((scale * self.sigma_data) ** 2 + sigma ** 2).sqrt()
        if self.prediction_type == "sample":
            c_out = sigma * self.sigma_data / ((scale * self.sigma_data) ** 2 + sigma ** 2).sqrt()
            c_skip = scale * self.sigma_data ** 2 / ((scale * self.sigma_data) ** 2 + self.sigma_data ** 2)
        elif self.prediction_type == "epsilon":
            c_out = (scale * self.sigma_data) / ((scale * self.sigma_data) ** 2 + sigma ** 2).sqrt()
            c_skip = sigma / ((scale * self.sigma_data) ** 2 + self.sigma_data ** 2)
        elif self.prediction_type == "velocity":
            raise NotImplementedError
        else:
            raise KeyError(f"Unknown prediction type: {self.prediction_type}")
        c_noise = 0.25 * sigma.log()

        return c_in, c_out, c_skip, c_noise
