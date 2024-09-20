from abc import ABCMeta, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

__all__ = ["DiffusionCriterion"]


class DiffusionCriterion(nn.Module, metaclass=ABCMeta):
    """
    
    """
    def __init__(self,
        sigma_data: float,
        scale: Callable[[float | torch.Tensor], float | torch.Tensor],
        scale_deriv: Callable[[float | torch.Tensor], float | torch.Tensor],
        sigma: Callable[[float | torch.Tensor], float | torch.Tensor],
        sigma_deriv: Callable[[float | torch.Tensor], float | torch.Tensor],
        prediction_type: str = "sample",
    ) -> None:
        super().__init__()

        self.sigma_data = sigma_data
        self.scale = scale
        self.scale_deriv = scale_deriv
        self.sigma = sigma
        self.sigma_deriv = sigma_deriv
        self.prediction_type = prediction_type
        assert self.prediction_type in ["sample", "epsilon", "velocity"]

    @abstractmethod
    def sample_timesteps(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def add_noise(self, samples: torch.Tensor, noises: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        scales = self.scale(timesteps)
        sigmas = self.sigma(timesteps)
        return scales * samples + sigmas * noises, scales, sigmas

    @abstractmethod
    def forward(self, samples: torch.Tensor, noises: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
