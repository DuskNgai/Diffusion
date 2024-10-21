from abc import ABCMeta, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

__all__ = ["NoiseScheduler"]


class NoiseScheduler(nn.Module, metaclass=ABCMeta):
    """
    Training phase noise scheduler for diffusion models, whose forward process is defined as:
        `x_t = scale(t) * x_0 + sigma(t) * noise`.
    """
    def __init__(self,
        sigma_data: float,
        scale_fn: Callable[[float | torch.Tensor], float | torch.Tensor],
        scale_deriv_fn: Callable[[float | torch.Tensor], float | torch.Tensor],
        sigma_fn: Callable[[float | torch.Tensor], float | torch.Tensor],
        sigma_deriv_fn: Callable[[float | torch.Tensor], float | torch.Tensor],
    ) -> None:
        super().__init__()

        self.sigma_data = sigma_data
        self.scale_fn = scale_fn
        self.scale_deriv_fn = scale_deriv_fn
        self.sigma_fn = sigma_fn
        self.sigma_deriv_fn = sigma_deriv_fn

    @abstractmethod
    def sample_timestep(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        scale = self.scale_fn(timestep)
        sigma = self.sigma_fn(timestep)
        return scale * sample + sigma * noise, scale, sigma
