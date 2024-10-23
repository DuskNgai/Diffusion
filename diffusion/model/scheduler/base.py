from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from sampler import FunctionType

__all__ = ["NoiseScheduler"]


class NoiseScheduler(nn.Module, metaclass=ABCMeta):
    """
    Training phase noise scheduler for diffusion models, whose forward process is defined as:
        `x_t = scale(t) * x_0 + sigma(t) * noise`.
    """
    def __init__(self,
        sigma_data: float,
        scale_fn: FunctionType,
        scale_deriv_fn: FunctionType,
        sigma_fn: FunctionType,
        sigma_deriv_fn: FunctionType,
    ) -> None:
        super().__init__()

        self.sigma_data = sigma_data
        self.scale_fn = scale_fn
        self.scale_deriv_fn = scale_deriv_fn
        self.sigma_fn = sigma_fn
        self.sigma_deriv_fn = sigma_deriv_fn

    @abstractmethod
    def sample_timestep(self, sample: torch.Tensor) -> torch.Tensor | torch.LongTensor:
        raise NotImplementedError

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        scale = self.scale_fn(timestep)
        sigma = self.sigma_fn(timestep)
        return scale * sample + sigma * noise, scale, sigma
