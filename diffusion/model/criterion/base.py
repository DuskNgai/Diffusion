from abc import ABCMeta, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

__all__ = ["DiffusionCriterion"]


class DiffusionCriterion(nn.Module, metaclass=ABCMeta):
    """
    The base class for diffusion model criterion.
    """
    def __init__(self,
        sigma_data: float,
        scale_fn: Callable[[float | torch.Tensor], float | torch.Tensor],
        scale_deriv_fn: Callable[[float | torch.Tensor], float | torch.Tensor],
        sigma_fn: Callable[[float | torch.Tensor], float | torch.Tensor],
        sigma_deriv_fn: Callable[[float | torch.Tensor], float | torch.Tensor],
        prediction_type: str,
    ) -> None:
        super().__init__()

        self.sigma_data = sigma_data
        self.scale_fn = scale_fn
        self.scale_deriv_fn = scale_deriv_fn
        self.sigma_fn = sigma_fn
        self.sigma_deriv_fn = sigma_deriv_fn

        self.prediction_type = prediction_type
        assert self.prediction_type in ["sample", "epsilon", "velocity"], f"Unknown prediction type: {self.prediction_type}"

    def forward(self,
        prediction: torch.Tensor,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        fn = getattr(self, f"forward_{self.prediction_type}")
        return fn(prediction, sample, noise, timestep)

    @abstractmethod
    def forward_sample(self,
        prediction: torch.Tensor,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_epsilon(self,
        prediction: torch.Tensor,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_velocity(self,
        prediction: torch.Tensor,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
