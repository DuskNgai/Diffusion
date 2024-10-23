from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from sampler import FunctionType

__all__ = ["DiffusionCriterion"]


class DiffusionCriterion(nn.Module, metaclass=ABCMeta):
    """
    The base class for diffusion model criterion.

    Args:
        sigma_data (`float`): The (estimated) standard deviation of the training data.
        scale_fn (`FunctionType`): The scale function for the noisy data.
        scale_deriv_fn (`FunctionType`): The derivative of the scale function.
        sigma_fn (`FunctionType`): The noise level for the noisy data.
        sigma_deriv_fn (`FunctionType`): The derivative of the noise level.
        prediction_type (`str`): The type of prediction to make. One of "sample", "epsilon", or "velocity".
    """
    def __init__(self,
        sigma_data: float,
        scale_fn: FunctionType,
        scale_deriv_fn: FunctionType,
        sigma_fn: FunctionType,
        sigma_deriv_fn: FunctionType,
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
