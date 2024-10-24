from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from sampler import FunctionType

__all__ = ["NoiseScheduler"]


class NoiseScheduler(nn.Module, metaclass=ABCMeta):
    """
    Training phase noise scheduler for diffusion models, whose forward process is defined as:
        `x_t = scale(t) * x_0 + sigma(t) * noise`.

    Args:
        prediction_type (`str`): The type of prediction to make. One of "sample", "epsilon", or "velocity".
        sigma_data (`float`): The (estimated) standard deviation of the training data.
        scale_fn (`FunctionType`): The scale function for the noisy data.
        scale_deriv_fn (`FunctionType`): The derivative of the scale function.
        sigma_fn (`FunctionType`): The noise level for the noisy data.
        sigma_deriv_fn (`FunctionType`): The derivative of the noise level.
    """
    def __init__(self,
        prediction_type: str,
        sigma_data: float,
        scale_fn: FunctionType,
        scale_deriv_fn: FunctionType,
        sigma_fn: FunctionType,
        sigma_deriv_fn: FunctionType,
    ) -> None:
        super().__init__()

        self.prediction_type = prediction_type
        assert self.prediction_type in ["sample", "epsilon", "velocity"], f"Unknown prediction type: {self.prediction_type}"

        self.sigma_data = sigma_data
        self.scale_fn = scale_fn
        self.scale_deriv_fn = scale_deriv_fn
        self.sigma_fn = sigma_fn
        self.sigma_deriv_fn = sigma_deriv_fn

    @abstractmethod
    def sample_timestep(self, sample: torch.Tensor) -> torch.Tensor | torch.LongTensor:
        """
        Sample timesteps from a predefined distribution.
        """
        raise NotImplementedError

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.scale_fn(timestep)
        sigma = self.sigma_fn(timestep)

        noisy = scale * sample + sigma * noise
        if self.prediction_type == "sample":
            target = sample
        elif self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "velocity":
            scale_deriv = self.scale_deriv_fn(timestep)
            sigma_deriv = self.sigma_deriv_fn(timestep)
            target = scale_deriv * sample + sigma_deriv * noise
        else:
            raise KeyError(f"Unknown prediction type: {self.prediction_type}")

        return noisy, target, scale, sigma
