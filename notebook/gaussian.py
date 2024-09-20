from pathlib import Path
import sys
from typing import Callable, Union

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models import ModelMixin
from diffusers.utils.torch_utils import randn_tensor
import torch
import torch.nn as nn

from scheduling import GeneralContinuousDiffusionScheduler

__all__ = [
    "GaussianModel",
    "GaussianModelScheduler",
    "GaussianModelPipeline",
] 


class GaussianModel(ModelMixin, ConfigMixin):
    """
    The Gaussian Model (GM) is a probabilistic model that assumes all the data points are generated from a Gaussian distributions.
    Its score function is known in closed form and can be used to validate the implementation of the sampler.

    The Gaussian Model has the following probability density function:
        `p(x) \propto exp(-0.5 * (x - mu)^T * cov^-1 * (x - mu))`,
    where `mu` is the mean vector and `cov` is the covariance matrix of the Gaussian distribution.
    """
    def __init__(self,
        mu: torch.Tensor,
        cov: torch.Tensor
    ) -> None:
        super().__init__()

        assert mu.dim() == 1 and cov.dim() == 2, "Support single Gaussian distribution only."
        assert mu.shape[-1] == cov.shape[-1] == cov.shape[-2], "The dim of mu and cov must be the same."

        self.mu = nn.Parameter(mu, requires_grad=False)
        self.cov = nn.Parameter(cov, requires_grad=False)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        The probability density function of the noisy samples generated from the Gaussian Model is given by:
            `p(x) \propto exp(-0.5 * (x - scale * mu)^T * (scale^2 * cov + sigma^2 * I)^-1 * (x - scale * mu))`,
        therefore, the score function of the Gaussian Model is given by:
            `score = - (scale^2 * cov + sigma^2 * I)^-1 * (x - scale * mu)`,

        Args:
            x (torch.Tensor): The input tensor.
            scale (torch.Tensor): The scale of noisy samples.
            sigma (torch.Tensor): The noise level of noisy samples.

        Returns:
            torch.Tensor: The score of the GMM at the given x.
        """

        cov = torch.inverse((scale ** 2) * self.cov + (sigma ** 2) * torch.eye(self.cov.shape[-1], device=self.cov.device))
        score = torch.einsum("ij,bj->bi", cov, scale * self.mu - x) # [B, 2]
        return score


class GaussianModelScheduler(GeneralContinuousDiffusionScheduler):
    @register_to_config
    def __init__(self,
        t_min: float,
        t_max: float,
        sigma_data: float = 1.0,
        scale: Callable[[Union[float, torch.Tensor]], Union[float, torch.Tensor]] = lambda t: 1.0,
        scale_deriv: Callable[[Union[float, torch.Tensor]], Union[float, torch.Tensor]] = lambda t: 0.0,
        sigma: Callable[[Union[float, torch.Tensor]], Union[float, torch.Tensor]] = lambda t: t,
        sigma_deriv: Callable[[Union[float, torch.Tensor]], Union[float, torch.Tensor]] = lambda t: 1.0,
        nsr_inv: Callable[[Union[float, torch.Tensor]], Union[float, torch.Tensor]] = lambda nsr: nsr,
        prediction_type: str = "epsilon",
        algorithm_type: str = "ode",
        timestep_schedule: str = "linear_lognsr",
        **kwargs
    ):
        super().__init__(
            t_min=t_min,
            t_max=t_max,
            sigma_data=sigma_data,
            scale=scale,
            sigma=sigma,
            nsr_inv=nsr_inv,
            prediction_type=prediction_type,
            algorithm_type=algorithm_type,
            timestep_schedule=timestep_schedule,
            **kwargs
        )


class GaussianModelPipeline(DiffusionPipeline):
    """
    The Gaussian Mixture Model (GMM) pipeline is a simple example of the Diffusion Pipeline.

    The relation ship between the score function and the prediction type is as follows:
        - If the prediction type is "epsilon", then `epsilon = - sigma * score`.
        - If the prediction type is "sample", then `sample = (sigma ** 2 * score + sample) / scale`.
        - If the prediction type is "velocity", then `velocity = (scale_deriv / scale) * sample + (scale_deriv / scale - sigma_deriv / sigma) * sigma ** 2 * score`.
    """

    model_cpu_offload_seq = "model"

    def __init__(self, model: GaussianModel, scheduler: GaussianModelScheduler) -> None:
        super().__init__()

        self.register_modules(model=model, scheduler=scheduler)

    @torch.inference_mode()
    def __call__(self,
        batch_size: int,
        num_inference_steps: int,
        generator: Union[torch.Generator, list[torch.Generator]] | None = None
    ) -> torch.Tensor:
        # 0. Sample the initial noisy samples.
        sample = randn_tensor((batch_size, 2), generator=generator, device=self.device) * self.scheduler.init_noise_sigma

        # 1. Initialize the scheduler.
        self.scheduler.set_timesteps(num_inference_steps)

        for timestep in self.progress_bar(self.scheduler.timesteps):
            scale = self.scheduler.config.scale(timestep)
            sigma = self.scheduler.config.sigma(timestep)

            # 2. Compute the score function of the GMM at the current sample.
            score = self.model(sample, scale, sigma)

            # 3. Perform the reverse diffusion process.
            if self.scheduler.config.prediction_type == "epsilon":
                output = - sigma * score
            elif self.scheduler.config.prediction_type == "sample":
                output = (sigma ** 2 * score + sample) / scale
            elif self.scheduler.config.prediction_type == "velocity":
                scale_deriv = self.scheduler.config.scale_deriv(timestep)
                sigma_deriv = self.scheduler.config.sigma_deriv(timestep)
                output = (scale_deriv / scale) * sample + (scale_deriv / scale - sigma_deriv / sigma) * sigma ** 2 * score
            else:
                raise ValueError(f"Prediction type {self.scheduler.config.prediction_type} is not supported.")

            # 4. Update the sample.
            sample = self.scheduler.step(output, timestep, sample).prev_sample

        # 5. Reset the scheduler.
        self.scheduler.step_index = None

        return sample