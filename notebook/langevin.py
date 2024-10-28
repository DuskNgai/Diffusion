import math
from typing import Optional, Union

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
import torch


class LangevinNoiseScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements general sampler for continuous time diffusion models, whose forward process is defined as:
        `x_t = scale(t) * x_0 + sigma(t) * noise`.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        t_min (`float`):
            Minimum time parameter for the forward diffusion process.
        t_max (`float`):
            Maximum time parameter for the forward diffusion process.
        algorithm_type (`str`, defaults to `ode`):
            Algorithm type for the solver; can be `ode` or `sde`. 
        kwargs:
            Additional keyword arguments for compatibility.
    """
    @register_to_config
    def __init__(self,
        t_min: float = 0.002,
        t_max: float = 80.0,
        algorithm_type: str = "ode",
        **kwargs,
    ):
        assert algorithm_type in ["ode", "sde"], f"Algorithm type {algorithm_type} is not supported."

        # Setable values
        self.num_inference_steps = None
        self.timesteps = None
        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self) -> Optional[int]:
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @step_index.setter
    def step_index(self, step_index: Optional[int] = None) -> None:
        """
        Sets the step index for the scheduler.

        Args:
            step_index (`int`, *optional*):
                The index counter for current timestep.
        """
        self._step_index = step_index

    @property
    def begin_index(self) -> Optional[int]:
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    @begin_index.setter
    def begin_index(self, begin_index: int = 0) -> None:
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(self,
        num_inference_steps: int = None,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
        """
        self.num_inference_steps = num_inference_steps

        ramp = torch.linspace(0, 1, self.num_inference_steps)
        timesteps = self._compute_uniform_timesteps(ramp)

        self.timesteps = timesteps.to(dtype=torch.float32, device=device)

    def _compute_uniform_timesteps(self, ramp: torch.Tensor) -> torch.Tensor:
        ts = self.config.t_max + ramp * (self.config.t_min - self.config.t_max)
        return ts

    def first_order_update(self,
        score: torch.Tensor,
        sample: torch.Tensor,
        noise: Union[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            score (`torch.Tensor`):
                The direct output score from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            noise (`torch.Tensor`):
                Noise tensor for the stochastic sampling.

        Returns:
            `torch.Tensor`:
                A previous instance of a sample created by the diffusion process.
        """
        t, u = self.timesteps[self.step_index + 1] if self.step_index < self.num_inference_steps - 1 else self.config.t_min, self.timesteps[self.step_index]
        x_u = sample
        h = u - t

        if self.config.algorithm_type == "ode":
            x_t = x_u + h * score
        elif self.config.algorithm_type == "sde":
            x_t = x_u + 2 * h * score + h ** 0.5 * noise
        else:
            raise ValueError(f"Algorithm type {self.config.algorithm_type} is not supported.")

        return x_t

    # Copied from diffusers.schedulers.scheduling_edm_dpmsolver_multistep.EDMDPMSolverMultistepScheduler._init_step_index
    def _init_step_index(self, timestep: torch.Tensor):
        """
        Initialize the step_index counter for the scheduler.
        """
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    # Copied from diffusers.schedulers.scheduling_edm_dpmsolver_multistep.EDMDPMSolverMultistepScheduler.index_for_timestep
    def index_for_timestep(self, timestep: torch.Tensor, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates = (schedule_timesteps == timestep).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        return step_index

    def step(self,
        score: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: Union[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, tuple]:
        """
        Predict the sample from the previous timestep following the reverse process.

        Args:
            score (`torch.Tensor`):
                The direct output score from learned diffusion model.
            timestep (`torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        if self.config.algorithm_type == "sde":
            noise = randn_tensor(
                score.shape,
                generator=generator,
                device=score.device,
                dtype=score.dtype
            )
        else:
            noise = None

        prev_sample = self.first_order_update(score, sample, noise)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)


class LangevinGaussianModelPipeline(DiffusionPipeline):
    """
    The Gaussian Model (GM) pipeline is a simple example of the Diffusion Pipeline.
    Here we use the Langevin diffusion process to sample from the Gaussian model.
    """

    model_cpu_offload_seq = "model"

    def __init__(self, model: torch.nn.Module, scheduler: LangevinNoiseScheduler) -> None:
        super().__init__()

        self.register_modules(model=model, scheduler=scheduler)

    @torch.inference_mode()
    def __call__(self,
        batch_size: int,
        num_inference_steps: int,
        random_sampling: bool
    ) -> torch.Tensor:
        # 0. Sample the initial noisy samples.
        if random_sampling:
            sample = torch.randn((batch_size, 2), device=self.device)
        else:
            sample = torch.zeros((batch_size, 2), device=self.device)

        # 1. Initialize the scheduler.
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        for timestep in self.progress_bar(self.scheduler.timesteps):
            # 2. Compute the score function of the GMM at the current sample.
            score = self.model(sample, 1.0, 0.0)

            # 3. Update the sample.
            sample = self.scheduler.step(score, timestep, sample).prev_sample

        # 4. Reset the scheduler.
        self.scheduler.step_index = None

        return sample
