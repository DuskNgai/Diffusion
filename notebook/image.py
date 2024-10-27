from pathlib import Path
import sys

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from diffusers import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
import torch

from sampler import ContinuousTimeNoiseScheduler


class UnconditionalGenerationPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "model"

    def __init__(self, model: torch.nn.Module, scheduler: ContinuousTimeNoiseScheduler) -> None:
        super().__init__()

        self.register_modules(model=model, scheduler=scheduler)

    @torch.inference_mode()
    def __call__(self,
        batch_size: int,
        num_inference_steps: int,
        generator: torch.Generator | list[torch.Generator] | None = None
    ) -> torch.Tensor:
        # 0. Sample the initial noisy samples.
        sample = randn_tensor(
            (batch_size, self.model.img_channels, self.model.img_resolution, self.model.img_resolution),
            generator=generator,
            device=self.device
        ) * self.scheduler.init_noise_sigma

        # 1. Initialize the scheduler.
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        for timestep in self.progress_bar(self.scheduler.timesteps):
            scale = self.scheduler.config.scale_fn(timestep)
            sigma = self.scheduler.config.sigma_fn(timestep)

            # 2. Compute the output of the model at the current sample.
            processed_noisy, processed_scale, processed_sigma = self.scheduler.preprocess(sample, scale, sigma)
            unprocessed_output = self.model(processed_noisy, processed_scale, processed_sigma)
            output = self.scheduler.postprocess(sample, unprocessed_output, scale, sigma)

            # 3. Update the sample.
            sample = self.scheduler.step(output, timestep, sample).prev_sample

        # 4. Reset the scheduler.
        self.scheduler.step_index = None

        return sample
