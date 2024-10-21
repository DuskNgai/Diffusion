from fvcore.common.registry import Registry
from omegaconf import DictConfig

from .base import NoiseScheduler

NOISE_SCHEDULER_REGISTRY = Registry("NOISE_SCHEDULER")
NOISE_SCHEDULER_REGISTRY.__doc__ = """
Registry for noise scheduler, which is a derived class of `NoiseScheduler`.
It adds noise to the input tensor based on the timestep.
"""

def build_noise_scheduler(cfg: DictConfig) -> NoiseScheduler:
    """
    Build the noise scheduler defined by `cfg.MODEL.NOISE_SCHEDULER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    noise_scheduler_name = cfg.MODEL.NOISE_SCHEDULER.NAME
    noise_scheduler = NOISE_SCHEDULER_REGISTRY.get(noise_scheduler_name)(cfg)
    return noise_scheduler
