from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2, Compose

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY, build_transform

__all__ = ["CIFAR10Dataset"]


@DATASET_REGISTRY.register()
class CIFAR10Dataset(CIFAR10):
    """
    A wrapper around torchvision.datasets.CIFAR10 to provide a consistent interface.
    """

    @configurable
    def __init__(
        self,
        root: str,
        train: bool,
        transform: v2.Compose | Compose | None = None,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, download=True)

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {
            "root": Path(cfg.DATASET.ROOT),
            "train": stage == RunningStage.TRAINING,
            "transform": build_transform(cfg, stage),
        }

    @property
    def sampler(self) -> None:
        return None

    @property
    def collate_fn(self) -> None:
        return None
