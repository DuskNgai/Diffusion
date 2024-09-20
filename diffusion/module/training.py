from typing import Any

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from rich import print
from timm.optim import optim_factory
from timm.scheduler import scheduler_factory
import torch

from coach_pl.configuration import configurable
from coach_pl.model import build_model, build_criterion
from coach_pl.module import MODULE_REGISTRY


__all__ = ["TrainingModule"]


@MODULE_REGISTRY.register()
class TrainingModule(LightningModule):

    @configurable
    def __init__(self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        cfg: DictConfig,
    ) -> None:
        super().__init__()

        self.model = torch.compile(model) if cfg.MODULE.COMPILE else model
        self.criterion = torch.compile(criterion) if cfg.MODULE.COMPILE else criterion

        self.batch_size = cfg.DATALOADER.BATCH_SIZE

        self.base_lr = cfg.MODULE.OPTIMIZER.BASE_LR
        self.optimizer_name = cfg.MODULE.OPTIMIZER.NAME
        self.optimizer_params = {k.lower(): v for k, v in cfg.MODULE.OPTIMIZER.PARAMS.items()}

        self.scheduler_name = cfg.MODULE.SCHEDULER.NAME
        self.scheduler_params = {k.lower(): v for k, v in cfg.MODULE.SCHEDULER.PARAMS.items()}
        self.scheduler_params["num_epochs"] = cfg.TRAINER.MAX_EPOCHS
        self.step_on_epochs = cfg.MODULE.SCHEDULER.STEP_ON_EPOCHS

        self.save_hyperparameters(ignore=["model", "criterion"])

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "model": build_model(cfg),
            "criterion": build_criterion(cfg),
            "cfg": cfg,
        }

    def configure_optimizers(self) -> Any:
        total_batch_size = self.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size
        lr = self.base_lr * total_batch_size / 256
        print(f"Total training batch size ({total_batch_size}) = single batch size ({self.batch_size}) * accumulate ({self.trainer.accumulate_grad_batches}) * world size ({self.trainer.world_size}), actural learning rate: {lr}")

        optimizer = optim_factory.create_optimizer_v2(
            self.model,
            opt=self.optimizer_name,
            lr=lr,
            **self.optimizer_params            
        )

        scheduler, _ = scheduler_factory.create_scheduler_v2(
            optimizer,
            sched=self.scheduler_name,
            step_on_epochs=self.step_on_epochs,
            updates_per_epoch=int(len(self.trainer.datamodule.train_dataloader()) / (self.trainer.world_size * self.trainer.accumulate_grad_batches)),
            **self.scheduler_params
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if self.step_on_epochs else "step",
                "frequency": 1,
            }
        }

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        if self.step_on_epochs:
            scheduler.step(self.current_epoch, metric)
        else:
            scheduler.step_update(self.global_step, metric)

    def forward(self, batch: Any) -> torch.Tensor:
        images, lables = batch
        noises = torch.randn_like(images)
        timesteps = self.criterion.sample_timesteps(images)

        noisy_images, scales, simgas = self.criterion.add_noise(images, noises, timesteps)
        outputs = self.model(noisy_images, scales, simgas, lables)
        loss = self.criterion(outputs, images, noises, timesteps)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=False, rank_zero_only=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("val/loss", loss, sync_dist=False, rank_zero_only=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("test/loss", loss, sync_dist=False, rank_zero_only=True)
        return loss
