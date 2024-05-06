from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from torch import Tensor

from ..tokens import TokenMask


class JEPA(Task, ABC):
    def __init__(
        self,
        backbone: str,
        mask_ratio: float = 0.4,
        mask_scale: int = 2,
        loss_includes_unmasked: bool = False,
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_interval: str = "epoch",
        lr_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        weight_decay_exemptions: Set[str] = set(),
    ):
        super().__init__(
            optimizer_init,
            lr_scheduler_init,
            lr_interval,
            lr_monitor,
            named_datasets,
            checkpoint,
            strict_checkpoint,
            log_train_metrics_interval,
            log_train_metrics_on_epoch,
            weight_decay_exemptions,
        )

        self.mask_ratio = mask_ratio
        self.mask_scale = mask_scale
        assert self.mask_ratio > 0
        self.loss_includes_unmasked = loss_includes_unmasked

        self.backbone = self.prepare_backbone(backbone)
        self.jepa_head = self.create_head()

        self.jepa_loss = nn.MSELoss()

    @abstractmethod
    def create_head(self) -> nn.Module:
        r"""Creates the head for the model"""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def create_token_mask(self, x: Tensor) -> TokenMask:
        r"""Creates the token mask"""
        raise NotImplementedError  # pragma: no cover

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection({})

    def forward(
        self,
        x: Tensor,
        mask: Optional[TokenMask] = None,
    ) -> Dict[str, Tensor]:
        x = self.backbone(x, mask)
        return {"jepa": self.jepa_head(x)}

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        x: Tensor = batch["img"]

        # generate mask and log images
        mask = self.create_token_mask(x)

        # generate ground truth with forward pass of unmasked image
        with torch.no_grad():
            target: Tensor = self(x)["jepa"]

        # generate predictions in latent space using masked image
        assert mask is not None
        result: Tensor = self(x, mask)["jepa"]

        # first calculate loss on unmasked tokens if requested
        if self.loss_includes_unmasked:
            y = mask.apply_to_tokens(target, None)
            pred_jepa = mask.apply_to_tokens(result, None)
            loss_unmasked = self.jepa_loss(pred_jepa, y)
        else:
            loss_unmasked = 0

        # calculate loss on masked tokens
        inv_mask = ~mask
        y = inv_mask.apply_to_tokens(target, None)
        pred_jepa = inv_mask.apply_to_tokens(result, None)
        loss_masked = self.jepa_loss(pred_jepa, y)

        # Compute total loss
        loss = loss_masked + loss_unmasked

        with torch.no_grad():
            masked_img = mask.apply_to_input(x.clone().detach())
            jepa_true = target.clone().detach()
            jepa_pred = result.clone().detach()

        output = {
            "masked": masked_img,
            "jepa_pred": jepa_pred,
            "jepa_true": jepa_true,
            "log": {
                "loss_jepa": loss,
            },
        }

        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return {
            "jepa": pred["jepa"],
        }
