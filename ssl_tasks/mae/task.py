#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from torch import Tensor
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from ..tokens import TokenMask


class MAE(Task, ABC):
    r"""Masked Autoencoder task.

    Args:
        backbone: The name of the backbone model.
        mask_ratio: The ratio of the input to mask.
        mask_scale: The scale of the mask block size.
        loss_includes_unmasked: Whether the loss includes unmasked tokens.
        optimizer_init: The initialization parameters for the optimizer.
        lr_scheduler_init: The initialization parameters for the learning rate scheduler.
        lr_interval: The interval for the learning rate update.
        lr_monitor: The metric to monitor for learning rate update.
        named_datasets: Whether to use named datasets.
        checkpoint: The path to the checkpoint file.
        strict_checkpoint: Whether to strictly enforce the checkpoint.
        log_train_metrics_interval: The interval for logging training metrics.
        log_train_metrics_on_epoch: Whether to log training metrics on epoch end.
        weight_decay_exemptions: The set of parameters to exempt from weight decay.
    """

    def __init__(
        self,
        backbone: str,
        mask_ratio: float = 0.4,
        mask_scale: int = 2,
        loss_includes_unmasked: bool = True,
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
        self.mae_head = self.create_head()
        self.mae_loss = nn.L1Loss()

    @abstractmethod
    def prepare_backbone(self, backbone: str) -> nn.Module:
        """
        Prepares the backbone for the model.

        Args:
            backbone (str): The name of the backbone to prepare.

        Returns:
            nn.Module: The prepared backbone module.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def create_head(self) -> nn.Module:
        r"""Creates the MAE head for the model"""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def create_token_mask(self, x: Tensor) -> TokenMask:
        r"""Create a token mask for an input."""
        raise NotImplementedError  # pragma: no cover

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(),
            }
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[TokenMask] = None,
    ) -> Dict[str, Tensor]:
        x = self.backbone(x, mask)
        return {"mae": self.mae_head(x)}

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

        # forward pass
        assert mask is not None
        result = self(x, mask)

        # first calculate loss on unmasked tokens if requested
        if self.loss_includes_unmasked:
            x_mae = mask.apply_to_input(x, 0)
            pred_mae = mask.apply_to_input(result["mae"], 0)
            loss_unmasked = self.mae_loss(pred_mae, x_mae)
        else:
            loss_unmasked = 0

        # calculate loss on masked tokens
        inv_mask = ~mask
        x_mae = inv_mask.apply_to_input(x, 0)
        pred_mae = inv_mask.apply_to_input(result["mae"], 0)
        loss_masked = self.mae_loss(pred_mae, x_mae)

        # TODO weight this so masked and unmasked losses contribute equally depending on mask ratio
        loss = loss_masked + loss_unmasked

        # log metrics
        if metrics is not None:
            metrics.update(x_mae, pred_mae)

        # log image with mask tokens filled by MAE predictions
        with torch.no_grad():
            masked_img = mask.apply_to_input(x.clone().detach())
            pred_img = mask.apply_to_input(x) + inv_mask.apply_to_input(result["mae"].clip(min=0, max=1))

        output = {
            "masked": masked_img,
            "mae_pred": pred_img,
            "log": {
                "loss_mae": loss,
            },
        }

        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return {
            "mae": pred["mae"],
        }
