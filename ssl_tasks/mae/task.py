#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from torch import Tensor

from ..contrastive import ContrastiveAugmentation
from ..tokens import TokenMask


class MAE(Task):
    def __init__(
        self,
        backbone: str,
        mask_ratio: float = 0.4,
        mask_scale: int = 2,
        augment_batches: int = 4,
        loss_includes_unmasked: bool = True,
    ):
        super().__init__(backbone)
        self.mask_ratio = mask_ratio
        self.mask_scale = mask_scale
        assert self.mask_ratio > 0
        self.loss_includes_unmasked = loss_includes_unmasked
        self.transform = ContrastiveAugmentation(self.backbone.img_size, num_batches=augment_batches)
        self.mae_loss = nn.L1Loss()
        self.mae_head = self.backbone.patch.inverse(self.backbone.img_size, self.backbone.dim)

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection(
            {
                "psnr": tm.PeakSignalNoiseRatio(),
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
        output = {}
        x: Tensor = batch["img"]

        # apply augmentation
        x = self.transform(x)

        # generate mask and log images
        N = x.shape[0]
        mask = self.backbone.create_token_mask(N, self.mask_ratio, x.device, self.mask_scale)

        # forward pass
        assert mask is not None
        result = self(x, mask)

        # first calculate loss on unmasked tokens if requested
        if self.loss_includes_unmasked:
            x_mae = mask.apply_to_image(x, None)
            pred_mae = mask.apply_to_image(result["mae"], None)
            loss_unmasked = self.mae_loss(pred_mae, x_mae)
        else:
            loss_unmasked = 0

        # calculate loss on masked tokens
        inv_mask = ~mask
        x_mae = inv_mask.apply_to_image(x, None)
        pred_mae = inv_mask.apply_to_image(result["mae"], None)
        loss_masked = self.mae_loss(pred_mae, x_mae)

        # TODO weight this so masked and unmasked losses contribute equally depending on mask ratio
        output["loss_mae"] = loss_masked + loss_unmasked

        # log image with mask tokens filled by MAE predictions
        with torch.no_grad():
            output["masked"] = mask.apply_to_image(x.clone().detach())
            output["pred"] = mask.apply_to_image(x) + inv_mask.apply_to_image(result["mae"].clip(min=0, max=1))

        # log metrics
        if metrics is not None:
            metrics.update(x_mae, pred_mae)

        return output
