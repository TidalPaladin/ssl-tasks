#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod, abstractproperty
from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from torch import Tensor

from ..tokens import TokenMask
from .augmentation import ContrastiveAugmentation
from .loss import PointwiseContrastiveEmbeddingLoss


class ContrastiveEmbedding(Task):
    def __init__(
        self,
        backbone: str,
        mask_ratio: float = 0.4,
        mask_freq: float = 1.0,
        mask_scale: int = 2,
        augment_batches: int = 4,
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
        self.mask_freq = mask_freq
        self.mask_scale = mask_scale

        self.backbone = self.prepare_backbone(backbone)
        self.embed_head = self.create_head()

        # NOTE: The img_size property may depend on self.backbone so assign it after self.backbone
        self.transform = ContrastiveAugmentation(self.img_size, num_batches=augment_batches)
        self.pointwise_embed_loss = PointwiseContrastiveEmbeddingLoss(cartesian=True)

    @abstractmethod
    def prepare_backbone(self, backbone: str) -> nn.Module:
        raise NotImplementedError  # pragma: no cover

    @abstractproperty
    def img_size(self) -> Tuple[int, int]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def create_head(self) -> nn.Module:
        r"""Creates the MAE head for the model"""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
        r"""Creates the MAE head for the model"""
        raise NotImplementedError  # pragma: no cover

    def create_metrics(self, *args, **kwargs) -> tm.MetricCollection:
        return tm.MetricCollection({})

    @abstractmethod
    def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
        raise NotImplementedError  # pragma: no cover

    @property
    def should_mask(self) -> bool:
        return self.training and bool(self.mask_ratio) and (float(torch.rand(1).item()) <= self.mask_freq)

    def prepare_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # generate augmented view, which may be global or local crop
        x_aug = self.transform(x)

        # generate global crop
        x = self.transform(x, global_crop=True)

        return x, x_aug

    def compute_loss(
        self, embeddings: Tensor, aug_embeddings: Tensor, target_embeddings: Optional[Tensor] = None
    ) -> Tensor:
        if target_embeddings is None:
            primary_loss_matrix = self.pointwise_embed_loss(embeddings, aug_embeddings)
        else:
            primary_loss_matrix = (
                +self.pointwise_embed_loss(embeddings, target_embeddings)
                + self.pointwise_embed_loss(aug_embeddings, target_embeddings)
                + self.pointwise_embed_loss(embeddings, aug_embeddings)
            ) / 3
        return primary_loss_matrix

    def step(
        self, batch: Any, batch_idx: int, state: State, metrics: Optional[tm.MetricCollection] = None
    ) -> Dict[str, Any]:
        x = batch["img"]

        # only use 1/3 the batch since we have to send 3x the inputs
        x = x[: len(x) // 3]

        # generate mask if requested
        N = x.shape[0]
        mask = self.create_token_mask(N, x.device)

        # prepare the two input views and mask
        x, x_aug = self.prepare_inputs(x)

        # log images
        batch["img_aug1"] = x
        batch["img_aug2"] = x_aug

        # NOTE: it is empirically very important that the loss be constructed from 3 inputs, all of which require grad
        # 1. A global crop with no masking
        # 2. The same image as in 1. but with masking applied
        # 3. A local crop without masking
        target = self(x)
        aug_result = self(x_aug)
        result = self(x, mask)

        # compute losses coming from mean of encoded tokens and output of decoder with EMB token
        loss = self.compute_loss(result["embed"], aug_result["embed"], target["embed"])

        sim_loss = loss.diag().sum()
        total_loss = loss.sum()
        output = {
            "embed": target["embed"],
            "x_aug": x_aug.detach(),
            "x": x.detach(),
            "log": {
                "loss_embed": total_loss,
                "embed_sim": sim_loss.detach(),
                "embed_dissim": (total_loss - sim_loss).detach(),
            },
        }

        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return {
            "embed": pred["embed"],
        }
