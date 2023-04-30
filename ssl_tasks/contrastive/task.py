#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, ClassVar, Dict, List, Optional, Tuple, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.tasks import Task
from torch import Tensor

from ..tokens import TokenMask
from .loss import PointwiseContrastiveEmbeddingLoss
from .augmentation import ContrastiveAugmentation
from ..helpers import update


class ContrastiveEmbedding(Task):

    def __init__(
        self,
        backbone: str,
        mask_ratio: float = 0.4,
        mask_freq: float = 1.0,
        mask_scale: int = 2,
        augment_batches: int = 4,
        linear_probes: List[str] = [],
    ):
        super().__init__(backbone)
        self.mask_ratio = mask_ratio
        self.mask_freq = mask_freq
        self.mask_scale = mask_scale
        self.transform = ContrastiveAugmentation(self.backbone.img_size, num_batches=augment_batches)

        self.pointwise_embed_loss = PointwiseContrastiveEmbeddingLoss(cartesian=True)
        self.embed_head = nn.Sequential(
            nn.Linear(self.backbone.dim, self.backbone.dim),
            nn.LayerNorm(self.backbone.dim),
        )

        _linear_probes = [
            cast(LinearProbing, TASK_REGISTRY.get(t).instantiate_with_metadata(backbone=backbone).fn)
            for t in linear_probes
        ]
        self._linear_probes = nn.ModuleDict({p.name: p for p in _linear_probes})
        assert all(isinstance(l, LinearProbing) for l in self._linear_probes.values())

    @property
    def should_mask(self) -> bool:
        return self.training and bool(self.mask_ratio) and (float(torch.rand(1).item()) <= self.mask_freq)

    @property
    def linear_probes(self) -> Dict[str, "LinearProbing"]:
        return cast(Dict, self._linear_probes)

    @property
    def num_probes(self) -> int:
        return len(self.linear_probes)

    def setup(self, stage: str):
        for task in self.linear_probes.values():
            task.trainer = self.trainer
            task.setup(stage)

    def forward(
        self,
        x: Tensor,
        mask: Optional[TokenMask] = None,
    ) -> Dict[str, Tensor]:
        N = x.shape[0]
        x = self.backbone(x, mask)
        tokens = self.backbone.tokens(["EMB"], batch_size=N)
        tokens = self.backbone.forward_pool(x, tokens)
        embed = self.embed_head(tokens[..., 0, :])
        result = {"embed": embed}
        return result

    @torch.no_grad()
    def _process_attn_matrix(self, img: Tensor, size: int, head: Optional[int] = None) -> Dict[str, Tensor]:
        attn: Tensor = self.backbone.attn_pool.attn.attention_matrix
        if attn is None:
            return {}

        # mean over heads
        if head is None:
            attn = attn.mean(dim=1)
        else:
            attn = attn[:, head, ...]

        def minmax(x: Tensor) -> Tensor:
            min = x.amin(dim=(-1, -2, -3), keepdim=True)
            max = x.amax(dim=(-1, -2, -3), keepdim=True)
            x = (x - min) / (max - min).clip_(1e-8)
            return x

        def process(x: Tensor) -> Tensor:
            x = x.view(-1, 1, *self.backbone.tokenized_image_size)
            x = minmax(x)
            x = F.interpolate(x, size=self.backbone.img_size, mode="nearest")
            x = img * x
            x = minmax(x)
            return x

        result = {k: process(v) for k, v in self.backbone.tokens.to_dict(attn, ["EMB"]).items()}
        return result

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

    def step(self, batch: Any, batch_idx: int, metrics: Optional[tm.MetricCollection] = None) -> Dict[str, Any]:
        output = {}
        output["log"] = {}
        output["log_hist"] = {}
        output["log_embed"] = {}
        x, y = batch
        # only use half the batch since we have to send double the inputs
        x = x[::2]
        y = y[::2]

        # generate mask if requested
        N = x.shape[0]
        mask = self.backbone.create_token_mask(N, self.mask_ratio, x.device, self.mask_scale)

        # prepare the two input views and mask
        x, x_aug = self.prepare_inputs(x)

        # log images
        output["log_img"] = {
            "img_aug1": x,
            "img_aug2": x_aug,
        }

        # NOTE: it is empirically very important that the loss be constructed from 3 inputs, all of which require grad
        # 1. A global crop with no masking
        # 2. The same image as in 1. but with masking applied
        # 3. A local crop without masking
        target = self(x)
        aug_result = self(x_aug)
        result = self(x, mask)

        # compute losses coming from mean of encoded tokens and output of decoder with EMB token
        loss = self.compute_loss(result["embed"], aug_result["embed"], target["embed"])

        # log metrics
        output["log"]["loss_embed"] = loss.sum()
        output["log"]["embed_sim"] = loss.diag().sum()
        output["log"]["embed_dissim"] = loss.sum() - output["log"]["embed_sim"]

        # log attention matrix
        attn = self._process_attn_matrix(x, size=1).get("EMB", None)
        if attn is not None:
            output["log_img"]["attn"] = attn

        # step attached linear probes
        output["linear_probes"] = self.step_linear_probes(target["embed"], y, batch_idx)

        # move linear probe losses to the correct level so they are recorded
        for name, probe_output in output["linear_probes"].items():
            update(output, probe_output)

        # double check we have a loss for the embedding token and 1 loss for each probe
        losses = {k: v for k, v in output["log"].items() if k.startswith("loss")}
        assert len(losses) == self.num_probes + 1

        return output

    def step_linear_probes(self, embed: Tensor, true: Tensor, batch_idx: int) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for name, task in self.linear_probes.items():
            output[name] = task.step((embed, true), batch_idx, task.get_metrics())
        return output

    def run_logging_loop(
        self, logger: pl.LightningModule, output: Dict[str, Any], batch_idx: int, sync_dist: bool = True
    ) -> None:
        super().run_logging_loop(logger, output, batch_idx, sync_dist)
        for name, task in self.linear_probes.items():
            task_output = output["linear_probes"][name]
            task.run_logging_loop(logger, task_output, batch_idx, sync_dist)


class LinearProbing(Task):
    def __init__(
        self,
        backbone: str,
        probe_name: str,
        num_classes: int,
        detach: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__(backbone)
        self.name = probe_name
        self.num_classes = num_classes
        self.detach = detach
        self.cls_loss = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.backbone.dim, 1 if self.is_binary else self.num_classes)
        )

        acc = tm.Accuracy(task="binary") if self.is_binary else tm.Accuracy(num_classes=num_classes)
        metrics = tm.MetricCollection({f"{self.name}/acc": acc})
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        del self.backbone

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 2

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {f"cls_{self.name}": self.cls_head(x)}

    def step(self, batch: Any, batch_idx: int, metrics: Optional[tm.MetricCollection] = None) -> Dict[str, Any]:
        output = {}
        output["log"] = {}
        output["log_hist"] = {}
        output["log_embed"] = {}
        x, y = batch

        # it is assumed that x is the constrastive embedding token
        if self.detach:
            x = x.detach()
        result = self(x)

        # compute loss
        pred = next(iter(result.values()))
        output["log"][f"loss_probe_{self.name}"] = self.cls_loss(pred, y)

        # log metrics
        if metrics is not None:
            pred = pred.sigmoid() if self.is_binary else pred.argmax(dim=-1)
            metrics[f"{self.name}/acc"].update(pred, y.long())

        return output
