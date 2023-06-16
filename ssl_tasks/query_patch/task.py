#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod, abstractproperty
from typing import Any, Dict, Optional, Set, Tuple, cast

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from kornia.geometry.transform import crop_by_indices
from torch import Tensor
from torchmetrics import MetricCollection
from torchvision.ops import box_iou

from ..contrastive import ContrastiveAugmentation
from .augmentation import SmallBoxCrop


class BoxIOU(tm.Metric):
    full_state_update: bool = False
    iou: Tensor
    total: Tensor

    def __init__(self):
        super().__init__()
        self.add_state("iou", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @classmethod
    def convert_coords(cls, boxes: Tensor) -> Tensor:
        center = boxes[..., :2]
        wh = boxes[..., -2:]
        mins = center - wh.div(2, rounding_mode="floor")
        maxes = center + wh.div(2, rounding_mode="floor")
        return torch.cat([mins, maxes], dim=-1)

    def update(self, preds: Tensor, target: Tensor):
        assert preds.shape == target.shape

        # this method computes iou over cartesian product pairs
        # take the diagonal to ignore irrelevant boxes
        iou = box_iou(preds.view(-1, 4), target.view(-1, 4)).diag().sum()
        self.iou += iou.type_as(self.iou)
        self.total += target.numel() // 4

    def compute(self):
        return self.iou / self.total


class QueryPatch(Task):
    def __init__(
        self,
        backbone: str,
        augment_batches: int = 4,
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: str = "train/total_loss_epoch",
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
            lr_scheduler_interval,
            lr_scheduler_monitor,
            named_datasets,
            checkpoint,
            strict_checkpoint,
            log_train_metrics_interval,
            log_train_metrics_on_epoch,
            weight_decay_exemptions,
        )

        self.backbone = self.prepare_backbone(backbone)
        self.box_decoder = self.create_head()

        self.transform = ContrastiveAugmentation(self.img_size, num_batches=augment_batches)
        self.box_transform = SmallBoxCrop(self.img_size, num_batches=augment_batches)
        self.box_loss = nn.L1Loss(reduction="none")

    @abstractmethod
    def prepare_backbone(self, backbone: str) -> nn.Module:
        raise NotImplementedError  # pragma: no cover

    @abstractproperty
    def img_size(self) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def create_head(self) -> nn.Module:
        r"""Creates the MAE head for the model"""
        raise NotImplementedError

    def create_metrics(self, state: State) -> MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection(
            {
                "iou": BoxIOU(),
            }
        )

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        x_box: Tensor,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def get_fractional_scale(self, proto: Optional[Tensor] = None) -> Tensor:
        H, W = self.img_size
        if proto is not None:
            return proto.new_tensor([W, H, W, H])
        else:
            return torch.tensor([W, H, W, H])

    def scale_fractional_to_absolute(self, x: Tensor) -> Tensor:
        fractional_scale = self.get_fractional_scale(proto=x)
        return x * fractional_scale

    def scale_absolute_to_fractional(self, x: Tensor) -> Tensor:
        fractional_scale = self.get_fractional_scale(proto=x)
        return x / fractional_scale

    def restrict_bounds(self, x: Tensor) -> Tensor:
        H, W = self.img_size
        upper_bound = x.new_tensor([W, H, W, H])
        return x.clip(min=x.new_tensor(0), max=upper_bound)

    def extract_predicted_crop(self, x: Tensor, box_pred: Tensor) -> Tensor:
        # ensure predicted crop is valid w/ nonzero area
        box_pred_bounded = box_pred.relu()
        box_pred_bounded[..., -2:].clip(min=8)
        pred_xyxy = BoxIOU.convert_coords(self.scale_fractional_to_absolute(box_pred_bounded))
        pred_xyxy = self.restrict_bounds(pred_xyxy)

        # convert to form expected by kornia
        top_left = pred_xyxy[..., :2]
        bottom_right = pred_xyxy[..., -2:]
        top_right = torch.stack([pred_xyxy[..., 2], pred_xyxy[..., 1]], dim=-1)
        bottom_left = torch.stack([pred_xyxy[..., 0], pred_xyxy[..., 3]], dim=-1)
        crop_bounds = torch.cat([top_left, top_right, bottom_right, bottom_left], dim=-2)

        pred_crop = crop_by_indices(x, crop_bounds, size=self.img_size)
        return pred_crop

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        x = batch["img"]

        # apply a positional augmentation before creating the small box crop.
        # once a box crop is made, no position augmentations of global image can be made.
        # here x will be full resolution so we can ensure box crop is of highest possible resolution
        x = self.transform.forward_position(x)

        # generate box crop and box coords.
        # x_box already accounts for the resize we will do to x
        x_box, box = self.box_transform(x)

        # apply color only augmentation and resize
        x = self.transform.forward_color(x)
        x = self.transform.resize(x)

        result = self(x, x_box)

        # loss of box coordinates
        box_xyxy = BoxIOU.convert_coords(box)
        pred_xyxy = BoxIOU.convert_coords(self.scale_fractional_to_absolute(result["box"]))
        box_loss = self.box_loss(
            result["box"],
            self.scale_absolute_to_fractional(box),
        ).mean()

        # create absolute coordinate prediction for visualization
        with torch.no_grad():
            pred_xyxy = self.restrict_bounds(pred_xyxy)
            box_xyxy = self.restrict_bounds(box_xyxy)

        # log metrics
        if metrics is not None:
            cast(tm.Metric, metrics["iou"]).update(pred_xyxy, box_xyxy)

        output = {
            "log": {
                "loss_query_patch": box_loss,
            },
            "query_patch_pred": pred_xyxy,
            "query_patch_target": box_xyxy,
            "x_aug": x.detach(),
        }

        return output
