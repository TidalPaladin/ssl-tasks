#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchmetrics as tm
from torchmetrics import MetricCollection
from deep_helpers.tasks import Task
from deep_helpers.structs import State
from kornia.geometry.transform import crop_by_indices
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes

from ..contrastive import ContrastiveAugmentation, UniqueEmbeddingLoss
from ..tokens import TokenMask
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
    ):
        super().__init__(backbone)
        self.transform = ContrastiveAugmentation(self.backbone.img_size, num_batches=augment_batches)
        self.box_transform = SmallBoxCrop(self.backbone.img_size, num_batches=augment_batches)
        self.box_loss = nn.L1Loss(reduction="none")
        self.query_loss = UniqueEmbeddingLoss()

        self.box_head = nn.Linear(self.dim, 4)
        nn.init.constant_(self.box_head.bias, 0.5)
        self.box_decoder = nn.ModuleList(self.backbone.DecoderBlock(self_attn=False) for _ in range(3))

    def create_metrics(self, state: State) -> MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection(
            {
                "iou": BoxIOU(),
            }
        )

    def forward_box_decoder(self, x: Tensor, tokens: Tensor) -> Tensor:
        for block in self.box_decoder:
            tokens = block(tokens, x)
        return tokens

    def forward(
        self,
        x: Tensor,
        x_box: Tensor,
        mask: Optional[TokenMask] = None,
    ) -> Dict[str, Tensor]:
        # encode the full image and box crop
        x = self.backbone(x, mask=mask)
        x_box = self.backbone(x_box, mask=mask)

        # run attention pooling to get a query representing the box crop
        N = x.shape[0]
        tokens = self.backbone.tokens(["BOX_Q"], batch_size=N)
        tokens = self.backbone.forward_pool(x_box, tokens)

        # run attention pooling to get a query representing the box crop
        tokens = self.forward_box_decoder(x, tokens)

        # get box coords
        box_pred = self.box_head(tokens[..., -1, None, :])
        assert box_pred.shape[-1] == 4

        result = {
            "box": box_pred,
        }
        return result

    def get_fractional_scale(self, proto: Optional[Tensor] = None) -> Tensor:
        H, W = self.backbone.img_size
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
        H, W = self.backbone.img_size
        upper_bound = x.new_tensor([W, H, W, H])
        return x.clip(min=x.new_tensor(0), max=upper_bound)

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError  # pragma: no cover

    def step(self, batch: Any, batch_idx: int, metrics: Optional[tm.MetricCollection] = None) -> Dict[str, Any]:
        output = {}
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
        # box_loss = box_loss.mean(dim=-1).mul(weights).sum()
        output["log"]["loss_box"] = box_loss

        # create absolute coordinate prediction for visualization
        with torch.no_grad():
            pred_xyxy = self.restrict_bounds(pred_xyxy)
            output["log_boxes"] = {}
            output["log_boxes"]["box_pred"] = (x, pred_xyxy, box_xyxy)

        # log metrics
        if metrics is not None:
            metrics["iou"].update(pred_xyxy, box_xyxy)

        return output

    @rank_zero_only
    def _log_boxes(self, prefix: str, log: Dict[str, Any], experiment: Any) -> None:
        with torch.no_grad():
            for k, (img, box_pred, box) in log.items():
                if img.dtype != torch.uint8:
                    img = img.mul(255).byte()
                img = self._draw_bounding_boxes(img, box_pred, colors="blue")
                img = self._draw_bounding_boxes(img, box, colors="red")
                experiment.add_images(f"{prefix}/{k}", img, global_step=self.global_step)

    def _convert_coords(self, boxes: Tensor) -> Tensor:
        center = boxes[..., :2]
        wh = boxes[..., -2:]
        mins = center - wh.div(2, rounding_mode="floor")
        maxes = center + wh.div(2, rounding_mode="floor")
        return torch.cat([mins, maxes], dim=-1)

    @torch.no_grad()
    def _draw_bounding_boxes(self, x: Tensor, boxes: Tensor, **kwargs) -> Tensor:
        assert boxes.shape[-1] == 4
        if x.ndim == 4:
            assert boxes.ndim == 3
            results = []
            for x_batch, box_batch in zip(x, boxes):
                t = self._draw_bounding_boxes(x_batch, box_batch, **kwargs)
                results.append(t)
            return torch.stack(results, 0)

        H, W = x.shape[-2:]
        boxes[..., :2].clip_(min=0)
        boxes[..., 2:].clip_(max=boxes.new_tensor([W, H]))
        return draw_bounding_boxes(x, boxes, **kwargs)

    def _log_extras(self, prefix: str, output: Dict[str, Any], experiment: Any):
        super()._log_extras(prefix, output, experiment)
        self._log_boxes(prefix, output.get("log_boxes", {}), experiment)

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

        pred_crop = crop_by_indices(x, crop_bounds, size=self.backbone.img_size)
        return pred_crop
