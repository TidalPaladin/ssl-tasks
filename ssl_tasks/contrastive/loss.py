#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ContrastiveEmbeddingLoss(nn.CosineEmbeddingLoss):
    r"""Embedding loss wrapper over cosine similarity that accepts two pairs of inputs and a class membership
    for each input pair. The loss aims to maximize the similarity between the two inputs of the same class
    and minimize the similarity between the two inputs of different classes.

    Args:
        cartesian: If True, compute the loss for all pairs of inputs. If False, compute the loss for each pair of
            inputs separately.
        margin: Margin value for the loss function. Default: 0.
    """

    def __init__(self, cartesian: bool = True, margin: float = 0):
        super().__init__(margin=margin, reduction="none")
        self.cartesian = cartesian

    def forward(self, x: Tensor, x_aug: Tensor, cls: Tensor, cls_aug: Tensor) -> Tensor:
        if self.cartesian:
            return self._forward_cartesian(x, x_aug, cls, cls_aug)
        else:
            return self._forward(x, x_aug, cls, cls_aug)

    def _forward_cartesian(self, x: Tensor, x_aug: Tensor, cls: Tensor, cls_aug: Tensor) -> Tensor:
        N, D = x.shape
        x = x.view(1, N, D).expand(N, -1, -1)
        x_aug = x_aug.view(N, 1, D).expand(-1, N, -1)
        cls = cls.view(1, N).expand(N, -1)
        cls_aug = cls_aug.view(N, 1).expand(-1, N)
        result = self._forward(x.reshape(-1, D), x_aug.reshape(-1, D), cls.flatten(), cls_aug.flatten())
        return result.view(N, N)

    def _forward(self, x: Tensor, x_aug: Tensor, cls: Tensor, cls_aug: Tensor) -> Tensor:
        assert x.shape[0] == x_aug.shape[0] == cls.shape[0] == cls_aug.shape[0]
        N, D = x.shape
        polarity = torch.where(cls == cls_aug, 1, -1).long()
        scale = self.compute_scale(polarity)
        loss = super().forward(x.view(-1, D), x_aug.view(-1, D), polarity.view(-1))
        return loss * scale

    @torch.no_grad()
    def compute_scale(self, polarity: Tensor) -> Tensor:
        N = polarity.numel()
        num_pos = (polarity == 1).sum()
        num_neg = N - num_pos
        return torch.where(polarity == 1, 1 / num_pos, 1 / num_neg)


class PointwiseContrastiveEmbeddingLoss(ContrastiveEmbeddingLoss):
    r"""A simplification of :class:`ContrastiveEmbeddingLoss` that assumes that the inputs are paired
    pointwise, i.e. the first input is the original and the second input is the augmented version of the same sample.
    """

    def forward(self, x: Tensor, x_aug: Tensor) -> Tensor:
        cls, cls_aug = self.build_pointwise_class_targets(x)
        return super().forward(x, x_aug, cls, cls_aug)

    @torch.no_grad()
    def build_pointwise_class_targets(self, pred: Tensor) -> Tuple[Tensor, Tensor]:
        N = pred.shape[0]
        t = torch.arange(N).to(pred.device)
        return t, t
