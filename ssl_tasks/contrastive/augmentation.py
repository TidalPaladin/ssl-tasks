#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor


class ContrastiveAugmentation(nn.Module):
    """Module to perform contrastive image augmentation"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        num_batches: int = 1,
    ) -> None:
        super().__init__()
        self.num_batches = num_batches
        self.resize = T.Resize(img_size)
        self.global_crop = T.RandomResizedCrop(img_size, scale=(0.5, 1.0))
        self.local_crop = T.RandomResizedCrop(img_size, scale=(0.2, 0.4))

        self.constrastive_resize = T.RandomChoice(
            [
                self.global_crop,
                self.local_crop,
            ]
        )
        self.rotate = T.RandomRotation(degrees=90)

        # TODO parameterize these augmentations
        self.color_augment = nn.Sequential(
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.2),
            T.RandomInvert(0.1),
        )

        self.position_augment = nn.Sequential(
            T.RandomVerticalFlip(0.5),
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(1.0, 1.75), shear=(-15, 15)),
        )

    @torch.no_grad()
    def forward_position(self, x: Tensor) -> Tensor:
        if self.num_batches > 1:
            return torch.cat([self._forward_position(t) for t in x.chunk(self.num_batches)], 0)
        else:
            return self._forward_position(x)

    def _forward_position(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.position_augment(x)
            x = self.rotate(x)
        return x

    @torch.no_grad()
    def forward_color(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            if self.num_batches > 1:
                return torch.cat([self.color_augment(t) for t in x.chunk(self.num_batches)], 0)
            else:
                return self.color_augment(x)

    @torch.no_grad()
    def forward(self, x: Tensor, global_crop: Optional[bool] = None) -> Tensor:
        if self.num_batches > 1:
            return torch.cat([self._batch_forward(t, global_crop) for t in x.chunk(self.num_batches)], 0)
        else:
            return self._batch_forward(x, global_crop)

    def _batch_forward(self, x: Tensor, global_crop: Optional[bool]) -> Tensor:
        dtype = x.dtype
        global_crop = (
            global_crop
            if global_crop is not None
            else (
                float(
                    torch.rand(
                        1,
                    ).item()
                )
                < 0.5
            )
        )
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.rotate(x)
            crop = self.global_crop if global_crop else self.local_crop
            x = crop(x)
            x = self.color_augment(x)
            x = self.position_augment(x)
        x = x.to(dtype)
        return x
