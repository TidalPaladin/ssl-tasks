#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor


class ContrastiveAugmentation(nn.Module):
    """Module to perform contrastive image augmentation.

    Args:
        img_size: Size of the image after augmentation
        num_batches: Number of unique augmentations to perform per batch.
            If ``num_batches == 1``, then the same augmentation is applied to all
            images in the batch.
        global_scale: Scale range for global crop.
        local_scale: Scale range for local crop.
        rotate_degrees: Rotation range in degrees.
        horizontal_flip: Probability of horizontal flip.
        vertical_flip: Probability of vertical flip.
        brightness: Brightness range.
        contrast: Contrast range.
        saturation: Saturation range.
        hue: Hue range.
        grayscale_prob: Probability of converting image to grayscale.
        invert_prob: Probability of inverting image.
        translate: Translate range.
        scale: Scale range.
        shear: Shear range.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        num_batches: int = 1,
        global_scale: Tuple[float, float] = (0.5, 1.0),
        local_scale: Tuple[float, float] = (0.2, 0.4),
        rotate_degrees: float = 90.0,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
        grayscale_prob: float = 0.2,
        invert_prob: float = 0.1,
        translate: Tuple[float, float] = (0.2, 0.2),
        scale: Tuple[float, float] = (1.0, 1.75),
        shear: Tuple[float, float] = (-15.0, 15.0),
    ) -> None:
        super().__init__()
        self.num_batches = num_batches
        self.resize = T.Resize(img_size)
        self.global_crop = T.RandomResizedCrop(img_size, scale=global_scale, antialias=True)
        self.local_crop = T.RandomResizedCrop(img_size, scale=local_scale, antialias=True)

        self.constrastive_resize = T.RandomChoice(
            [
                self.global_crop,
                self.local_crop,
            ]
        )
        self.rotate = T.RandomRotation(degrees=rotate_degrees)

        # TODO parameterize these augmentations
        self.color_augment = nn.Sequential(
            T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            T.RandomGrayscale(p=grayscale_prob),
            T.RandomInvert(p=invert_prob),
        )

        self.position_augment = nn.Sequential(
            T.RandomVerticalFlip(0.5 if vertical_flip else 0.0),
            T.RandomHorizontalFlip(0.5 if horizontal_flip else 0.0),
            T.RandomAffine(degrees=0, translate=translate, scale=scale, shear=shear),
        )

    @torch.no_grad()
    def forward_position(self, x: Tensor) -> Tensor:
        r"""Performs only contrastive position augmentation."""
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
        r"""Performs only contrastive color augmentation. This is useful for performing
        contrastive color augmentation on a batch of images without modifying their positions.
        """
        with torch.autocast(device_type="cuda", enabled=False):
            if self.num_batches > 1:
                return torch.cat([self.color_augment(t) for t in x.chunk(self.num_batches)], 0)
            else:
                return self.color_augment(x)

    @torch.no_grad()
    def forward(self, x: Tensor, global_crop: Optional[bool] = None) -> Tensor:
        r"""Runs the augmentation.

        Args:
            x: Input image tensor of shape :math:`(N, C, H, W)`.
            global_crop: If ``True``, then global crop is used. If ``False``, then local crop is used.
                If ``None``, then global crop is used with probability 0.5.
        """
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
