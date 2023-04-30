#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor


class SmallBoxCrop(nn.Module):
    r"""Extracts a small augmented box crop.

    Returns:
        Tuple of box crop, box coordinates
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        scale: Tuple[float, float] = (0.1, 0.4),
        ratio: Tuple[float, float] = (0.8, 1.2),
        augment_color: bool = True,
        augment_position: bool = False,
        num_batches: int = 1,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_batches = num_batches
        self.scale = scale
        self.ratio = ratio
        self.augment_color = augment_color
        self.augment_position = augment_position

        self.color_augment = nn.Sequential(
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.2),
            T.RandomInvert(0.1),
        )

        self.position_augment = nn.Sequential(
            T.RandomInvert(0.1),
            T.RandomVerticalFlip(0.5),
            T.RandomHorizontalFlip(0.5),
        )

    def get_crop_size(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        scale_min, scale_max = self.scale
        ratio_min, ratio_max = self.ratio

        # choose width freely based on scale
        width = x.new_empty(1, device="cpu").uniform_(scale_min, scale_max).mul_(W)

        # choose height based on ratio
        height = width * torch.empty_like(width).uniform_(ratio_min, ratio_max)

        result = torch.cat([width, height]).long()
        assert result.device == torch.device("cpu")
        assert not result.is_floating_point()
        return result

    def get_crop_center(self, x: Tensor, crop_size: Tensor) -> Tensor:
        img_size = x.new_tensor(x.shape[-2:], dtype=torch.long, device="cpu")
        # choose a center that doesn't let the crop size overflow image
        lower_bounds = crop_size.div(2).ceil_()
        upper_bounds = img_size - crop_size.div(2, rounding_mode="floor")

        h_min, h_max = float(lower_bounds[0].item()), float(upper_bounds[0].item())
        delta_h = x.new_empty(1, device="cpu").uniform_(h_min, h_max).long()

        w_min, w_max = float(lower_bounds[1].item()), float(upper_bounds[1].item())
        delta_w = x.new_empty(1, device="cpu").uniform_(w_min, w_max).long()

        result = torch.cat([delta_w, delta_h]).long()
        assert result.device == torch.device("cpu")
        assert not result.is_floating_point()
        return result

    def get_crop_bounds(self, img_size: Tuple[int, int], crop_size: Tensor) -> Tensor:
        # assumes image has aleady been translated such that the center of the image is the crop center
        H, W = img_size
        crop_width, crop_height = crop_size.tolist()
        top = H // 2 - crop_height // 2
        left = W // 2 - crop_width // 2
        return crop_size.new_tensor([top, left, crop_width, crop_height])

    def shift_and_crop(self, x: Tensor, rotate: float = 90) -> Tuple[Tensor, Tensor]:
        H, W = x.shape[-2:]
        crop_size = self.get_crop_size(x)
        crop_center = self.get_crop_center(x, crop_size)
        translation = crop_center.new_tensor([W // 2, H // 2]) - crop_center

        # first translate to the target location
        x = TF.affine(
            x,
            angle=0.0,
            translate=[int(t) for t in translation],
            scale=1.0,
            shear=[0.0, 0.0],
        )

        # rotate before crop to avoid fill
        if self.augment_position:
            x = TF.rotate(x, rotate)

        # apply the crop
        top, left, crop_width, crop_height = self.get_crop_bounds((H, W), crop_size).tolist()
        x = x[..., top : top + crop_height, left : left + crop_width]

        # resize crop back to model size
        x = F.interpolate(x, self.img_size, mode="bilinear")

        # build a box of form cx, cy, dx, dy
        box = torch.cat([crop_center, crop_size])
        return x, box

    def resize_box(self, box: Tensor, img_size: Tuple[int, int], resized_img_size: Tuple[int, int]) -> Tensor:
        H, W = img_size
        Hr, Wr = resized_img_size
        scale = box.new_tensor([Wr, Hr, Wr, Hr]) / box.new_tensor([W, H, W, H])
        return box.mul(scale).long()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.num_batches > 1:
            results = [self._batch_forward(t) for t in x.chunk(self.num_batches)]
            x = torch.cat([r[0] for r in results], dim=0)
            box = torch.cat([r[1] for r in results], dim=0)
            return x, box
        else:
            return self._batch_forward(x)

    def _batch_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            # shift the full size image and generate a local crop with coords
            H, W = x.shape[-2:]
            x, box = self.shift_and_crop(x)

            # resize the box to convert coords from original img size to resized img size
            box = self.resize_box(box, (H, W), self.img_size)
            N = x.shape[0]
            box = box.view(1, 1, -1).expand(N, -1, -1)

            # augment the local crop
            if self.augment_color:
                x = self.color_augment(x)
            if self.augment_position:
                x = self.position_augment(x)

        x = x.to(dtype)
        box = box.to(device=x.device, dtype=dtype)
        return x, box
