#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .helpers import divide_tuple


@dataclass
class TokenMask:
    mask: Tensor
    img_size: Tuple[int, int]
    patch_size: Tuple[int, int]

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += "mask={tuple(self.mask.shape)}, "
        s += "img_size={self.img_size}, "
        s += "patch_size={self.patch_size}"
        s += ")"
        return s

    def __invert__(self) -> "TokenMask":
        return replace(self, mask=~self.mask)

    @property
    def shape(self) -> torch.Size:
        return self.mask.shape

    @property
    def tokenized_image_size(self) -> Tuple[int, int]:
        return divide_tuple(self.img_size, self.patch_size)

    def apply_to_image(self, x: Tensor, fill_value: Optional[float] = 0) -> Tensor:
        N, C, H, W = x.shape
        Hp, Wp = self.patch_size
        Ht, Wt = self.tokenized_image_size
        tokenized_img = rearrange(x, "n c (ht hp) (wt wp) -> n (ht wt) (hp wp c)", hp=Hp, wp=Wp)
        if fill_value is not None:
            tokenized_img[~self.mask] = fill_value
            return rearrange(
                tokenized_img, "n (ht wt) (hp wp c) -> n c (ht hp) (wt wp)", n=N, hp=Hp, wp=Wp, ht=Ht, wt=Wt
            )
        else:
            return tokenized_img[self.mask]

    def apply_to_tokens(
        self, x: Tensor, fill_value: Optional[Union[float, Tensor]] = None, inverse: bool = False
    ) -> Tensor:
        N = x.shape[0]
        if fill_value is None:
            m = ~self.mask if inverse else self.mask
            x = rearrange(x[m], "(n l) c -> n l c", n=N)
        else:
            x[self.mask] = fill_value.type_as(x) if isinstance(fill_value, Tensor) else fill_value
        return x

    def restore_tokens(self, x: Tensor, fill_value: float = 0) -> Tensor:
        Ht, Wt = self.tokenized_image_size
        N, L, D = x.shape
        output = x.new_full((N, Ht * Wt, D), fill_value)
        output[self.mask] = x.view(-1, D)
        return output

    @classmethod
    def create(
        cls,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        batch_size: int = 1,
        mask_ratio: float = 0.25,
        scale: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> "TokenMask":
        if not 0 < mask_ratio < 1.0:
            raise ValueError(f"Invalid `mask_ratio` {mask_ratio}")

        Ht, Wt = divide_tuple(img_size, patch_size)
        Ht, Wt = divide_tuple((Ht, Wt), (scale, scale))
        Lmask = Ht * Wt
        num_masked_tokens = int(round(Lmask * mask_ratio))

        # initialize empty mask
        mask = torch.full((batch_size, Lmask), True, device=device, dtype=torch.bool)

        # select exactly num_masked_tokens random locations, with unique locations for each batch element
        token_idx = torch.randperm(Lmask).view(1, Lmask).expand(batch_size, -1)
        indices = torch.argsort(torch.rand_like(token_idx, dtype=torch.float32), dim=-1)[..., :num_masked_tokens]
        token_idx = torch.gather(token_idx, dim=-1, index=indices)
        assert token_idx.shape == (batch_size, num_masked_tokens)
        batch_idx = torch.arange(batch_size).view(batch_size, 1).expand(-1, num_masked_tokens)

        # update mask based on chosen locations
        mask[batch_idx.flatten(), token_idx.flatten()] = False

        if scale > 1:
            mask = mask.view(batch_size, 1, Ht, Wt).float()
            mask = F.interpolate(mask, scale_factor=scale, mode="nearest")
            Ht, Wt = divide_tuple(img_size, patch_size)
            mask = mask.view(batch_size, Ht * Wt).bool()

        return cls(mask, img_size, patch_size)

    def repeat(self, x: int) -> "TokenMask":
        return replace(self, mask=self.mask.repeat(x, 1))

    @classmethod
    def cat(cls, masks: Sequence["TokenMask"]) -> "TokenMask":
        return replace(masks[0], mask=torch.cat([m.mask for m in masks], 0))

    def unmasked(self) -> "TokenMask":
        return replace(self, mask=torch.full_like(self.mask, False))
