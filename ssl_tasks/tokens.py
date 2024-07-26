#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .helpers import divide_tuple


@dataclass
class TokenMask:
    r"""Class for masking tokens in a vision transformer.

    .. note::
        In the mask, ``True`` indicates an unmasked token and ``False`` indicates a masked token.
    """

    mask: Tensor
    size: Sequence[int]
    patch_size: Sequence[int]

    def __post_init__(self) -> None:
        self.size = tuple(self.size)
        self.patch_size = tuple(self.patch_size)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"mask={tuple(self.mask.shape)}, "
        s += f"size={self.size}, "
        s += f"patch_size={self.patch_size}"
        s += ")"
        return s

    def __invert__(self) -> "TokenMask":
        return replace(self, mask=~self.mask)

    @property
    def shape(self) -> torch.Size:
        return self.mask.shape

    @property
    def tokenized_image_size(self) -> Sequence[int]:
        return divide_tuple(self.size, self.patch_size)

    @property
    def num_tokens(self) -> int:
        return self.mask.shape[-1]

    @property
    def unmasked_count(self) -> Tensor:
        return self.mask.sum(dim=-1)

    def apply_to_input(self, x: Tensor, fill_value: float | Tensor = 0) -> Tensor:
        r"""Apply the mask to an input.

        Args:
            x: Input tensor
            fill_value: Value to fill the masked tokens with.

        Shapes:
            x - :math:`(N, C, *S)` where :math:`S` is some number of spatial dimensions
            Output - :math:`(N, C, *S_t)` where :math:`S_t` is the tokenized image size

        Returns:
            Tensor: The input tensor with the mask applied
        """
        N, C, *S = x.shape

        # Tokenize the input according to the patch size
        pattern_in = "n c " + " ".join([f"(t{i} p{i})" for i in range(len(S))])
        pattern_out = (
            "n ("
            + " ".join([f"t{i}" for i in range(len(S))])
            + ") ("
            + " ".join([f"p{i}" for i in range(len(S))])
            + ") c"
        )
        sizes = {
            "n": N,
            "c": C,
            **{f"t{i}": t for i, t in enumerate(self.tokenized_image_size)},
            **{f"p{i}": p for i, p in enumerate(self.patch_size)},
        }
        tokenized = rearrange(x, f"{pattern_in} -> {pattern_out}", **sizes)

        # Apply masking to the tokenized input
        tokenized[~self.mask] = fill_value
        return rearrange(tokenized, f"{pattern_out} -> {pattern_in}", **sizes)

    def apply_to_tokens(
        self,
        x: Tensor,
        fill_value: Optional[Union[float, Tensor]] = None,
        padding_value: float | Tensor = 0,
        inverse: bool = False,
    ) -> Tensor:
        r"""Apply the mask to tokens.

        When ``fill_value=None`` and ``mask.is_ragged``, the result is padded to match the number of tokens in the
        largest batch element. Padding is done with zeros and is applied to the end of each batch sequence.

        Args:
            x: Input tensor
            fill_value: Value to fill the masked tokens with. If ``None``, the masked tokens are removed.
            padding_value: Padding value used when the mask is ragged.
            inverse: If ``True``, the mask is inverted before applying it.

        Shapes:
            x - :math:`(N, L, D)` where :math:`L` is the number of tokens
            Output - :math:`(N, L, D)` or :math:`(N, L', D)` where :math:`L'` is the number of unmasked tokens

        Returns:
            Input tensor with the mask applied
        """
        N, L, D = x.shape
        if fill_value is None:
            m: Tensor = ~self.mask if inverse else self.mask
            if self.is_ragged:
                # Build indices where we want to put non-padding values
                max_tokens = int(self.unmasked_count.amax().item())
                indices = torch.stack(
                    [
                        torch.arange(N, device=x.device).view(N, 1).expand(-1, max_tokens),
                        torch.arange(max_tokens, device=x.device).view(1, max_tokens).expand(N, -1),
                    ],
                    dim=-1,
                )
                indices = indices[indices[..., -1] < self.unmasked_count.view(-1, 1)]

                if isinstance(padding_value, Tensor):
                    o = padding_value.type_as(x).broadcast_to((N, max_tokens, D))
                else:
                    o = x.new_full((N, max_tokens, D), padding_value)
                x = torch.index_put(o, indices.unbind(-1), x[m])
            else:
                x = rearrange(x[m], "(n l) c -> n l c", n=N)
        else:
            fill_value = fill_value.type_as(x) if isinstance(fill_value, Tensor) else fill_value
            mask = self.mask.view(N, L, 1)
            x = torch.where(mask, x, fill_value)
        return x

    def restore_tokens(self, x: Tensor, fill_value: float = 0) -> Tensor:
        r"""Restore dropped tokens to form a full input.

        This is only needed when tokens have been removed from the input by passing ``None`` as the ``fill_value``
        to :meth:`apply_to_tokens` or :meth:`apply_to_input`.

        Args:
            x: Tensor of tokens
            fill_value: Value to fill the masked tokens with

        Shapes:
            x - :math:`(N, L', D)`
            Output - :math:`(N, L, D)`

        Returns:
            Tensor of tokens restored to a full input
        """
        N, _, D = x.shape
        output = x.new_full((N, self.num_tokens, D), fill_value)
        if self.is_ragged:
            max_tokens = int(self.unmasked_count.amax().item())
            indices = torch.stack(
                [
                    torch.arange(N, device=x.device).view(N, 1).expand(-1, max_tokens),
                    torch.arange(max_tokens, device=x.device).view(1, max_tokens).expand(N, -1),
                ],
                dim=-1,
            )
            indices = indices[indices[..., -1] < self.unmasked_count.view(-1, 1)]
            output[self.mask] = x[indices.unbind(-1)]
        else:
            output[self.mask] = x.view(-1, D)
        return output

    @classmethod
    def create(
        cls,
        size: Sequence[int],
        patch_size: Sequence[int],
        batch_size: int = 1,
        mask_ratio: float = 0.25,
        scale: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> "TokenMask":
        r"""Create a token mask for an input.

        Args:
            size: Size of the input image or volume
            patch_size: Size of the patch
            batch_size: Size of the batch
            mask_ratio: Ratio of tokens to mask
            scale: Dilates the mask by this factor. For example, if ``scale == 2`` and ``len(size) == 2``,
                masking will be done in (2x2) contiguous blocks.
            device: Device to create the mask on

        Raises:
            ValueError: If ``mask_ratio`` is not in the range (0, 1)
            ValueError: If ``scale`` is less than 1

        Returns:
            TokenMask: A token mask
        """
        if not 0 < mask_ratio < 1.0:
            raise ValueError(f"Invalid `mask_ratio` {mask_ratio}")
        if scale < 1:
            raise ValueError(f"Invalid `scale` {scale}")

        # Get the tokenized input size accounting for patch size and scale
        tokenized_size = divide_tuple(size, patch_size)
        tokenized_size = divide_tuple(tokenized_size, (scale,) * len(tokenized_size))

        # Compute the total number of tokens and number of masked tokens
        Lmask = int(math.prod(tokenized_size))
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

        # dilate the mask if a non-unit scale was chosen
        if scale > 1:
            mask = mask.view(batch_size, 1, *tokenized_size).float()
            mask = F.interpolate(mask, scale_factor=scale, mode="nearest")
            tokenized_size = divide_tuple(size, patch_size)
            mask = mask.view(batch_size, math.prod(tokenized_size)).bool()

        return cls(mask, size, patch_size)

    @property
    def is_ragged(self) -> bool:
        return len(set(self.mask.sum(-1).tolist())) > 1

    @property
    def indices(self) -> Tensor:
        return self.mask.nonzero()

    def repeat(self, x: int) -> "TokenMask":
        return replace(self, mask=self.mask.repeat(x, 1))

    @classmethod
    def cat(cls, masks: Sequence["TokenMask"]) -> "TokenMask":
        return replace(masks[0], mask=torch.cat([m.mask for m in masks], 0))

    def unmasked(self) -> "TokenMask":
        return replace(self, mask=torch.full_like(self.mask, False))

    def resize(self, size: Sequence[int]) -> "TokenMask":
        r"""Resize the mask to a new size.

        Args:
            size: New size of the mask

        Returns:
            TokenMask: A token mask
        """
        new_tokenized_size = divide_tuple(size, self.patch_size)
        mask = self.mask.view(*self.tokenized_image_size)
        mask = (
            F.interpolate(mask.view(1, 1, *mask.shape).float(), size=new_tokenized_size, mode="nearest")
            .bool()
            .view(1, -1)
        )
        return replace(self, size=size, mask=mask)
