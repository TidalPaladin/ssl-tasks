#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .augmentation import ContrastiveAugmentation
from .loss import ContrastiveEmbeddingLoss, PointwiseContrastiveEmbeddingLoss, UniqueEmbeddingLoss
from .task import ContrastiveEmbedding


__all__ = [
    "ContrastiveEmbeddingLoss",
    "PointwiseContrastiveEmbeddingLoss",
    "UniqueEmbeddingLoss",
    "ContrastiveEmbedding",
    "ContrastiveAugmentation",
]
