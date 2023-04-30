#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from .contrastive import ContrastiveEmbedding
from .mae import MAE
from .query_patch import BoxIOU, QueryPatch


__version__ = importlib.metadata.version("ssl-tasks")

__all__ = [
    "ContrastiveEmbedding",
    "MAE",
    "QueryPatch",
    "BoxIOU",
]
