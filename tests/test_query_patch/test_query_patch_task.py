#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Tuple, cast

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from ssl_tasks.query_patch import QueryPatch as QueryPatchBase
from ssl_tasks.tokens import TokenMask


class BoxDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decoder = nn.TransformerDecoderLayer(dim, 1, dim_feedforward=dim, batch_first=True)
        self.head = nn.Linear(dim, 4)

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        out = self.decoder(query, kv)
        return self.head(out)


class TestQueryPatch:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        class QueryPatch(QueryPatchBase):
            def prepare_backbone(self, _: str) -> nn.Module:
                return backbone

            @property
            def img_size(self) -> Tuple[int, int]:
                return cast(Any, self.backbone).img_size

            def create_head(self) -> nn.Module:
                r"""Creates the MAE head for the model"""
                dim = cast(Any, self.backbone).dim
                return BoxDecoder(dim)

            def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
                r"""Creates the MAE head for the model"""
                return TokenMask.create(self.img_size, cast(Any, self.backbone).patch_size, batch_size, device=device)

            def forward(
                self,
                x: Tensor,
                x_box: Tensor,
            ) -> Dict[str, Tensor]:
                # Encode image and crop
                x = self.backbone(x)
                x_box = self.backbone(x_box)

                query = x_box.mean(dim=1, keepdim=True)
                box = self.box_decoder(query, x)
                return {"box": box}

        return QueryPatch(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
