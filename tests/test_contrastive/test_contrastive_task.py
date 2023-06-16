#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, Tuple, cast

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from ssl_tasks.contrastive.task import ContrastiveEmbedding as ContrastiveEmbeddingBase
from ssl_tasks.tokens import TokenMask


class TestContrastiveEmbedding:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        class ContrastiveEmbedding(ContrastiveEmbeddingBase):
            def prepare_backbone(self, _: str) -> nn.Module:
                return backbone

            @property
            def img_size(self) -> Tuple[int, int]:
                return cast(Any, self.backbone).img_size

            def create_head(self) -> nn.Module:
                r"""Creates the MAE head for the model"""
                return nn.Linear(cast(Any, self.backbone).dim, 10)

            def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
                r"""Creates the MAE head for the model"""
                return TokenMask.create(self.img_size, cast(Any, self.backbone).patch_size, batch_size, device=device)

            def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
                x = self.backbone(x, mask)
                x = x.mean(dim=1)
                x = self.embed_head(x)
                return {"embed": x}

        return ContrastiveEmbedding(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)

    def test_predict(self, task, datamodule, logger):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=logger,
        )
        trainer.predict(task, datamodule=datamodule)
