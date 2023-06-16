#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, Tuple, cast

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from ssl_tasks.helpers import divide_tuple
from ssl_tasks.mae.task import MAE as MAEBase
from ssl_tasks.tokens import TokenMask


class TestMAE:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        class MAE(MAEBase):
            def prepare_backbone(self, _: str) -> nn.Module:
                return backbone

            @property
            def img_size(self) -> Tuple[int, int]:
                return cast(Any, self.backbone).img_size

            def create_head(self) -> nn.Module:
                r"""Creates the MAE head for the model"""
                patch_size: Tuple[int, int] = cast(Any, self.backbone).patch_size
                dim: int = cast(Any, self.backbone).dim
                out_channels = 1
                H, W = divide_tuple(self.img_size, patch_size)
                Hp, Wp = patch_size
                return nn.Sequential(
                    nn.Linear(dim, Hp * Wp * out_channels),
                    Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=H, w=W, p1=Hp, p2=Wp),
                )

            def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
                r"""Creates the MAE head for the model"""
                return TokenMask.create(self.img_size, cast(Any, self.backbone).patch_size, batch_size, device=device)

            def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
                x = self.backbone(x, mask)
                x = self.mae_head(x)
                return {"mae": x}

        return MAE(backbone, optimizer_init=optimizer_init)

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
