#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from deep_helpers.testing import handle_cuda_mark
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData as TVFakeData
from torchvision.transforms.v2 import ConvertImageDtype, ToImage

from ssl_tasks.tokens import TokenMask


def pytest_runtest_setup(item):
    handle_cuda_mark(item)


class Backbone(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        img_size: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=4, stride=4),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=1, batch_first=True),
            num_layers=2,
        )

    @property
    def patch_size(self) -> Tuple[int, int]:
        return (4, 4)

    def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Tensor:
        x = self.patch_embed(x)
        if mask is not None:
            mask.apply_to_tokens(x)
        x = self.blocks(x)
        return x


class FakeData(TVFakeData):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, _ = super().__getitem__(index)
        img = ToImage()(img)
        img = ConvertImageDtype(torch.float32)(img)
        return {"img": img}


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, img_size: Tuple[int, int] = (32, 32)):
        super().__init__()
        self.dataset = FakeData(size=100, image_size=img_size)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=4)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=4)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=4)


@pytest.fixture
def backbone():
    return Backbone()


@pytest.fixture
def datamodule():
    return DummyDataModule()


@pytest.fixture
def optimizer_init():
    return {
        "class_path": "torch.optim.Adam",
        "init_args": {"lr": 1e-3},
    }


@pytest.fixture
def logger(mocker):
    logger = mocker.MagicMock(name="logger")
    return logger


@pytest.fixture
def trainer():
    trainer = pl.Trainer(
        fast_dev_run=True,
        logger=None,
    )
