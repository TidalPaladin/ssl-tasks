#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class MAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        length: int = 100,
        batch_size: int = 4,
        img_size: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        self.length = length
        self.batch_size = batch_size
        self.img_size = img_size

    def setup(self, stage=None):
        self.data = torch.rand(self.length, 3, *self.img_size)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)
