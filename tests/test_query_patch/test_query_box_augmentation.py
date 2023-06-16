#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import torch
from torch import Tensor, autocast  # type: ignore

from ssl_tasks.query_patch import SmallBoxCrop


class TestSmallBoxCrop:
    @pytest.fixture
    def img_size(self):
        return (32, 32)

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def channels(self):
        return 3

    @pytest.fixture
    def batch(self, batch_size, channels, img_size) -> Tensor:
        torch.random.manual_seed(42)
        return torch.rand(batch_size, channels, *img_size)

    @pytest.mark.parametrize("img_size", [(32, 32), (64, 64)])
    @pytest.mark.parametrize("channels", [1, 3])
    def test_img_size(self, img_size, batch, channels):
        x_aug, box = SmallBoxCrop(img_size)(batch)
        assert x_aug.shape == (batch.shape[0], channels, *img_size)
        assert box.shape == (batch.shape[0], 1, 4)

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_dtype(self, dtype, img_size, batch):
        aug = SmallBoxCrop(img_size)
        with autocast("cpu", dtype=dtype):
            x_aug, box = aug(batch.to(dtype))
        assert x_aug.dtype == dtype
        assert box.dtype == dtype

    @pytest.mark.cuda
    def test_gpu(self, img_size, batch):
        aug = SmallBoxCrop(img_size)
        batch = batch.cuda()
        x_aug, box = aug(batch)
        assert x_aug.device == batch.device
        assert box.device == batch.device

    def test_multibatch_augmentation(self, img_size, batch):
        aug1 = SmallBoxCrop(img_size, num_batches=1)
        aug2 = SmallBoxCrop(img_size, num_batches=2)
        x_aug_1, box_1 = aug1(batch)
        x_aug_2, box_2 = aug2(batch)
        assert (x_aug_1 != x_aug_2).any()
        assert (box_1 != box_2).any()
