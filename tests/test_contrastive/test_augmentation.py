#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch import Tensor, autocast  # type: ignore

from ssl_tasks.contrastive import ContrastiveAugmentation


class TestContrastiveAugmentation:
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
        aug = ContrastiveAugmentation(img_size)
        assert aug(batch).shape == (batch.shape[0], channels, *img_size)

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_dtype(self, dtype, img_size, batch):
        aug = ContrastiveAugmentation(img_size)
        with autocast("cpu", dtype=dtype):
            out = aug(batch.to(dtype))
        assert out.dtype == dtype

    @pytest.mark.cuda
    def test_gpu(self, img_size, batch):
        aug = ContrastiveAugmentation(img_size)
        batch = batch.cuda()
        out = aug(batch)
        assert out.device == batch.device

    def test_multibatch_augmentation(self, img_size, batch):
        aug1 = ContrastiveAugmentation(img_size, num_batches=1)
        aug2 = ContrastiveAugmentation(img_size, num_batches=2)
        out1 = aug1(batch)
        out2 = aug2(batch)
        assert (out1 != out2).any()

    @pytest.mark.parametrize("num_batches", [1, 2])
    def test_forward_color(self, mocker, img_size, batch, num_batches):
        aug = ContrastiveAugmentation(img_size, num_batches=num_batches)
        m = mocker.spy(aug.position_augment, "forward")
        aug.forward_color(batch)
        m.assert_not_called()

    @pytest.mark.parametrize("num_batches", [1, 2])
    def test_forward_position(self, mocker, img_size, batch, num_batches):
        aug = ContrastiveAugmentation(img_size, num_batches=num_batches)
        m = mocker.spy(aug.color_augment, "forward")
        aug.forward_position(batch)
        m.assert_not_called()

    @pytest.mark.parametrize("global_crop", [True, False])
    def test_crop_override(self, mocker, img_size, batch, global_crop):
        aug = ContrastiveAugmentation(img_size)
        m = mocker.spy(aug.global_crop, "forward")
        aug(batch, global_crop=global_crop)
        assert m.call_count == int(global_crop)
