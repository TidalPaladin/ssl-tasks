#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import torch

from ssl_tasks.contrastive import ContrastiveEmbeddingLoss, PointwiseContrastiveEmbeddingLoss


class TestContrastiveEmbeddingLoss:
    @pytest.mark.parametrize("cartesian", [True, False])
    @pytest.mark.parametrize(
        "cuda",
        [
            False,
            pytest.param(True, marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, cuda, cartesian):
        loss = ContrastiveEmbeddingLoss(cartesian=cartesian)
        x = torch.randn(10, 32)
        x_aug = torch.randn(10, 32)
        cls = torch.randint(0, 10, (10,))
        cls_aug = torch.randint(0, 10, (10,))
        if cuda:
            x = x.cuda()
            x_aug = x_aug.cuda()
            cls = cls.cuda()
            cls_aug = cls_aug.cuda()
        result = loss(x, x_aug, cls, cls_aug)
        assert result.device == x.device

        if cartesian:
            assert result.shape == (10, 10)
        else:
            assert result.shape == (10,)


class TestPointwiseContrastiveEmbeddingLoss:
    @pytest.mark.parametrize("cartesian", [True, False])
    @pytest.mark.parametrize(
        "cuda",
        [
            False,
            pytest.param(True, marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, cuda, cartesian):
        loss = PointwiseContrastiveEmbeddingLoss(cartesian=cartesian)
        x = torch.randn(10, 32)
        x_aug = torch.randn(10, 32)
        if cuda:
            x = x.cuda()
            x_aug = x_aug.cuda()
        result = loss(x, x_aug)
        assert result.device == x.device

        if cartesian:
            assert result.shape == (10, 10)
        else:
            assert result.shape == (10,)

    def test_basic_example(self):
        loss = PointwiseContrastiveEmbeddingLoss(cartesian=True)
        x = torch.randn(10, 32)
        x_aug_1 = x.clone()
        x_aug_2 = torch.randn(10, 32)
        loss1 = loss(x, x_aug_1)
        loss2 = loss(x, x_aug_2)
        assert loss1.sum() < loss2.sum()
