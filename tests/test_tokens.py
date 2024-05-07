import math

import pytest
import torch
from torch.testing import assert_close

from ssl_tasks.helpers import divide_tuple
from ssl_tasks.tokens import TokenMask


class TestTokenMask:
    @pytest.fixture(autouse=True)
    def seed(self):
        torch.random.manual_seed(0)

    @pytest.mark.parametrize(
        "size, patch_size",
        [
            ((224, 224), (16, 16)),
            ((256, 256), (32, 32)),
            ((512, 512), (64, 64)),
            ((32, 224, 224), (4, 16, 16)),
        ],
    )
    def test_create_sizes(self, size, patch_size):
        exp_len = math.prod(divide_tuple(size, patch_size))
        token_mask = TokenMask.create(size, patch_size)
        assert token_mask.mask.numel() == exp_len, f"Expected {exp_len} tokens, got {token_mask.mask.numel()}"
        assert token_mask.num_tokens == exp_len, f"Expected {exp_len} tokens, got {token_mask.num_tokens}"

    @pytest.mark.parametrize(
        "ratio",
        [
            0.25,
            0.5,
            0.75,
            pytest.param(0, marks=pytest.mark.xfail(reason="Invalid ratio", strict=True)),
            pytest.param(1, marks=pytest.mark.xfail(reason="Invalid ratio", strict=True)),
        ],
    )
    def test_create_ratio(self, ratio):
        size = (224, 224)
        patch_size = (2, 2)
        token_mask = TokenMask.create(size, patch_size, mask_ratio=ratio)
        if ratio == 0:
            assert not token_mask.mask.any(), "Expected no tokens to be masked"
        elif ratio == 1:
            assert token_mask.mask.all(), "Expected all tokens to be masked"
        else:
            assert_close(token_mask.mask.float().mean(), torch.tensor(1 - ratio), atol=0.01, rtol=0)

    def test_repr(self):
        token_mask = TokenMask.create((224, 224), (16, 16))
        assert isinstance(repr(token_mask), str)
        assert "size=(224, 224), patch_size=(16, 16)" in repr(token_mask)

    def test_invert(self):
        token_mask = TokenMask.create((224, 224), (16, 16), mask_ratio=0.25)
        inv = ~token_mask
        assert (inv.mask == token_mask.mask.logical_not()).all(), "Expected inverted mask"

    @pytest.mark.parametrize(
        "size, patch_size",
        [
            ((224, 224), (16, 16)),
            ((256, 256), (32, 32)),
            ((512, 512), (64, 64)),
            ((32, 224, 224), (4, 16, 16)),
        ],
    )
    @pytest.mark.parametrize("fill_value", [0, None, torch.zeros(8)])
    def test_apply_to_tokens(self, size, patch_size, fill_value):
        D = 8
        L = math.prod(divide_tuple(size, patch_size))
        token_mask = TokenMask.create(size, patch_size)
        x = torch.randn(1, L, D, requires_grad=True)
        o = token_mask.apply_to_tokens(x, fill_value)

        assert o.shape[-1] == D
        if fill_value is None:
            assert o.shape[1] == token_mask.unmasked_count
            assert_close(o, x[token_mask.mask].broadcast_to(o.shape).contiguous())
        else:
            exp = torch.as_tensor(fill_value, dtype=o.dtype).broadcast_to(o[~token_mask.mask].shape).contiguous()
            assert_close(o[~token_mask.mask], exp)
            assert_close(o[token_mask.mask], x[token_mask.mask])

        o.sum().backward()

    @pytest.mark.parametrize(
        "size, patch_size",
        [
            ((224, 224), (16, 16)),
            ((256, 256), (32, 32)),
            ((512, 512), (64, 64)),
            ((32, 224, 224), (4, 16, 16)),
        ],
    )
    @pytest.mark.parametrize("fill_value", [0, torch.zeros(8)])
    def test_apply_to_input(self, size, patch_size, fill_value):
        D = 8
        ratio = 0.25
        token_mask = TokenMask.create(size, patch_size, mask_ratio=ratio)
        x = torch.ones(1, D, *size, requires_grad=True)
        o = token_mask.apply_to_input(x, fill_value)
        assert o.shape == x.shape
        assert_close(o.mean(), o.new_tensor(1 - ratio))
        o.sum().backward()

    @pytest.mark.parametrize(
        "size, patch_size",
        [
            ((224, 224), (16, 16)),
            ((256, 256), (32, 32)),
            ((512, 512), (64, 64)),
            ((32, 224, 224), (4, 16, 16)),
        ],
    )
    def test_restore_tokens(self, size, patch_size):
        D = 8
        L = math.prod(divide_tuple(size, patch_size))
        token_mask = TokenMask.create(size, patch_size)
        x = torch.randn(1, L, D, requires_grad=True)
        masked = token_mask.apply_to_tokens(x, fill_value=None)
        o = token_mask.restore_tokens(masked)

        assert_close(o[token_mask.mask], x[token_mask.mask])
        assert_close(o[~token_mask.mask], o.new_tensor(0).broadcast_to(o[~token_mask.mask].shape).contiguous())
        o.sum().backward()

    @pytest.mark.parametrize(
        "mask, exp",
        [
            (([True, True], [True, True]), False),
            (([True, False], [True, True]), True),
            (([False, True], [True, True]), True),
            (([False, False], [False, False]), False),
        ],
    )
    def test_is_ragged(self, mask, exp):
        token_mask = TokenMask(torch.tensor(mask, dtype=torch.bool), (2, 2), (2, 2))
        assert token_mask.is_ragged == exp

    def test_apply_to_tokens_ragged(self):
        torch.random.manual_seed(0)
        N, L, D = 2, 2, 8
        mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
        token_mask = TokenMask(mask, (2, 2), (2, 2))

        x = torch.randn(N, L, D, requires_grad=True)
        o = token_mask.apply_to_tokens(x, fill_value=None)
        assert o.shape == (N, L, D)
        assert (o[0, 1] == 0).all()
        o.sum().backward()
