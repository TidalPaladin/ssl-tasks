#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Tuple


def check_divisible(
    num: Tuple[int, int],
    denom: Tuple[int, int],
    msg: str = "{num} is not divisible by {denom}",
):
    assert len(num) == len(denom)
    for i in range(len(num)):
        if num[i] % denom[i] != 0:
            raise ValueError(msg)


def divide_tuple(num: Tuple[int, int], denom: Tuple[int, int]) -> Tuple[int, int]:
    assert len(num) == len(denom)
    return tuple(num[i] // denom[i] for i in range(len(num)))


def multiply_tuple(x: Tuple[int, int], y: Tuple[int, int]) -> Tuple[int, int]:
    assert len(x) == len(y)
    return tuple(x[i] * y[i] for i in range(len(x)))


def add_tuple(x: Tuple[int, int], y: Tuple[int, int]) -> Tuple[int, int]:
    assert len(x) == len(y)
    return tuple(x[i] + y[i] for i in range(len(x)))


def update(dest: Dict, src: Dict) -> None:
    for k, v in src.items():
        if k not in dest or not isinstance(v, dict):
            dest[k] = v
        else:
            dest[k].update(v)
