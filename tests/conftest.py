#!/usr/bin/env python
# -*- coding: utf-8 -*-
from deep_helpers.testing import handle_cuda_mark


def pytest_runtest_setup(item):
    handle_cuda_mark(item)
