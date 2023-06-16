#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ssl_tasks import __version__


def test_version():
    assert isinstance(__version__, str)
