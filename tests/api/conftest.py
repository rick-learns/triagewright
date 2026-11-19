from __future__ import annotations

import pytest

from tests.api.stack import Stack, fake_stack


@pytest.fixture
def stack() -> Stack:
    return fake_stack()
