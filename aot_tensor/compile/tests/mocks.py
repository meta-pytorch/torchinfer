# Copyright (c) Meta Platforms, Inc. and affiliates.


"""Shared test fixtures for triton_aot tests."""

from typing import Any

# @manual=//triton:triton
from triton.runtime.autotuner import Autotuner


class MockAutotuner(Autotuner):
    """``Autotuner`` subclass with a no-op ``__init__`` for unit tests."""

    # pyre-ignore[14]: Intentionally narrower __init__ for tests.
    def __init__(self, **attrs: Any) -> None:
        # Skip super().__init__() -- real Autotuner needs a real kernel.
        for k, v in attrs.items():
            setattr(self, k, v)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("MockAutotuner is a test fixture, not callable.")
