# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest

import torch
from aot_tensor.arg_specs.common import (
    ConstexprArg,
    resolve_runtime_args,
    sample_spec_and_hash,
)
from aot_tensor.arg_specs.cute import CuTeArgSpec, CuTeTensorArg
from aot_tensor.compile.cutedsl.eager import _compile_cache_key


def _toy_kernel() -> None:
    pass


class CompileCacheKeyTest(unittest.TestCase):
    def test_equals_aot_spec_hash(self) -> None:
        # A cached eager kernel must match what the lowered .so specializes to.
        specs: list[CuTeArgSpec] = [CuTeTensorArg("x"), ConstexprArg("flag", bool)]
        x = torch.empty(4, 8)
        values = resolve_runtime_args(specs, "op", x, True)
        _, aot_hash = sample_spec_and_hash(specs, "op", "eager_test", x, True)
        self.assertEqual(_compile_cache_key(_toy_kernel, specs, "op", values), aot_hash)
