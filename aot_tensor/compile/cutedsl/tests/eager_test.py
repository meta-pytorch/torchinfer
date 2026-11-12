# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest
from typing import Any, Callable
from unittest.mock import patch

import torch
from aot_tensor.arg_specs.common import (
    ConstexprArg,
    resolve_runtime_args,
    sample_spec_and_hash,
)
from aot_tensor.arg_specs.cute import CuTeArgSpec, CuTeTensorArg
from aot_tensor.compile.cutedsl.eager import _compile_cache_key, run_cutedsl_jit


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


class RunCutedslJitTest(unittest.TestCase):
    def test_constexprs_reach_compile_but_not_the_compiled_call(self) -> None:
        specs: list[CuTeArgSpec] = [CuTeTensorArg("x"), ConstexprArg("flag", bool)]
        calls: list[tuple[str, tuple[Any, ...]]] = []

        def compile_fn(fn: Any, *args: Any) -> Callable[..., None]:
            calls.append(("compile", args))
            return lambda *a: calls.append(("call", a))

        # build_cute_call_args reads the CUDA stream; the rest runs as is.
        with patch(
            "aot_tensor.compile.cutedsl.eager.build_cute_call_args",
            return_value=("x", True, "stream"),
        ):
            run_cutedsl_jit(
                compile_fn, _toy_kernel, specs, "run_test_op", torch.empty(4, 8), True
            )
        self.assertEqual(
            calls,
            [("compile", ("x", True, "stream")), ("call", ("x", "stream"))],
        )
