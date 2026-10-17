# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest
from typing import Any
from unittest.mock import MagicMock

import torch
from aot_tensor.compile.compile_state import (
    get_aott_compile_state,
    get_kernel_specs,
    register_active,
)
from aot_tensor.compile.cutedsl.adapter import (
    _collect_cutedsl_spec,
    collect,
    CuTeAdapter,
)
from aot_tensor.cute_specs import CuTeScalarArg, CuTeTensorArg
from aot_tensor.types import CuTeAOT, cutedsl_aot


class CuTeDSLSpecCollectionTest(unittest.TestCase):
    """_collect_cutedsl_spec (collect/dedup/multi)."""

    def setUp(self) -> None:
        CuTeAOT._instances.clear()
        get_aott_compile_state().reset()
        register_active(CuTeAdapter)

    def tearDown(self) -> None:
        CuTeAOT._instances.clear()
        get_aott_compile_state().reset()

    def _op(self) -> Any:
        def vector_add_kernel() -> None:
            pass

        return cutedsl_aot(
            jit_fn=vector_add_kernel,
            arg_specs=[CuTeTensorArg("x"), CuTeScalarArg("n", "i32")],
        )

    def test_collect_dedups_and_adds_new_specs(self) -> None:
        op = self._op()

        # Shape-independent key: different shapes (same dtype/rank/layout) dedup.
        _collect_cutedsl_spec(op, torch.randn(8), 4)
        _collect_cutedsl_spec(op, torch.randn(16), 4)
        self.assertEqual(len(get_kernel_specs(CuTeAdapter.name)[op].specs), 1)

        # A different rank is a distinct specialization -> new spec.
        _collect_cutedsl_spec(op, torch.randn(4, 4), 4)
        self.assertEqual(len(get_kernel_specs(CuTeAdapter.name)[op].specs), 2)
        self.assertEqual(len(get_kernel_specs(CuTeAdapter.name)[op].hashes), 2)

    def test_collect_registers_singleton_dsl(self) -> None:
        """collect() registers one reused CuTeAdapter instance in dsl_state."""
        get_aott_compile_state().reset()
        op = self._op()
        collect(op, torch.randn(8), 4)
        first = get_aott_compile_state().dsl_state["cutedsl"].dsl
        collect(op, torch.randn(8), 4)
        second = get_aott_compile_state().dsl_state["cutedsl"].dsl
        self.assertIsInstance(first, CuTeAdapter)
        self.assertIs(first, second)


class CuTeFindKernelStrictnessTest(unittest.TestCase):
    """Regression: find_kernel scans every CuTeAOT binding in the wrapper's
    globals, not just the called ones.

    The pre-decoupling _find_cutedsl_aot_kernel filtered to markers whose bound
    name appeared at a call site, so an imported-but-uncalled kernel was silently
    ignored. find_kernel now shares find_sole_marker_in_globals with Triton
    (which never filtered), so an uncalled kernel is included -- surfacing as the
    one-kernel-per-wrapper assertion instead of being dropped.
    """

    def setUp(self) -> None:
        CuTeAOT._instances.clear()
        get_aott_compile_state().reset()
        register_active(CuTeAdapter)

    def tearDown(self) -> None:
        CuTeAOT._instances.clear()
        get_aott_compile_state().reset()

    def _collected_op(self, name: str) -> CuTeAOT:
        def kernel() -> None:
            pass

        kernel.__name__ = name
        op = cutedsl_aot(
            jit_fn=kernel,
            arg_specs=[CuTeTensorArg("x"), CuTeScalarArg("n", "i32")],
        )
        _collect_cutedsl_spec(op, torch.randn(8), 4)
        return op

    def _wrapper(self, **globals_: Any) -> Any:
        target = MagicMock()
        target.__globals__ = globals_
        target.__name__ = "cutedsl_wrapper"
        return target

    def test_uncalled_kernel_is_not_filtered_out(self) -> None:
        # A lone kernel the wrapper never calls is still found; the old call-site
        # filter would have returned None here.
        op = self._collected_op("solo")
        result = CuTeAdapter().find_kernel(self._wrapper(solo=op))
        self.assertIsNotNone(result)
        found, names = result
        self.assertIs(found, op)
        self.assertEqual(names, {"solo"})

    def test_extra_uncalled_kernel_trips_one_per_wrapper(self) -> None:
        # Two kernels in globals (e.g. one called + one merely imported) now trip
        # the one-kernel-per-wrapper assertion instead of silently keeping one.
        called = self._collected_op("called")
        imported_only = self._collected_op("imported_only")
        with self.assertRaises(AssertionError) as ctx:
            CuTeAdapter().find_kernel(
                self._wrapper(called=called, imported_only=imported_only)
            )
        self.assertIn("Expected exactly 1 kernel", str(ctx.exception))
