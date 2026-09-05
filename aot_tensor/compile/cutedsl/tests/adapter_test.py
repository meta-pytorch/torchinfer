# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import ast
import unittest
from types import SimpleNamespace
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
    CuTeAOTOperatorTransform,
)
from aot_tensor.cute_specs import CuTeScalarArg, CuTeTensorArg
from aot_tensor.types import CuTeAOT, cutedsl_aot
from parameterized import parameterized


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


class CuTeAOTOperatorTransformTest(unittest.TestCase):
    """CuTeDSL wrapper-codegen transform (lives in adapter.py): @torch.jit.unused
    stripping + call detection/rewrite to torch.ops.triton_aot.*."""

    def _transform(self, global_names: set[str]) -> CuTeAOTOperatorTransform:
        # kernel only needs ``.name`` for the rewrite; a stub avoids a real CuTeAOT.
        return CuTeAOTOperatorTransform(
            # pyre-ignore[6]: stub only needs ``.name`` for the rewrite.
            kernel=SimpleNamespace(name="_cutedsl_vector_add"),
            global_names=set(global_names),
        )

    @parameterized.expand(
        [
            ("strips_when_calls_cutedsl_kernel", True, True, 1),
            ("preserves_when_no_kernel_call", True, False, 2),
            ("no_op_when_no_jit_unused", False, True, 1),
        ]
    )
    def test_strip_jit_unused(
        self,
        _name: str,
        has_jit_unused: bool,
        calls_kernel: bool,
        expected_decorators: int,
    ) -> None:
        decorators = "@torch.jit.unused\n" if has_jit_unused else ""
        body = "    _vector_add_aot(x, y, out, n)" if calls_kernel else "    return x"
        source = f"""
{decorators}@torch.fx.wrap
def launcher(x):
{body}
"""
        tree = ast.parse(source)
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)
        self._transform({"_vector_add_aot"}).visit_FunctionDef(func)
        self.assertEqual(len(func.decorator_list), expected_decorators)

    def test_calls_cutedsl_kernel(self) -> None:
        t = self._transform({"_vector_add_aot"})
        called = ast.parse("def f(x):\n    _vector_add_aot(x)\n").body[0]
        not_called = ast.parse("def f(x):\n    return x\n").body[0]
        # Call only in a nested def: the outer f must NOT be flagged.
        nested_only = ast.parse(
            "def f(x):\n"
            "    def g(y):\n"
            "        return _vector_add_aot(y)\n"
            "    return x\n"
        ).body[0]
        assert isinstance(called, ast.FunctionDef)
        assert isinstance(not_called, ast.FunctionDef)
        assert isinstance(nested_only, ast.FunctionDef)
        self.assertTrue(t._calls_cutedsl_kernel(called))
        self.assertFalse(t._calls_cutedsl_kernel(not_called))
        self.assertFalse(t._calls_cutedsl_kernel(nested_only))

    def test_visit_rewrites_call_to_torch_ops(self) -> None:
        source = (
            "def launcher(x, y, out, n):\n"
            "    _vector_add_aot(x, y, out, n)\n"
            "    return out\n"
        )
        tree = ast.parse(source)
        self._transform({"_vector_add_aot"}).visit(tree)
        code = ast.unparse(tree)
        self.assertIn("torch.ops.triton_aot._cutedsl_vector_add(x, y, out, n)", code)
        self.assertNotIn("_vector_add_aot(", code)
