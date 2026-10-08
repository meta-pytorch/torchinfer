# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import sys
import unittest
from typing import Any
from unittest.mock import call, MagicMock, patch

from aot_tensor.compile.aott_compile import enable_spec_collection
from aot_tensor.compile.compile_state import get_aott_compile_state
from aot_tensor.compile.cutedsl import eager as cutedsl_eager
from aot_tensor.compile.tests.mocks import MockAutotuner
from aot_tensor.cute_specs import CuTeScalarArg, CuTeTensorArg
from aot_tensor.types import (
    _default_cutedsl_op_name,
    AnnotationHint,
    CuTeAOT,
    cutedsl_aot,
    get_all_triton_aot_instances,
    reset_all_triton_aot_autotune_cache,
    TritonAOT,
)
from parameterized import parameterized


class _MarkerResetTest(unittest.TestCase):
    """Base for tests that build markers or toggle collection: clear both marker
    classes' instance lists and reset the AOTTCompileState singleton before and
    after each test so cases don't leak state into one another."""

    def setUp(self) -> None:
        TritonAOT._instances.clear()
        CuTeAOT._instances.clear()
        get_aott_compile_state().reset()

    def tearDown(self) -> None:
        TritonAOT._instances.clear()
        CuTeAOT._instances.clear()
        get_aott_compile_state().reset()


class AnnotationHintTest(unittest.TestCase):
    """Tests for AnnotationHint validation."""

    @parameterized.expand(
        [
            ("scalar_div16", "i32", 16),
            ("scalar_div8", "i32", 8),
            ("scalar_eq1", "i32", 1),
            ("ptr_align16", "*fp32", 16),
            ("float_div16", "fp32", 16),
        ]
    )
    def test_valid_hints_accepted(self, _name: str, dtype: str, hint: int) -> None:
        ann = AnnotationHint(dtype, hint)
        self.assertEqual(ann.dtype, dtype)
        self.assertEqual(ann.hint, hint)

    @parameterized.expand(
        [
            ("hint_4", "i32", 4, r"invalid annotation hint"),
            ("hint_32", "i32", 32, r"invalid annotation hint"),
            ("hint_0", "i32", 0, r"invalid annotation hint"),
            ("hint_7", "fp32", 7, r"invalid annotation hint"),
            ("ptr_align_1", "*fp32", 1, r"invalid pointer alignment"),
            ("ptr_align_8", "*bf16", 8, r"invalid pointer alignment"),
        ]
    )
    def test_invalid_hint_rejected(
        self, _name: str, dtype: str, hint: int, error: str
    ) -> None:
        """Bad hint values and bad pointer alignments are rejected."""
        with self.assertRaisesRegex(RuntimeError, error):
            AnnotationHint(dtype, hint)

    def test_to_tuple(self) -> None:
        """to_tuple() produces a plain tuple for raw spec format."""
        ann = AnnotationHint("i32", 16)
        t = ann.to_tuple()
        self.assertIsInstance(t, tuple)
        self.assertNotIsInstance(t, AnnotationHint)
        self.assertEqual(t, ("i32", 16))

    def test_raw_tuple_converted_by_triton_aot(self) -> None:
        """TritonAOT.__init__ converts raw tuples to AnnotationHint."""
        fn = MagicMock()
        inst = TritonAOT(fn, {"N": ("i32", 16)})
        self.assertIsInstance(inst.annotations["N"], AnnotationHint)

    def test_invalid_raw_tuple_rejected_by_triton_aot(self) -> None:
        """TritonAOT.__init__ rejects invalid raw tuples via normalization."""
        fn = MagicMock()
        with self.assertRaisesRegex(RuntimeError, r"invalid pointer alignment"):
            TritonAOT(fn, {"X": ("*fp32", 1)})


class AOTTMarkerMetaTest(_MarkerResetTest):
    """Unit tests for AOTTMarkerMeta metaclass and get_all_triton_aot_instances."""

    def test_new_class_starts_with_empty_instances(self) -> None:
        self.assertEqual(TritonAOT.get_instances(), [])

    def test_single_instance_tracked(self) -> None:
        fn = MagicMock()
        inst = TritonAOT(fn, {})
        instances = TritonAOT.get_instances()
        self.assertEqual(len(instances), 1)
        self.assertIs(instances[0], inst)

    def test_multiple_instances_tracked_in_order(self) -> None:
        fn1 = MagicMock()
        fn2 = MagicMock()
        inst1 = TritonAOT(fn1, {})
        inst2 = TritonAOT(fn2, {"x": "i32"})
        instances = TritonAOT.get_instances()
        self.assertEqual(len(instances), 2)
        self.assertIs(instances[0], inst1)
        self.assertIs(instances[1], inst2)

    def test_get_all_triton_aot_instances_returns_same_list(self) -> None:
        fn = MagicMock()
        inst = TritonAOT(fn, {})
        result = get_all_triton_aot_instances()
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], inst)

    def test_instance_preserves_fn_and_annotations(self) -> None:
        fn = MagicMock()
        annotations = {"x": "i32", "y": "fp32"}
        inst = TritonAOT(fn, annotations)
        self.assertIs(inst.fn, fn)
        self.assertEqual(inst.annotations, annotations)


class ResetAllTritonAOTAutotuneCacheTest(_MarkerResetTest):
    """Tests for reset_all_triton_aot_autotune_cache."""

    def _make_autotuner_mock(self) -> Any:
        """Wrap ``MockAutotuner`` with a fake ``.fn`` so the production code
        path ``... f"{autotune_fn.fn.__name__}" ...`` works."""
        mock = MockAutotuner(
            cache={"key1": "val1", "key2": "val2"},
            configs=[MagicMock()],
            arg_names=["a", "b"],
        )
        mock.fn = MagicMock()
        mock.fn.__name__ = "mock_kernel"
        return mock

    def _make_jit_mock(self) -> Any:
        """Synthetic JITFunction-shaped fake (does NOT pass ``is_autotuner``)."""
        cls = type("JITFunction", (), {})
        cls.__module__ = "triton.runtime.jit"
        mock = object.__new__(cls)
        mock.cache = {"should": "remain"}
        return mock

    def _enable_compile(self) -> None:
        """Turn on AOT spec collection (registers the marker collectors)."""
        enable_spec_collection()

    def test_returns_false_when_compile_not_enabled(self) -> None:
        result = reset_all_triton_aot_autotune_cache()
        self.assertFalse(result)

    def test_returns_false_when_no_autotuner_instances(self) -> None:
        self._enable_compile()
        TritonAOT(self._make_jit_mock(), {})

        result = reset_all_triton_aot_autotune_cache()
        self.assertFalse(result)

    @parameterized.expand(
        [
            ("single_autotuner", 1, 0),
            ("multiple_autotuners", 2, 0),
            ("mixed_autotuner_and_jit", 1, 1),
        ]
    )
    def test_clears_autotuner_caches(
        self,
        _name: str,
        num_autotuners: int,
        num_jit: int,
    ) -> None:
        """Autotuner caches are cleared; non-autotuner caches are preserved."""
        self._enable_compile()
        autotuners = [self._make_autotuner_mock() for _ in range(num_autotuners)]
        jit_fns = [self._make_jit_mock() for _ in range(num_jit)]
        for m in autotuners + jit_fns:
            TritonAOT(m, {})

        result = reset_all_triton_aot_autotune_cache()

        self.assertTrue(result)
        for at in autotuners:
            self.assertEqual(len(at.cache), 0)
        for jit in jit_fns:
            self.assertGreater(len(jit.cache), 0)


class TritonAOTCallTest(_MarkerResetTest):
    """TritonAOT.run dispatch: forwards each call to the registered spec
    collector (and only when one is registered), then always runs the wrapped
    kernel. Uses a stand-in collector -- the real one is registered by
    ``enable_spec_collection`` and its lazy-import path is covered e2e for
    CuTeDSL in ``CuTeAOTTest``."""

    @parameterized.expand([("registered", True), ("cleared", False)])
    def test_run_forwards_to_collector_only_when_registered(
        self, _name: str, registered: bool
    ) -> None:
        fn = MagicMock()
        fn.run.return_value = "OUT"
        marker = TritonAOT(fn, {})
        collector = MagicMock()
        if registered:
            TritonAOT.set_spec_collector(collector)

        self.assertEqual(marker.run(7), "OUT")

        # The wrapped kernel always runs; the collector fires once, with the
        # marker passed exactly once -- call(marker, 7), not (marker, marker, 7)
        # -- which guards the self-double-bind of reading via the instance.
        fn.run.assert_called_once_with(7)
        expected = [call(marker, 7)] if registered else []
        self.assertEqual(collector.call_args_list, expected)


class DefaultCuTeDSLOpNameTest(unittest.TestCase):
    """Op name derives from the kernel: _cutedsl_<jit_fn name or class name>."""

    def test_function_kernel_uses_name(self) -> None:
        def vector_add_kernel() -> None:
            pass

        self.assertEqual(
            _default_cutedsl_op_name(vector_add_kernel),
            "_cutedsl_vector_add_kernel",
        )

    def test_callable_instance_uses_class_name(self) -> None:
        class VectorAddKernel:
            def __call__(self) -> None:
                pass

        self.assertEqual(
            _default_cutedsl_op_name(VectorAddKernel()),
            "_cutedsl_VectorAddKernel",
        )


class CuTeAOTTest(_MarkerResetTest):
    """CuTeAOT construction (via cutedsl_aot) and __call__ dispatch."""

    def test_cutedsl_aot_builds_op(self) -> None:
        def vector_add_kernel() -> None:
            pass

        op = cutedsl_aot(jit_fn=vector_add_kernel, arg_specs=[CuTeTensorArg("x")])
        self.assertIsInstance(op, CuTeAOT)
        self.assertEqual(op.name, "_cutedsl_vector_add_kernel")
        self.assertEqual(op.module_basename, "types_test")
        self.assertIs(op.jit_fn, vector_add_kernel)
        self.assertIs(op, CuTeAOT.get_instances()[-1])

    def test_invalid_op_name_rejected(self) -> None:
        # A lambda's __name__ is "<lambda>", so the derived op name is not alnum.
        with self.assertRaisesRegex(RuntimeError, "invalid op name"):
            cutedsl_aot(jit_fn=lambda: None, arg_specs=[])

    @parameterized.expand([("enabled", True), ("disabled", False)])
    def test_call_runs_jit(self, _name: str, enabled: bool) -> None:
        def vector_add_kernel() -> None:
            pass

        op = cutedsl_aot(
            jit_fn=vector_add_kernel, arg_specs=[CuTeScalarArg("n", "i32")]
        )
        if enabled:
            enable_spec_collection()
        # __call__ imports cutlass.cute and injects cute.compile into the runner;
        # mock it via sys.modules so this CPU test needs no real (GPU-only) cutlass.
        # ``import cutlass.cute as cute`` binds ``cute = cutlass_mock.cute``, so
        # the injected fn is ``cutlass_mock.cute.compile``.
        cutlass_mock = MagicMock()
        with (
            patch.dict(
                sys.modules,
                {"cutlass": cutlass_mock, "cutlass.cute": cutlass_mock.cute},
            ),
            patch.object(cutedsl_eager, "run_cutedsl_jit", return_value="OUT") as run,
            patch("aot_tensor.compile.cutedsl.adapter.collect") as collect,
        ):
            self.assertEqual(op(5), "OUT")

        run.assert_called_once_with(
            cutlass_mock.cute.compile, vector_add_kernel, op.arg_specs, op.name, 5
        )
        # collection fires once (with the op + args) only when AOT compile is enabled
        expected_calls = [call(op, 5)] if enabled else []
        self.assertEqual(collect.call_args_list, expected_calls)
