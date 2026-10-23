# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import ast
import importlib
import pickle
import re
import tempfile
import unittest
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import torch
import triton
import triton.language as tl
from aot_tensor.compile.adapter_base import CompileContext, hash_spec
from aot_tensor.compile.compile_state import (
    add_spec,
    get_aott_compile_state,
    get_kernel_specs,
    register_active,
)
from aot_tensor.compile.tests.mocks import MockAutotuner
from aot_tensor.compile.triton.adapter import (
    _calls_triton_aot_kernel,
    _collect_triton_spec,
    _ensure_multi_config_autotuner,
    _extract_default_values,
    _inferred_has_perf_advantage,
    _resolve_autotune_cache,
    _resolve_call_args,
    _sample_satisfies_annotation,
    _warn_if_host_mismatches_target,
    collect,
    infer_spec,
    TritonAdapter,
    TritonAOTOperatorTransform,
    TritonCompileConfig,
)
from aot_tensor.compile.triton.spec_processing import AutotuneAttrs
from aot_tensor.constants import DEFAULT_OP_NAMESPACE_PREFIX
from aot_tensor.transform.wrapper_codegen_utils import strip_jit_unused_decorator
from aot_tensor.types import AnnotationHint, TritonAOT
from parameterized import parameterized
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import KernelInterface

_DSL_MOD = "aot_tensor.compile.triton.adapter"


POSITION_ANNOTATIONS: Dict[str, Any] = {
    "SeqEmb": ("*bf16", 16),
    "Offsets": ("*i64", 16),
    "Lengths": ("*i64", 16),
    "Out": ("*bf16", 16),
    "PosInds": ("*i32", 16),
    "TsInds": ("*i32", 16),
    "NumTargets": ("*i64", 16),
}


def _mock_position_kernel(
    SeqEmb,
    Offsets,
    Lengths,
    PosEmb,
    TsEmb,
    Out,
    TS,
    PosInds,
    TsInds,
    NumTargets,
):
    # type: (Any, Any, Any, Any, Any, Any, Any, Any, Any, Any) -> None
    pass


class _ResetStateTest(unittest.TestCase):
    """Base for tests that mutate the AOTTCompileState singleton: reset it
    before and after each test so cases do not leak state into one another."""

    def setUp(self) -> None:
        get_aott_compile_state().reset()
        register_active(TritonAdapter)

    def tearDown(self) -> None:
        get_aott_compile_state().reset()


class TritonSpecStoreTest(_ResetStateTest):
    """The DSL-keyed spec store (add_spec)."""

    @parameterized.expand(
        [
            # name, signatures to add, expected #specs, expected #hashes.
            # identical signatures collapse, distinct ones accumulate.
            ("new_single", [["*fp32", "i64"]], 1, 1),
            ("two_distinct", [["*fp32", "i64"], ["*fp16", "i32"]], 2, 2),
            ("dedup_identical", [["*fp32", "i64"], ["*fp32", "i64"]], 1, 1),
        ]
    )
    def test_add_spec_store(
        self,
        _name: str,
        signatures: List[List[Any]],
        expected_specs: int,
        expected_hashes: int,
    ) -> None:
        """add_spec keys specs by fn, dedups identical, keeps distinct."""
        mock_fn = MagicMock(spec=KernelInterface)
        for sig in signatures:
            spec: Dict[str, List[Any]] = {"signature": sig}
            add_spec(TritonAdapter.name, mock_fn, spec, hash_spec(spec))

        kernel_specs = get_kernel_specs(TritonAdapter.name)
        self.assertIn(mock_fn, kernel_specs)
        self.assertEqual(len(kernel_specs[mock_fn].specs), expected_specs)
        self.assertEqual(len(kernel_specs[mock_fn].hashes), expected_hashes)

    def test_add_spec_multiple_functions(self) -> None:
        mock_fn1 = MagicMock(spec=KernelInterface)
        mock_fn2 = MagicMock(spec=KernelInterface)

        spec1: Dict[str, List[Any]] = {"signature": ["*fp32"]}
        spec2: Dict[str, List[Any]] = {"signature": ["*fp16"]}

        add_spec(TritonAdapter.name, mock_fn1, spec1, hash_spec(spec1))
        add_spec(TritonAdapter.name, mock_fn2, spec2, hash_spec(spec2))

        kernel_specs = get_kernel_specs(TritonAdapter.name)
        self.assertIn(mock_fn1, kernel_specs)
        self.assertIn(mock_fn2, kernel_specs)
        self.assertEqual(len(kernel_specs[mock_fn1].specs), 1)
        self.assertEqual(len(kernel_specs[mock_fn2].specs), 1)


class InferSpecTest(unittest.TestCase):
    """Test the infer_spec function that maps kernel params to specs."""

    @staticmethod
    def _make_fake_kernel_with_constexpr():  # pyre-ignore[3]
        """Create a fake kernel function with tl.constexpr params."""

        def fake_kernel(
            X,
            N,
            eps,
            max_fp8: tl.constexpr = 0,
            TRAINING: tl.constexpr = False,
        ) -> None:
            pass

        fake_kernel.arg_names = ["X", "N", "eps", "max_fp8", "TRAINING"]
        return fake_kernel

    @staticmethod
    def _make_fake_kernel_no_constexpr():  # pyre-ignore[3]
        """Create a fake kernel function with no constexpr params."""

        def fake_kernel(
            X,
            N,
            alpha,
        ) -> None:
            pass

        fake_kernel.arg_names = ["X", "N", "alpha"]
        return fake_kernel

    @parameterized.expand(
        [
            (
                "constexpr_without_annotation_uses_value",
                {
                    "N": AnnotationHint("i32", 16),
                    "eps": "fp32",
                },
                {"max_fp8": 448.0, "TRAINING": True},
                [("*fp32", 16), ("i32", 16), "fp32", 448.0, True],
            ),
            (
                "constexpr_int_and_bool_values",
                {
                    "N": AnnotationHint("i32", 16),
                    "eps": "fp32",
                },
                {"max_fp8": 0, "TRAINING": False},
                [("*fp32", 16), ("i32", 16), "fp32", 0, False],
            ),
        ]
    )
    def test_infer_spec_with_constexpr(
        self,
        _name: str,
        annotations: Dict[str, Any],
        kwargs: Dict[str, Any],
        expected: List[Any],
    ) -> None:
        """Test infer_spec with tl.constexpr params and various annotation configs."""
        fake_kernel = self._make_fake_kernel_with_constexpr()
        x = torch.randn(4, 4)
        result = infer_spec(
            fake_kernel,
            annotations,
            x,
            4,
            1e-5,
            **kwargs,
        )
        self.assertEqual(result["signature"], expected)

    def test_unannotated_scalar_auto_inferred(self) -> None:
        """Non-constexpr params without annotation are auto-inferred from value type."""
        fake_kernel = self._make_fake_kernel_no_constexpr()
        x = torch.randn(4, 4)
        result = infer_spec(
            fake_kernel,
            {},
            x,
            10,
            0.5,
        )
        spec = result["signature"]
        self.assertEqual(spec[0], ("*fp32", 16))
        self.assertEqual(spec[1], "i64")
        self.assertEqual(spec[2], "fp32")

    def test_pointer_range_32_key(self) -> None:
        """Raw spec lists small tensor args for tt.pointer_range=32 on any
        host build; only AMD compile targets consume it."""
        x = torch.randn(4, 4)
        result = infer_spec(
            cast(KernelInterface[Callable[..., Any]], _two_block_kernel),
            {},
            x,
            x,
            BLOCK_M=16,
            BLOCK_N=16,
        )
        self.assertEqual(result.get("pointer_range_32"), [0, 1])

    @parameterized.expand(
        [
            ("fp64", torch.float64, "*fp64"),
            ("fp32", torch.float32, "*fp32"),
            ("fp16", torch.float16, "*fp16"),
            ("bf16", torch.bfloat16, "*bf16"),
            ("fp8e4nv", torch.float8_e4m3fn, "*fp8e4nv"),
            ("fp8e4b8", torch.float8_e4m3fnuz, "*fp8e4b8"),
            ("i32", torch.int32, "*i32"),
            ("i64", torch.int64, "*i64"),
            ("i8", torch.int8, "*i8"),
            ("u8", torch.uint8, "*u8"),
            ("i16", torch.int16, "*i16"),
            # Triton mangles bool to *u1, not *i1. Absent from this table is how
            # the *i1-keyed entry in SCALAR_TYPES went unnoticed: no test
            # exercised bool, so the dead key read as though it worked.
            ("u1", torch.bool, "*u1"),
        ]
    )
    def test_tensor_dtype_mangle_type(
        self,
        _name: str,
        torch_dtype: torch.dtype,
        expected_dtype_str: str,
    ) -> None:
        """infer_spec uses mangle_type to produce canonical Triton dtype strings."""
        fake_kernel = self._make_fake_kernel_no_constexpr()
        x = torch.zeros(4, dtype=torch_dtype)
        result = infer_spec(fake_kernel, {}, x, 10, 0.5)
        # PyTorch allocator guarantees ≥64-byte alignment → always (dtype, 16)
        self.assertEqual(result["signature"][0], (expected_dtype_str, 16))

    def test_tensor_unaligned_no_divisibility(self) -> None:
        """Unaligned tensor produces bare dtype string without alignment tuple."""
        fake_kernel = self._make_fake_kernel_no_constexpr()
        base = torch.zeros(17, dtype=torch.float32)
        x = base[1:]  # data_ptr offset by 4 bytes, not 16-byte aligned
        self.assertNotEqual(
            x.data_ptr() % 16, 0, "Test setup: tensor must be unaligned"
        )
        result = infer_spec(fake_kernel, {}, x, 10, 0.5)
        self.assertEqual(result["signature"][0], "*fp32")

    def test_unsupported_tensor_dtype_raises_with_clear_message(self) -> None:
        """The unsupported-dtype path stays reachable, on a dtype still unmapped.

        This previously used ``torch.bool``, which was correct while the table
        keyed bool on ``*i1`` -- a string Triton never emits -- so every bool
        pointer fell through here. Adding the ``*u1`` key is the point of this
        diff, and bool moves up into ``test_tensor_dtype_mangle_type`` with the
        other supported dtypes.

        The branch is kept on a genuinely absent key rather than deleted with
        the bool case: it guards codegen from a downstream KeyError, and
        SCALAR_TYPES maps only two of the fp8 variants, so ``float8_e5m2``
        exercises it for real.
        """
        fake_kernel = self._make_fake_kernel_no_constexpr()
        x = torch.zeros(4, dtype=torch.float8_e5m2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"unsupported tensor type for X.*Supported tensor dtypes",
        ):
            infer_spec(fake_kernel, {}, x, 10, 0.5)

    @parameterized.expand(
        [
            ("u64_int", 2**63, 0.5, r"unsupported int value.*i64 range"),
            ("non_constexpr_bool", True, 0.5, r"bool without.*tl.constexpr"),
        ]
    )
    def test_scalar_error_cases(
        self,
        _name: str,
        bad_value: object,
        float_arg: float,
        pattern: str,
    ) -> None:
        """Bad scalar values raise with clear error messages."""
        fake_kernel = self._make_fake_kernel_no_constexpr()
        x = torch.randn(4, 4)
        with self.assertRaisesRegex(RuntimeError, pattern):
            infer_spec(fake_kernel, {}, x, bad_value, float_arg)

    def _infer_position_signature(
        self,
        *,
        pos_dtype: torch.dtype,
        ts_dtype: torch.dtype,
        timestamp_dtype: torch.dtype,
    ) -> List[Any]:
        spec = infer_spec(
            cast(KernelInterface[Callable[..., Any]], _mock_position_kernel),
            POSITION_ANNOTATIONS,
            SeqEmb=torch.empty((16, 8), dtype=torch.bfloat16),
            Offsets=torch.empty((3,), dtype=torch.int64),
            Lengths=torch.empty((2,), dtype=torch.int64),
            PosEmb=torch.empty((32, 8), dtype=pos_dtype),
            TsEmb=torch.empty((2048, 8), dtype=ts_dtype),
            Out=torch.empty((16, 8), dtype=torch.bfloat16),
            TS=torch.empty((16,), dtype=timestamp_dtype),
            PosInds=torch.empty((16,), dtype=torch.int32),
            TsInds=torch.empty((16,), dtype=torch.int32),
            NumTargets=torch.empty((2,), dtype=torch.int64),
        )

        return spec["signature"]

    def test_position_tensors_infer_model_specific_dtypes(self) -> None:
        self.assertNotIn("PosEmb", POSITION_ANNOTATIONS)
        self.assertNotIn("TsEmb", POSITION_ANNOTATIONS)
        self.assertNotIn("TS", POSITION_ANNOTATIONS)

        grv0_signature = self._infer_position_signature(
            pos_dtype=torch.float32,
            ts_dtype=torch.float32,
            timestamp_dtype=torch.int64,
        )
        self.assertEqual(grv0_signature[3], ("*fp32", 16))
        self.assertEqual(grv0_signature[4], ("*fp32", 16))
        self.assertEqual(grv0_signature[6], ("*i64", 16))

        blue_reels_vdd_signature = self._infer_position_signature(
            pos_dtype=torch.bfloat16,
            ts_dtype=torch.bfloat16,
            timestamp_dtype=torch.float32,
        )
        self.assertEqual(blue_reels_vdd_signature[3], ("*bf16", 16))
        self.assertEqual(blue_reels_vdd_signature[4], ("*bf16", 16))
        self.assertEqual(blue_reels_vdd_signature[6], ("*fp32", 16))


class SampleSatisfiesAnnotationTest(unittest.TestCase):
    """Tests for _sample_satisfies_annotation."""

    @parameterized.expand(
        [
            ("i32_in_range", 100, "i32", True),
            ("i32_max", 2**31 - 1, "i32", True),
            ("i32_overflow", 2**31, "i32", False),
            ("i32_negative", -(2**31), "i32", True),
            ("i32_underflow", -(2**31) - 1, "i32", False),
            ("i64_always_ok", 2**62, "i64", True),
            ("fp32_always_ok", 3.14, "fp32", True),
        ]
    )
    def test_bare_string_annotation(
        self,
        _name: str,
        sample: object,
        ann: str,
        expected: bool,
    ) -> None:
        self.assertEqual(_sample_satisfies_annotation(sample, ann), expected)

    @parameterized.expand(
        [
            ("int_fits_and_div16", 128, AnnotationHint("i32", 16), True),
            ("int_fits_but_not_div16", 17, AnnotationHint("i32", 16), False),
            ("int_overflow_div16", 2**31, AnnotationHint("i32", 16), False),
            ("int_fits_and_div8", 24, AnnotationHint("i32", 8), True),
            ("int_fits_but_not_div8", 17, AnnotationHint("i32", 8), False),
            ("int_equal_to_1", 1, AnnotationHint("i32", 1), True),
            ("int_not_equal_to_1", 2, AnnotationHint("i32", 1), False),
        ]
    )
    def test_hint_int_annotation(
        self,
        _name: str,
        sample: int,
        ann: AnnotationHint,
        expected: bool,
    ) -> None:
        self.assertEqual(_sample_satisfies_annotation(sample, ann), expected)

    def test_hint_tensor_aligned(self) -> None:
        t = torch.randn(4)
        self.assertEqual(t.data_ptr() % 16, 0)
        self.assertTrue(_sample_satisfies_annotation(t, AnnotationHint("*fp32", 16)))

    def test_hint_tensor_aligned_8_but_not_16(self) -> None:
        base = torch.zeros(17, dtype=torch.int64)
        t = base[1:]
        self.assertEqual(t.data_ptr() % 8, 0)
        self.assertNotEqual(t.data_ptr() % 16, 0)
        self.assertFalse(_sample_satisfies_annotation(t, AnnotationHint("*i64", 16)))

    def test_hint_tensor_unaligned(self) -> None:
        base = torch.randn(17)
        t = base[1:]
        self.assertNotEqual(t.data_ptr() % 16, 0)
        self.assertFalse(_sample_satisfies_annotation(t, AnnotationHint("*fp32", 16)))

    def test_bare_pointer_annotation_always_satisfies(self) -> None:
        t = torch.randn(4)
        self.assertTrue(_sample_satisfies_annotation(t, "*fp32"))


class InferredHasPerfAdvantageTest(unittest.TestCase):
    """Tests for _inferred_has_perf_advantage."""

    @parameterized.expand(
        [
            (
                "inferred_adds_alignment",
                {"signature": ["*i64", "i64"]},
                {"signature": [("*i64", 16), "i64"]},
                True,
            ),
            (
                "both_have_alignment",
                {"signature": [("*i64", 16), "i64"]},
                {"signature": [("*i64", 16), "i64"]},
                False,
            ),
            (
                "inferred_widens_type_no_advantage",
                {"signature": [("*fp32", 16), "i32"]},
                {"signature": [("*fp32", 16), "i64"]},
                False,
            ),
        ]
    )
    def test_perf_advantage(
        self,
        _name: str,
        annotated: Dict[str, Any],
        inferred: Dict[str, Any],
        expected: bool,
    ) -> None:
        self.assertEqual(_inferred_has_perf_advantage(annotated, inferred), expected)


class CollectTritonSpecTest(_ResetStateTest):
    """Tests for _collect_triton_spec()."""

    @staticmethod
    def _make_fake_kernel_xn():  # pyre-ignore[3]
        def fake_kernel(
            X,
            N,
        ) -> None:
            pass

        fake_kernel.arg_names = ["X", "N"]
        return fake_kernel

    def test_collect_spec_adds_kernel_spec(self) -> None:
        fake_kernel = self._make_fake_kernel_xn()
        x = torch.randn(4, 4)
        _collect_triton_spec(fake_kernel, {}, x, 10)

        kernel_specs = get_kernel_specs(TritonAdapter.name)
        self.assertIn(fake_kernel, kernel_specs)
        self.assertEqual(len(kernel_specs[fake_kernel].specs), 1)
        self.assertEqual(
            kernel_specs[fake_kernel].specs[0]["signature"],
            [("*fp32", 16), "i64"],
        )

    @parameterized.expand(
        [
            ("bare_i32_satisfied", {"N": "i32"}, 10, 1),
            ("bare_i64_matches_inference", {"N": "i64"}, 10, 1),
            ("bare_pointer_drops_alignment", {"X": "*i64"}, 10, 2),
            ("tuple_i32_div16_satisfied", {"N": AnnotationHint("i32", 16)}, 16, 1),
            ("bare_i32_overflow_conflict", {"N": "i32"}, 3_000_000_000, 2),
            ("tuple_i32_div16_not_divisible", {"N": AnnotationHint("i32", 16)}, 17, 2),
        ]
    )
    def test_collect_spec_annotation_variant(
        self,
        _name: str,
        annotations: Dict[str, Any],
        n_value: int,
        expected_count: int,
    ) -> None:
        """Annotation-as-variant: no conflict → 1 spec, conflict → 2 specs."""
        fake_kernel = self._make_fake_kernel_xn()
        if "X" in annotations:
            x = torch.randn(4, 4).to(torch.int64)
        else:
            x = torch.randn(4, 4)
        _collect_triton_spec(fake_kernel, annotations, x, n_value)
        self.assertEqual(
            len(get_kernel_specs(TritonAdapter.name)[fake_kernel].specs), expected_count
        )

    def test_collect_spec_deduplicates(self) -> None:
        def fake_kernel(
            X,  # pyre-ignore[2]
        ) -> None:
            pass

        # pyrefly: ignore [missing-attribute]
        fake_kernel.arg_names = ["X"]

        x = torch.randn(4, 4)
        # pyrefly: ignore [bad-argument-type]
        _collect_triton_spec(fake_kernel, {}, x)
        # pyrefly: ignore [bad-argument-type]
        _collect_triton_spec(fake_kernel, {}, x)

        self.assertEqual(
            len(get_kernel_specs(TritonAdapter.name)[fake_kernel].specs), 1
        )

    def test_collect_registers_singleton_dsl(self) -> None:
        """collect() registers one reused TritonAdapter instance in dsl_state."""
        get_aott_compile_state().reset()
        marker = TritonAOT(self._make_fake_kernel_xn(), {})
        x = torch.randn(4, 4)
        collect(marker, x, 10)
        first = get_aott_compile_state().dsl_state["triton"].dsl
        collect(marker, x, 10)
        second = get_aott_compile_state().dsl_state["triton"].dsl
        self.assertIsInstance(first, TritonAdapter)
        self.assertIs(first, second)


class ResolveCallArgsTest(unittest.TestCase):
    """Tests for _resolve_call_args, especially AMD backend option filtering."""

    @staticmethod
    def _make_autotuned_kernel_with_backend_options():  # pyre-ignore[3]
        """Create a fake autotuned kernel whose configs contain AMD backend options."""

        def fake_kernel(
            X,
            N,
            BLOCK_M: tl.constexpr = 32,
            BLOCK_N: tl.constexpr = 16,
        ) -> None:
            pass

        jit_fn = triton.jit(fake_kernel)

        configs = [
            triton.Config(
                {
                    "BLOCK_M": 32,
                    "BLOCK_N": 16,
                    "matrix_instr_nonkdim": 16,
                    "waves_per_eu": 2,
                    "kpack": 2,
                },
                num_stages=2,
                num_warps=2,
            ),
        ]

        return MockAutotuner(
            fn=jit_fn,
            configs=configs,
            arg_names=list(
                fake_kernel.__code__.co_varnames[: fake_kernel.__code__.co_argcount]
            ),
        )

    def test_filters_amd_backend_options_from_autotune_configs(self) -> None:
        fake_autotuner = self._make_autotuned_kernel_with_backend_options()
        x = torch.randn(4, 4)
        triton_fn, call_args = _resolve_call_args(fake_autotuner, x, 10)

        self.assertNotIn("matrix_instr_nonkdim", call_args)
        self.assertNotIn("waves_per_eu", call_args)
        self.assertNotIn("kpack", call_args)

        self.assertIn("BLOCK_M", call_args)
        self.assertIn("BLOCK_N", call_args)
        self.assertEqual(call_args["BLOCK_M"], -1)
        self.assertEqual(call_args["BLOCK_N"], -1)

    def test_non_autotuned_kernel_unchanged(self) -> None:
        def fake_kernel(
            X,  # pyre-ignore[2]
            N,  # pyre-ignore[2]
        ) -> None:
            pass

        x = torch.randn(4, 4)
        # pyre-ignore[6]: Testing bare-function path (non-autotuner)
        triton_fn, call_args = _resolve_call_args(fake_kernel, x, 10)

        self.assertEqual(call_args["X"].shape, x.shape)
        self.assertEqual(call_args["N"], 10)

    def test_internal_kwargs_filtered(self) -> None:
        fake_autotuner = self._make_autotuned_kernel_with_backend_options()
        x = torch.randn(4, 4)
        triton_fn, call_args = _resolve_call_args(
            fake_autotuner, x, 10, warmup=False, grid=(1,)
        )
        self.assertNotIn("warmup", call_args)
        self.assertNotIn("grid", call_args)

    def test_caller_supplied_constexpr_overrides_placeholder(self) -> None:
        fake_autotuner = self._make_autotuned_kernel_with_backend_options()
        x = torch.randn(4, 4)
        triton_fn, call_args = _resolve_call_args(fake_autotuner, x, 10, BLOCK_M=128)
        self.assertEqual(call_args["BLOCK_M"], 128)
        self.assertEqual(call_args["BLOCK_N"], -1)


class EnsureMultiConfigAutotunerTest(unittest.TestCase):
    """Tests for ``_ensure_multi_config_autotuner()``."""

    @staticmethod
    def _make_config(**overrides: Any) -> triton.Config:
        base: Dict[str, Any] = {
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "BLOCK_K": 32,
            "GROUP_M": 8,
        }
        kwargs: Dict[str, Any] = dict(base, **overrides.pop("kwargs", {}))
        return triton.Config(
            kwargs,
            num_warps=overrides.pop("num_warps", 4),
            num_stages=overrides.pop("num_stages", 3),
        )

    def test_non_autotuner_is_noop(self) -> None:
        plain_fn = MagicMock(spec=KernelInterface)
        self.assertIsNone(_ensure_multi_config_autotuner(plain_fn))

    def test_single_config_is_duplicated(self) -> None:
        cfg = self._make_config()
        autotuner = MockAutotuner(configs=[cfg], cache={})
        _ensure_multi_config_autotuner(autotuner)
        self.assertEqual(len(autotuner.configs), 2)
        cfg_a, cfg_b = autotuner.configs
        self.assertEqual(cfg_a.kwargs, cfg_b.kwargs)
        self.assertEqual(cfg_a.num_warps, cfg_b.num_warps)
        self.assertEqual(cfg_a.num_stages, cfg_b.num_stages)

    def test_duplicate_is_a_copy_not_the_same_object(self) -> None:
        cfg = self._make_config()
        autotuner = MockAutotuner(configs=[cfg], cache={})
        _ensure_multi_config_autotuner(autotuner)
        cfg_a, cfg_b = autotuner.configs
        self.assertIsNot(cfg_a, cfg_b)
        self.assertIs(cfg_a, cfg)

    @parameterized.expand([("multi_config", 2), ("zero_configs", 0)])
    def test_noop_when_not_single_config(self, _name: str, num_configs: int) -> None:
        """Only a single-Config autotuner is duplicated; 0 or >1 stays unchanged."""
        cfgs = [self._make_config(num_warps=w) for w in range(num_configs)]
        autotuner = MockAutotuner(configs=list(cfgs), cache={})
        _ensure_multi_config_autotuner(autotuner)
        self.assertEqual(autotuner.configs, cfgs)

    def test_idempotent(self) -> None:
        autotuner = MockAutotuner(configs=[self._make_config()], cache={})
        _ensure_multi_config_autotuner(autotuner)
        snapshot = list(autotuner.configs)
        _ensure_multi_config_autotuner(autotuner)
        self.assertEqual(autotuner.configs, snapshot)

    def test_collect_spec_calls_helper_before_infer_spec(self) -> None:
        """Duplication must happen BEFORE ``infer_spec`` so the very first
        ``TritonAOT.run`` autotuner invocation sees ``len(configs) == 2``."""
        call_order: List[str] = []

        def fake_infer_spec(
            fn: Any,
            annotations: Dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> Dict[str, List[Any]]:
            # pyre-ignore[16]
            call_order.append(f"infer_spec(len_configs={len(fn.configs)})")
            return {"signature": ["*fp32"]}

        cfg = triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3)
        autotuner = MockAutotuner(configs=[cfg], cache={})
        state = get_aott_compile_state()
        state.reset()
        register_active(TritonAdapter)
        try:
            with patch(
                "aot_tensor.compile.triton.adapter.infer_spec",
                side_effect=fake_infer_spec,
            ):
                _collect_triton_spec(autotuner, {}, "ignored")
            self.assertEqual(len(autotuner.configs), 2)
            self.assertEqual(call_order, ["infer_spec(len_configs=2)"])
        finally:
            state.reset()


class ExtractDefaultValuesTest(unittest.TestCase):
    """Tests for _extract_default_values."""

    def test_extracts_defaults_from_python_function(self) -> None:
        def kernel_fn(x, y, allow_tf32: bool = False, BLOCK_M: int = 128) -> None:
            pass

        mock_jit_fn = MagicMock()
        mock_jit_fn.fn = kernel_fn

        defaults = _extract_default_values(mock_jit_fn)

        self.assertEqual(defaults, {"allow_tf32": False, "BLOCK_M": 128})

    def test_returns_empty_when_no_fn_attribute(self) -> None:
        mock_jit_fn = MagicMock(spec=[])

        defaults = _extract_default_values(mock_jit_fn)

        self.assertEqual(defaults, {})

    def test_skips_params_without_defaults(self) -> None:
        def kernel_fn(x, y, BLOCK_M: int = 64) -> None:
            pass

        mock_jit_fn = MagicMock()
        mock_jit_fn.fn = kernel_fn

        defaults = _extract_default_values(mock_jit_fn)

        self.assertNotIn("x", defaults)
        self.assertNotIn("y", defaults)
        self.assertEqual(defaults["BLOCK_M"], 64)


class WarnIfHostMismatchesTargetTest(unittest.TestCase):
    """Covers ``_warn_if_host_mismatches_target`` (Triton-only host/target check)."""

    @parameterized.expand(
        [
            # name, gpu_target, cuda_available, host_capability
            ("non_cuda_backend", GPUTarget("hip", "gfx942", 64), True, (9, 0)),
            ("no_cuda_host", GPUTarget("cuda", 90, 32), False, None),
            ("matching_arch", GPUTarget("cuda", 90, 32), True, (9, 0)),
        ]
    )
    def test_no_warning(
        self,
        _name: str,
        gpu_target: GPUTarget,
        cuda_available: bool,
        host_capability: Optional[Tuple[int, int]],
    ) -> None:
        with (
            patch(f"{_DSL_MOD}.torch.cuda.is_available", return_value=cuda_available),
            patch(
                f"{_DSL_MOD}.torch.cuda.get_device_capability",
                return_value=host_capability or (0, 0),
            ),
            self.assertNoLogs(_DSL_MOD, level="WARNING"),
        ):
            _warn_if_host_mismatches_target(gpu_target)

    def test_mismatched_arch_warns(self) -> None:
        with (
            patch(f"{_DSL_MOD}.torch.cuda.is_available", return_value=True),
            patch(f"{_DSL_MOD}.torch.cuda.get_device_capability", return_value=(8, 0)),
            self.assertLogs(_DSL_MOD, level="WARNING") as cm,
        ):
            _warn_if_host_mismatches_target(GPUTarget("cuda", 90, 32))
        self.assertTrue(
            any("sm_80" in m and "sm_90" in m for m in cm.output),
            f"expected host/target arch in warning, got: {cm.output}",
        )


@triton.jit
def _two_block_kernel(
    x_ptr,
    y_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Toy kernel — only ``BLOCK_M`` and ``BLOCK_N`` are kernel constexprs.
    Anything else passed via ``Config(**kwargs)`` is by definition a
    backend opt and must NOT be classified as a kernel constexpr."""
    pass


@triton.jit
def _option_parameter_kernel(x_ptr, enable_fp_fusion: tl.constexpr) -> None:
    pass


@triton.jit
def _mul_add(x_ptr, y_ptr, N, BLOCK: tl.constexpr) -> None:
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 1.5 + 1.0, mask=mask)


def strip_comments(code: str) -> str:
    """Remove Python comments from code for comparison."""
    lines = code.split("\n")
    stripped_lines = []
    for line in lines:
        stripped_line = re.sub(r"#.*$", "", line).rstrip()
        if stripped_line:
            stripped_lines.append(stripped_line)
    return "\n".join(stripped_lines)


def get_ref_import_header(abs_path: str, kernel_name: str, module_name: str) -> str:
    """Generate expected code for generate_so_loading_code tests."""
    kernel_dir = f"{module_name}_{kernel_name}"
    meta_path = f"{abs_path}/{kernel_dir}/{kernel_name}_meta.py"
    so_name = kernel_name.lstrip("_")
    so_path = f"{abs_path}/{kernel_dir}/{so_name}.so"

    return f"""import importlib.util
_meta_spec = importlib.util.spec_from_file_location("{kernel_name}_meta", "{meta_path}")
_meta_module = importlib.util.module_from_spec(_meta_spec)
_meta_spec.loader.exec_module(_meta_module)
{kernel_name}_meta = _meta_module.{kernel_name}_meta
torch.ops.load_library("{so_path}")"""


class TestAddImportHeader(unittest.TestCase):
    """Unit tests for TritonAOTOperatorTransform.generate_so_loading_code."""

    @patch(
        "aot_tensor.compile.triton.adapter.try_get_autotuner",
        return_value=None,
    )
    @patch("aot_tensor.compile.triton.adapter.unwrap_to_jit")
    def test_generate_so_loading_code(
        self,
        mock_unwrap_to_jit: MagicMock,
        mock_try_get_autotuner: MagicMock,
    ) -> None:
        """Test that generate_so_loading_code returns correct importlib.util based loading."""
        kernel_name = "_test_kernel"
        module_name = "triton_ops"
        abs_path = "/tmp/triton_aot_compile"

        mock_jit_fn = MagicMock()
        mock_jit_fn._fn_name = kernel_name
        mock_jit_fn.__module__ = f"test_module.{module_name}"

        mock_unwrap_to_jit.return_value = mock_jit_fn

        mock_kernel = MagicMock()
        fake_target = MagicMock(backend="cuda")
        transformer = TritonAOTOperatorTransform(
            kernel=mock_kernel,
            op_namespace=DEFAULT_OP_NAMESPACE_PREFIX,
            gpu_target=fake_target,
        )

        code = f"""
def test_fn():
    {kernel_name}[grid](x, y)
"""
        tree = ast.parse(code)

        import_header = transformer.generate_so_loading_code(tree, abs_path)

        expected_header = get_ref_import_header(abs_path, kernel_name, module_name)
        self.assertEqual(strip_comments(import_header), expected_header)


class StripJitUnusedDecoratorTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "strips_when_function_calls_kernel",
                "_triton_aot_swish_layer_norm",
                "_weighted_layer_norm_fwd",
                True,
                True,
                1,
            ),
            (
                "strips_for_any_name_when_calls_kernel",
                "arbitrary_function_name",
                "_weighted_layer_norm_fwd",
                True,
                True,
                1,
            ),
            (
                "preserves_when_no_kernel_call",
                "some_other_function",
                "_weighted_layer_norm_fwd",
                True,
                False,
                2,
            ),
            (
                "no_op_when_no_jit_unused",
                "_triton_aot_addmm_fwd",
                "_addmm_fwd",
                False,
                True,
                1,
            ),
        ]
    )
    def test_strip_jit_unused(
        self,
        _name: str,
        func_name: str,
        kernel_name: str,
        has_jit_unused: bool,
        calls_kernel: bool,
        expected_decorators: int,
    ) -> None:
        decorators = "@torch.jit.unused\n" if has_jit_unused else ""
        body = f"    {kernel_name}[grid](x)" if calls_kernel else "    return x"
        source = f"""
{decorators}@torch.fx.wrap
def {func_name}(x):
{body}
"""
        tree = ast.parse(source)
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)

        strip_jit_unused_decorator(
            func, lambda n: _calls_triton_aot_kernel(n, kernel_name)
        )

        self.assertEqual(len(func.decorator_list), expected_decorators)

    @parameterized.expand(
        [
            (
                "matches_kernel_subscript",
                "    grid = (N,)\n    _weighted_layer_norm_fwd[grid](x)\n    return x\n",
                "_weighted_layer_norm_fwd",
                True,
            ),
            (
                "wrong_kernel_name",
                "    grid = (N,)\n    _weighted_layer_norm_fwd[grid](x)\n    return x\n",
                "_some_other_kernel",
                False,
            ),
            ("no_call", "    return x + 1\n", "_weighted_layer_norm_fwd", False),
        ]
    )
    def test_calls_triton_aot_kernel(
        self, _name: str, body: str, kernel_name: str, expected: bool
    ) -> None:
        func = ast.parse(f"def wrapper(x):\n{body}").body[0]
        assert isinstance(func, ast.FunctionDef)
        self.assertEqual(_calls_triton_aot_kernel(func, kernel_name), expected)


class AutotuneParamsExplicitOnAllBackendOptsTest(unittest.TestCase):
    def _autotuner_with(
        self, cfg_kwargs: dict[str, int]
    ) -> triton.runtime.autotuner.Autotuner:
        autotuner = triton.runtime.autotuner.autotune(
            configs=[triton.Config(cfg_kwargs)], key=[]
        )(_two_block_kernel)
        # Pre-populate the cache so ``TritonAOTOperatorTransform`` finds a
        # representative config to introspect.
        autotuner.cache = {("dummy",): triton.Config(cfg_kwargs)}
        return autotuner

    def _params_for(self, backend: str, cfg_kwargs: dict[str, int]) -> list[str]:
        autotuner = self._autotuner_with(cfg_kwargs)
        transform = TritonAOTOperatorTransform(
            autotuner,
            op_namespace=DEFAULT_OP_NAMESPACE_PREFIX,
            gpu_target=MagicMock(backend=backend),
        )
        return transform._autotune_params

    @parameterized.expand(
        [
            ("nvidia", "cuda", {"BLOCK_M": 64, "BLOCK_N": 32}, ["BLOCK_M", "BLOCK_N"]),
            (
                "amd",
                "hip",
                {"BLOCK_M": 64, "matrix_instr_nonkdim": 16, "waves_per_eu": 2},
                ["BLOCK_M"],
            ),
        ]
    )
    def test_wrapper_explicit_on_all_backend_opts(
        self,
        _name: str,
        backend: str,
        cfg_kwargs: dict[str, int],
        expected_prefix: list[str],
    ) -> None:
        params = self._params_for(backend, cfg_kwargs)
        expected_tail = [f.name for f in AutotuneAttrs.fields_for(backend)]
        self.assertEqual(params, expected_prefix + expected_tail)

    def test_gpu_target_default_falls_back_to_driver(self) -> None:
        """When ``gpu_target`` is omitted, fall back to
        ``driver.active.get_current_target()``."""
        autotuner = self._autotuner_with({"BLOCK_M": 64, "BLOCK_N": 32})
        with patch("aot_tensor.compile.triton.adapter.driver") as mock_driver:
            mock_driver.active.get_current_target.return_value = MagicMock(
                backend="cuda"
            )
            transform = TritonAOTOperatorTransform(
                autotuner, op_namespace=DEFAULT_OP_NAMESPACE_PREFIX
            )
        mock_driver.active.get_current_target.assert_called_once()
        self.assertEqual(transform.gpu_target.backend, "cuda")


class CompileOptionsLaunchTest(_ResetStateTest):
    """``TritonAOT.run`` in front of the real collector, and the wrapper
    rewrite of the same launch: a launch keyword that repeats a compile option
    is dropped, one that changes it raises. Collection records the options the
    AOT compile then uses, whatever the marker's form."""

    def setUp(self) -> None:
        super().setUp()
        TritonAOT.set_spec_collector(collect)

    def tearDown(self) -> None:
        super().tearDown()
        TritonAOT._instances.clear()

    def _launch(
        self, marker: TritonAOT, target: GPUTarget, x: torch.Tensor, **kwargs: Any
    ) -> MagicMock:
        # Only the JIT launch and the driver are faked: no GPU here.
        with (
            patch("aot_tensor.types.driver") as config,
            patch.object(_two_block_kernel, "run", return_value="OUT") as jit_run,
        ):
            config.active.get_current_target.return_value = target
            out = marker[(1,)](x, x, BLOCK_M=16, BLOCK_N=16, **kwargs)
        self.assertEqual(out, "OUT")
        return jit_run

    @parameterized.expand(
        [
            (
                "cuda",
                GPUTarget("cuda", 90, 32),
                {"enable_fp_fusion": False},
                {"enable_fp_fusion": False},
            ),
            # HIP defines no enable_reflect_ftz, so its JIT launch drops it.
            (
                "hip",
                GPUTarget("hip", "gfx942", 64),
                {"enable_fp_fusion": False, "enable_reflect_ftz": False},
                {"enable_fp_fusion": False},
            ),
        ]
    )
    def test_repeating_a_compile_option_collects_the_plain_spec(
        self,
        _name: str,
        target: GPUTarget,
        compile_options: Dict[str, bool],
        launched: Dict[str, bool],
    ) -> None:
        x = torch.randn(4, 4)
        marker = TritonAOT(_two_block_kernel, {}, compile_options=compile_options)
        plain_spec = infer_spec(_two_block_kernel, {}, x, x, BLOCK_M=16, BLOCK_N=16)

        repeated = self._launch(marker, target, x, **compile_options)
        specs = get_kernel_specs(TritonAdapter.name)[_two_block_kernel].specs
        self.assertEqual(specs, [plain_spec])
        plain = self._launch(marker, target, x)
        self.assertEqual(specs, [plain_spec])

        self.assertIs(repeated.call_args.args[0], x)
        self.assertEqual(
            repeated.call_args.kwargs,
            {"grid": (1,), "warmup": False, "BLOCK_M": 16, "BLOCK_N": 16, **launched},
        )
        self.assertEqual(plain.call_args.kwargs, repeated.call_args.kwargs)
        # Recorded whole: the AOT compile drops what its backend does not define.
        self.assertEqual(
            get_kernel_specs(TritonAdapter.name)[_two_block_kernel].compile_options,
            compile_options,
        )

    @parameterized.expand([("collecting", True), ("jit_only", False)])
    def test_changing_a_compile_option_raises_before_collection(
        self, _name: str, collecting: bool
    ) -> None:
        if not collecting:
            TritonAOT.set_spec_collector(None)
        x = torch.randn(4, 4)
        marker = TritonAOT(
            _two_block_kernel, {}, compile_options={"enable_fp_fusion": False}
        )

        with (
            patch.object(_two_block_kernel, "run") as jit_run,
            self.assertRaisesRegex(RuntimeError, "enable_fp_fusion=True"),
        ):
            marker[(1,)](x, x, BLOCK_M=16, BLOCK_N=16, enable_fp_fusion=True)

        jit_run.assert_not_called()
        self.assertNotIn(_two_block_kernel, get_kernel_specs(TritonAdapter.name))

    def _rewrite(self, launch_keyword: str) -> ast.Module:
        marker = TritonAOT(
            _two_block_kernel, {}, compile_options={"enable_fp_fusion": False}
        )
        transform = TritonAOTOperatorTransform(
            marker,
            op_namespace=DEFAULT_OP_NAMESPACE_PREFIX,
            gpu_target=GPUTarget("cuda", 90, 32),
        )
        tree = ast.parse(
            "def launch(x, y, exact):\n"
            "    grid = (1,)\n"
            "    _two_block_kernel[grid](\n"
            f"        x, y, BLOCK_M=16, BLOCK_N=16, {launch_keyword}\n"
            "    )\n"
        )
        self.assertTrue(transform.contains_triton_call(tree))
        return transform.visit(tree)

    def test_wrapper_drops_launch_keywords_that_repeat_compile_options(self) -> None:
        (op_call,) = [
            node
            for node in ast.walk(self._rewrite("enable_fp_fusion=False"))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_two_block_kernel"
        ]
        self.assertEqual(
            [kw.arg for kw in op_call.keywords],
            ["BLOCK_M", "BLOCK_N"] + [f.name for f in AutotuneAttrs.fields_for("cuda")],
        )

    @parameterized.expand(
        [
            (
                "other_value",
                "enable_fp_fusion=True",
                " but compile_options declare enable_fp_fusion=False",
            ),
            # The rewrite cannot know a runtime value, as TritonAOT.run does.
            (
                "expression",
                "enable_fp_fusion=exact",
                ", which the wrapper rewrite cannot check against the declared "
                "enable_fp_fusion=False",
            ),
        ]
    )
    def test_wrapper_refuses_launch_keywords_that_change_compile_options(
        self, _name: str, launch_keyword: str, reason: str
    ) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            rf"_two_block_kernel is launched with {launch_keyword}{reason}",
        ):
            self._rewrite(launch_keyword)

    @parameterized.expand(
        [
            # Triton passes a Config's keys and a heuristic's value to the JIT
            # as launch keywords, so the JIT would not run with the declaration.
            (
                "config",
                lambda: triton.autotune(
                    configs=[
                        triton.Config(
                            {"BLOCK_M": 16, "BLOCK_N": 16, "enable_fp_fusion": True}
                        )
                    ],
                    key=[],
                )(_two_block_kernel),
            ),
            (
                "heuristics",
                lambda: triton.heuristics({"enable_fp_fusion": lambda _: True})(
                    _two_block_kernel
                ),
            ),
            (
                "heuristics_under_autotune",
                lambda: triton.autotune(
                    configs=[triton.Config({"BLOCK_M": 16, "BLOCK_N": 16})], key=[]
                )(
                    triton.heuristics({"enable_fp_fusion": lambda _: True})(
                        _two_block_kernel
                    )
                ),
            ),
            # The JIT binds the launch keyword to the parameter.
            ("parameter", lambda: _option_parameter_kernel),
        ]
    )
    def test_options_the_kernel_binds_itself_are_refused(
        self, _name: str, make_kernel: Callable[[], KernelInterface[Any]]
    ) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"cannot set \['enable_fp_fusion'\], which the kernel binds itself",
        ):
            TritonAOT(make_kernel(), {}, compile_options={"enable_fp_fusion": False})

    def test_options_the_kernel_does_not_bind_are_accepted(self) -> None:
        kernel = triton.autotune(
            configs=[triton.Config({"BLOCK_M": 16}, maxnreg=64)], key=[]
        )(triton.heuristics({"BLOCK_N": lambda _: 16})(_two_block_kernel))

        marker = TritonAOT(kernel, {}, compile_options={"enable_fp_fusion": False})
        self.assertEqual(marker.compile_options, {"enable_fp_fusion": False})

    def test_autotuned_launch_gets_the_options_and_the_config(self) -> None:
        TritonAOT.set_spec_collector(None)
        kernel = triton.autotune(
            configs=[triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2)],
            key=[],
        )(_two_block_kernel)
        marker = TritonAOT(kernel, {}, compile_options={"enable_fp_fusion": False})
        x = torch.randn(4, 4)

        with (
            patch("aot_tensor.types.driver") as config,
            patch.object(_two_block_kernel, "run", return_value="OUT") as jit_run,
        ):
            config.active.get_current_target.return_value = GPUTarget("cuda", 90, 32)
            out = marker[(1,)](x, x)

        self.assertEqual(out, "OUT")
        launched = jit_run.call_args.kwargs
        self.assertEqual(
            (launched["enable_fp_fusion"], launched["BLOCK_M"], launched["num_warps"]),
            (False, 16, 2),
        )

    @parameterized.expand([("options_first", 0), ("plain_first", 1)])
    def test_markers_of_one_kernel_must_declare_the_same_compile_options(
        self, _name: str, first: int
    ) -> None:
        x = torch.randn(4, 4)
        markers = [
            TritonAOT(
                _two_block_kernel, {}, compile_options={"enable_fp_fusion": False}
            ),
            TritonAOT(_two_block_kernel, {}),
        ]
        self._launch(markers[first], GPUTarget("cuda", 90, 32), x)
        collected = get_kernel_specs(TritonAdapter.name)[_two_block_kernel]
        before = (dict(collected.compile_options), list(collected.specs))

        with (
            patch("aot_tensor.types.driver"),
            patch.object(_two_block_kernel, "run") as jit_run,
            self.assertRaisesRegex(RuntimeError, "its AOT compile can use only one"),
        ):
            # A new spec, which collection would have added.
            markers[1 - first][(1,)](x, x, BLOCK_M=32, BLOCK_N=16)

        jit_run.assert_not_called()
        self.assertEqual((collected.compile_options, collected.specs), before)

    def test_markers_of_one_autotuned_kernel_must_declare_the_same_options(
        self,
    ) -> None:
        kernel = triton.autotune(
            configs=[triton.Config({"BLOCK_M": 16, "BLOCK_N": 16})], key=[]
        )(_two_block_kernel)
        unfused = TritonAOT(kernel, {}, compile_options={"enable_fp_fusion": False})
        x = torch.randn(4, 4)
        collect(unfused, x, x)

        # Recorded under the marker's fn, the autotuner the AOT compile takes.
        collected = get_kernel_specs(TritonAdapter.name)
        self.assertEqual(list(collected), [kernel])
        self.assertEqual(collected[kernel].compile_options, {"enable_fp_fusion": False})
        with self.assertRaisesRegex(RuntimeError, "its AOT compile can use only one"):
            collect(TritonAOT(kernel, {}), x, x)

    def test_options_are_recorded_when_collection_raises_partway(self) -> None:
        marker = TritonAOT(
            _mul_add, {"N": "i32"}, compile_options={"enable_fp_fusion": False}
        )
        x = torch.randn(1024)
        # The annotated spec is stored before the inferred one refuses the bool.
        with self.assertRaisesRegex(RuntimeError, "is a bool"):
            collect(marker, x, x, True, BLOCK=1024)

        collect(marker, x, x, 1024, BLOCK=1024)
        self.assertEqual(
            get_kernel_specs(TritonAdapter.name)[_mul_add].compile_options,
            {"enable_fp_fusion": False},
        )

    def test_stacked_markers_with_different_options_do_not_compile(self) -> None:
        inner = TritonAOT(
            _two_block_kernel, {}, compile_options={"enable_fp_fusion": False}
        )
        # Launching the outer marker collects under both markers' fns.
        self._launch(TritonAOT(inner, {}), GPUTarget("cuda", 90, 32), torch.randn(4, 4))
        self.assertEqual(
            [
                kspec.compile_options
                for kspec in get_kernel_specs(TritonAdapter.name).values()
            ],
            [{}, {"enable_fp_fusion": False}],
        )

        with (
            patch(f"{_DSL_MOD}.compile_to_cpp") as compile_to_cpp,
            self.assertRaisesRegex(RuntimeError, "would build into one directory"),
        ):
            self._compile_and_build()
        compile_to_cpp.assert_not_called()

    @parameterized.expand([("plain", {}), ("unfused", {"enable_fp_fusion": False})])
    def test_stacked_markers_with_the_same_options_compile(
        self, _name: str, options: Dict[str, bool]
    ) -> None:
        inner = TritonAOT(_two_block_kernel, {}, compile_options=options)
        outer = TritonAOT(inner, {}, compile_options=options)
        self._launch(outer, GPUTarget("cuda", 90, 32), torch.randn(4, 4))

        with patch(f"{_DSL_MOD}.compile_to_cpp") as compile_to_cpp:
            self._compile_and_build()
        self.assertEqual(
            [c.kwargs["compile_options"] for c in compile_to_cpp.call_args_list],
            [options, options],
        )

    @parameterized.expand([("unfused_first", True), ("plain_first", False)])
    def test_kernels_compile_with_their_own_options(
        self, _name: str, unfused_first: bool
    ) -> None:
        unfused = TritonAOT(_mul_add, {}, compile_options={"enable_fp_fusion": False})
        plain = TritonAOT(_two_block_kernel, {})
        x = torch.randn(1024)
        launches = [
            lambda: collect(unfused, x, x, 1024, BLOCK=1024),
            lambda: collect(plain, x, x, BLOCK_M=16, BLOCK_N=16),
        ]
        for launch in launches if unfused_first else reversed(launches):
            launch()

        with patch(f"{_DSL_MOD}.compile_to_cpp") as compile_to_cpp:
            self._compile_and_build()
        self.assertEqual(
            {
                c.kwargs["func"]: c.kwargs["compile_options"]
                for c in compile_to_cpp.call_args_list
            },
            {_mul_add: {"enable_fp_fusion": False}, _two_block_kernel: {}},
        )

    def _compile_and_build(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(f"{_DSL_MOD}.build_triton_aot_extension", return_value="k.so"),
        ):
            TritonAdapter().compile_and_build(
                CompileContext(
                    tmpdir,
                    importlib.import_module,
                    [TritonCompileConfig(gpu_target=GPUTarget("cuda", 80, 32))],
                )
            )

    @parameterized.expand(
        [("plain", {}, True), ("unfused", {"enable_fp_fusion": False}, False)]
    )
    def test_aot_compile_uses_the_options_collection_recorded(
        self, _name: str, compile_options: Dict[str, bool], fused: bool
    ) -> None:
        # The module attribute named after ``_mul_add`` is the JIT function, as
        # when ``triton_aot(...)`` wraps an imported kernel.
        marker = TritonAOT(_mul_add, {}, compile_options=compile_options)
        target = GPUTarget("cuda", 80, 32)
        x = torch.randn(1024)
        with (
            patch("aot_tensor.types.driver") as config,
            patch.object(_mul_add, "run"),
        ):
            config.active.get_current_target.return_value = target
            marker[(1,)](x, x, x.numel(), BLOCK=1024)

        compiled: List[Any] = []
        compile_ = triton.compiler.compile

        def spy(*args: Any, **kwargs: Any) -> Any:
            compiled.append(compile_(*args, **kwargs))
            return compiled[-1]

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "aot_tensor.compile.triton.pipeline.triton.compiler.compile",
                side_effect=spy,
            ),
            patch(f"{_DSL_MOD}.build_triton_aot_extension", return_value="k.so"),
        ):
            TritonAdapter().compile_and_build(
                CompileContext(
                    tmpdir,
                    importlib.import_module,
                    [TritonCompileConfig(gpu_target=target)],
                )
            )

        (kernel,) = compiled
        self.assertEqual(kernel.metadata.enable_fp_fusion, fused)
        self.assertEqual("fma.rn.f32" in kernel.asm["ptx"], fused)
        self.assertEqual("mul.rn.f32" in kernel.asm["ptx"], not fused)

    @parameterized.expand(
        [
            ("jit", lambda: _two_block_kernel, {"BLOCK_M": 16, "BLOCK_N": 16}),
            (
                "autotuned",
                lambda: triton.autotune(
                    configs=[triton.Config({"BLOCK_M": 16, "BLOCK_N": 16})], key=[]
                )(_two_block_kernel),
                {},
            ),
        ]
    )
    def test_wrapper_needs_the_options_the_kernel_was_compiled_with(
        self,
        _name: str,
        make_kernel: Callable[[], KernelInterface[Any]],
        launch_kwargs: Dict[str, int],
    ) -> None:
        kernel = make_kernel()
        plain = TritonAOT(kernel, {})
        x = torch.randn(4, 4)
        collect(plain, x, x, **launch_kwargs)
        unfused = TritonAOT(kernel, {}, compile_options={"enable_fp_fusion": False})

        with (
            patch(f"{_DSL_MOD}.TritonAOTOperatorTransform"),
            patch(f"{_DSL_MOD}.generate_wrapper_files_skeleton") as skeleton,
        ):
            TritonAdapter().generate_wrapper_files(MagicMock(), plain, "/unused", None)
            with self.assertRaisesRegex(
                RuntimeError,
                r"_two_block_kernel was compiled with compile_options \{\}, but .* "
                r"declares \{'enable_fp_fusion': False\}",
            ):
                TritonAdapter().generate_wrapper_files(
                    MagicMock(), unfused, "/unused", None
                )
        skeleton.assert_called_once()


class TritonAOTDynamoTest(unittest.TestCase):
    """``torch.compile`` and strict ``torch.export`` trace ``TritonAOT.run``
    with dynamo, which cannot trace into the Triton driver."""

    def tearDown(self) -> None:
        TritonAOT._instances.clear()

    def test_a_marker_without_compile_options_traces_to_its_kernel(self) -> None:
        class _Kernel(KernelInterface[Callable[..., Any]]):
            def run(self, x: torch.Tensor, *, grid: Any, warmup: bool) -> torch.Tensor:
                return x + 1

        marker = TritonAOT(_Kernel(), {})
        traced = torch.compile(
            lambda x: marker[(1,)](x), backend="eager", fullgraph=True
        )
        self.assertEqual(traced(torch.zeros(1)).item(), 1.0)


class _ReducesToEmptyDict(dict):
    """Pickles and deep-copies to ``{}``, like Triton 3.8's ``_AutotuneCache``."""

    def __reduce__(self) -> tuple[type[dict], tuple[()]]:
        return (dict, ())


class ResolveAutotuneCacheTest(unittest.TestCase):
    def test_dump_keeps_entries_of_a_cache_that_pickles_to_empty(self) -> None:
        cache = _ReducesToEmptyDict({(64,): triton.Config({"BLOCK": 64}, num_warps=4)})
        # @lint-ignore PYTHONPICKLEISBAD
        self.assertEqual(pickle.loads(pickle.dumps(cache)), {})
        with tempfile.TemporaryDirectory() as fn_dir:
            _resolve_autotune_cache(MockAutotuner(cache=cache), "k", fn_dir, {})
            with open(f"{fn_dir}/k_autotune_cache", "rb") as f:
                # @lint-ignore PYTHONPICKLEISBAD
                dumped = pickle.load(f)
        self.assertEqual(
            {k: (c.kwargs, c.num_warps) for k, c in dumped.items()},
            {(64,): ({"BLOCK": 64}, 4)},
        )
