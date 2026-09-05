# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors[2]: triton func without type

import dataclasses
import logging
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# @manual=//triton:triton
import triton
import triton.language as tl
from aot_tensor.compile.spec_conversion import (
    collect_constraints,
    SignatureConstraints,
    SignatureElement,
)
from aot_tensor.compile.triton.spec_processing import (
    _autotune_specs,
    _check_constants_consistency,
    _check_optional_consistency,
    _check_signature_consistency,
    _check_uniform_signature_length,
    _compute_constexpr_keys,
    _dedup_specs,
    _detect_optional_args,
    _max_dynamic_shared_per_block_optin,
    _validate_converted_specs,
    _wider_type,
    AutotuneAttrs,
    compute_autotune_param_names,
    gen_compile_arg,
    KernelSpec,
    OpsUnit,
    RawKernelSpec,
)
from aot_tensor.compile.triton.utils import kernel_param_names
from parameterized import parameterized
from triton.backends.compiler import GPUTarget
from triton.compiler.code_generator import ASTFunction


@triton.jit
def _addmm_fwd(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
) -> None:
    """Dummy kernel for testing - just a placeholder."""
    pass


@triton.jit
def _reduction_like_fwd(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_BLOCK: tl.constexpr,
) -> None:
    """Mirrors prime_perf_optimizer ``_gemm_reduction_kernel`` shape.

    BLOCK_M / BLOCK_K are autotune-d. N_BLOCK is *not* autotune-d -- it
    is computed by the launcher as ``next_power_of_2(N)`` and passed as
    a constexpr per call site, so it varies across calls without ever
    appearing in ``cfg.kwargs``.
    """
    pass


def _make_kernel_spec(**kwargs: Any) -> KernelSpec:
    """Build a KernelSpec with sensible defaults for tests."""
    defaults: Dict[str, Any] = {
        "signature": {},
        "constants": {},
        "divisible_by_16": set(),
        "divisible_by_8": set(),
    }
    defaults.update(kwargs)
    return KernelSpec(**defaults)


class GenCompileArgTest(unittest.TestCase):
    def test_ops_divisibility_info_reaches_triton(self) -> None:
        """Test that divisibility info from triton_aot/ops reaches triton IR."""
        raw_spec: Dict[str, Any] = {
            "signature": [
                ("*fp32", 16),
                ("*fp32", 16),
                ("*fp32", 16),
                ("*fp32", 16),
                ("i32", 16),
                ("i32", 16),
                ("i32", 16),
            ],
        }

        unit = OpsUnit.from_raw_specs([raw_spec], GPUTarget("cuda", 80, 32))
        spec = unit.specs[0]

        self.assertEqual(
            spec.divisible_by_16,
            {0, 1, 2, 3, 4, 5, 6},
            "convert_specs should identify all args with 16 hint as divisible_by_16",
        )

        result = gen_compile_arg(spec, _addmm_fwd)
        ast_source = result[0]

        ast_function = ASTFunction(
            ret_types=[],
            arg_types=[],
            constants=ast_source.constants,
            attrs=ast_source.attrs,
        )

        triton_would_set_divisibility_for: List[int] = []
        for idx in spec.divisible_by_16:
            path = (idx,)
            attr_specs = ast_function.attrs.get(path, [])
            if attr_specs:
                for attr_name, _ in attr_specs:
                    if attr_name == "tt.divisibility":
                        triton_would_set_divisibility_for.append(idx)

        self.assertEqual(
            set(triton_would_set_divisibility_for),
            spec.divisible_by_16,
            "Triton should set tt.divisibility for all divisible_by_16 args. "
            f"Expected: {spec.divisible_by_16}, "
            f"Got: {set(triton_would_set_divisibility_for)}. ",
        )

    def test_signature_translated_by_param_name(self) -> None:
        """spec.signature[idx] → ASTSource.signature[param_name_at_idx]."""
        spec = KernelSpec(
            signature={0: "*fp32", 4: "i32", 5: "i32", 6: "i32"},
            constants={},
            divisible_by_16=set(),
            divisible_by_8=set(),
        )
        ast_source = gen_compile_arg(spec, _addmm_fwd)[0]
        # _addmm_fwd's first param is x_ptr; index 4-6 are M/N/K.
        self.assertEqual(ast_source.signature["x_ptr"], "*fp32")
        self.assertEqual(ast_source.signature["M"], "i32")
        self.assertEqual(ast_source.signature["N"], "i32")
        self.assertEqual(ast_source.signature["K"], "i32")

    def test_constants_become_constexpr_in_signature(self) -> None:
        """A constant arg gets ASTSource.constants entry AND signature='constexpr'."""
        spec = KernelSpec(
            signature={0: "*fp32"},
            constants={15: 64},  # BLOCK_M
            divisible_by_16=set(),
            divisible_by_8=set(),
        )
        ast_source = gen_compile_arg(spec, _addmm_fwd)[0]
        self.assertEqual(ast_source.signature["BLOCK_M"], "constexpr")
        self.assertEqual(ast_source.constants[(15,)], 64)

    def test_divisible_by_16_emits_divisibility_attr(self) -> None:
        """Each idx in spec.divisible_by_16 gets a (idx,) → tt.divisibility=16 entry."""
        spec = KernelSpec(
            signature={0: "*fp32", 1: "*fp32"},
            constants={},
            divisible_by_16={0, 1},
            divisible_by_8=set(),
        )
        ast_source = gen_compile_arg(spec, _addmm_fwd)[0]
        for idx in (0, 1):
            self.assertIn((idx,), ast_source.attrs)
            self.assertEqual(ast_source.attrs[(idx,)], [["tt.divisibility", 16]])


class ValidateConvertedSpecsTest(unittest.TestCase):
    """Tests for _validate_converted_specs()."""

    def test_in_range_indices_pass(self) -> None:
        spec = _make_kernel_spec(
            signature={0: "*fp32", 2: "i32"},
            constants={3: 64},
            divisible_by_16={0},
        )
        # Should not raise; method returns None.
        self.assertIsNone(
            _validate_converted_specs([spec], optional=set(), num_params=4)
        )

    @parameterized.expand(
        [
            ("signature_too_large", {"signature": {0: "*fp32", 99: "i32"}}),
            ("constants_too_large", {"constants": {42: 64}}),
            ("divisible_by_16_too_large", {"divisible_by_16": {5}}),
            ("divisible_by_8_too_large", {"divisible_by_8": {7}}),
            ("signature_negative", {"signature": {-1: "*fp32"}}),
        ]
    )
    def test_out_of_range_idx_raises(
        self, _name: str, spec_kwargs: Dict[str, Any]
    ) -> None:
        spec = _make_kernel_spec(**spec_kwargs)
        with self.assertRaisesRegex(ValueError, r"out of range"):
            _validate_converted_specs([spec], optional=set(), num_params=4)

    def test_num_params_zero_skips_bound_check(self) -> None:
        """When num_params=0 (default), bound check is disabled — back-compat."""
        spec = _make_kernel_spec(signature={99: "*fp32"})
        # Should not raise even with out-of-range idx, since num_params=0.
        self.assertIsNone(_validate_converted_specs([spec], optional=set()))


class CheckUniformSignatureLengthTest(unittest.TestCase):
    """Tests for _check_uniform_signature_length()."""

    def test_empty_returns_zero(self) -> None:
        self.assertEqual(_check_uniform_signature_length([]), 0)

    def test_single_spec_returns_its_length(self) -> None:
        specs: list[RawKernelSpec] = [{"signature": [("*fp32", 16), "i32", None]}]
        self.assertEqual(_check_uniform_signature_length(specs), 3)

    def test_uniform_lengths_return_common_count(self) -> None:
        specs: list[RawKernelSpec] = [
            {"signature": [("*fp32", 16), "i32"]},
            {"signature": [("*fp32", 16), "i32"]},
            {"signature": [("*bf16", 16), "i64"]},
        ]
        self.assertEqual(_check_uniform_signature_length(specs), 2)

    def test_mismatched_lengths_raise(self) -> None:
        specs: list[RawKernelSpec] = [
            {"signature": [("*fp32", 16), "i32"]},
            {"signature": [("*fp32", 16), "i32", "i32"]},
        ]
        with self.assertRaisesRegex(ValueError, r"inconsistent signature lengths"):
            _check_uniform_signature_length(specs)


class CheckOptionalConsistencyTest(unittest.TestCase):
    """Tests for _check_optional_consistency()."""

    def test_pointer_and_none_pass(self) -> None:
        """Pointer in one spec, None in another — valid optional pattern."""
        ref = _make_kernel_spec(signature={0: "*fp32"})
        spec = _make_kernel_spec(constants={0: None})
        self.assertIsNone(_check_optional_consistency(ref, spec, idx=1, optional={0}))

    @parameterized.expand(
        [
            (
                "non_pointer_type",
                {"signature": {0: "i32"}},
                {"constants": {0: None}},
                r"non-pointer type",
            ),
            (
                "non_none_constant",
                {"signature": {0: "*fp32"}},
                {"constants": {0: 42}},
                r"non-None constant",
            ),
        ]
    )
    def test_invalid_optional_raises(
        self,
        _name: str,
        ref_kwargs: Dict[str, Any],
        spec_kwargs: Dict[str, Any],
        pattern: str,
    ) -> None:
        ref = _make_kernel_spec(**ref_kwargs)
        spec = _make_kernel_spec(**spec_kwargs)
        with self.assertRaisesRegex(ValueError, pattern):
            _check_optional_consistency(ref, spec, idx=1, optional={0})

    def test_empty_optional_no_checks(self) -> None:
        """No optional positions → no checks performed."""
        ref = _make_kernel_spec(signature={0: "i32"})
        spec = _make_kernel_spec(signature={0: "i64"})
        self.assertIsNone(_check_optional_consistency(ref, spec, idx=1, optional=set()))


class CheckSignatureConsistencyTest(unittest.TestCase):
    """Tests for _check_signature_consistency() — cross-spec dtype invariance.

    Compatible int widths (i32/i64) are allowed to coexist — they are
    handled by ``_wider_type`` in ``_compute_invariants`` and int range
    guards in ``gen_guarded_calls``.
    """

    def test_same_dtype_passes(self) -> None:
        ref = _make_kernel_spec(signature={1: "i32", 2: "fp32"})
        spec = _make_kernel_spec(signature={1: "i32", 2: "fp32"})
        self.assertIsNone(
            _check_signature_consistency(ref, spec, idx=1, optional=set())
        )

    def test_compatible_int_widths_pass(self) -> None:
        """i32 vs i64 is allowed (handled by _wider_type + int range guard)."""
        ref = _make_kernel_spec(signature={1: "i32"})
        spec = _make_kernel_spec(signature={1: "i64"})
        self.assertIsNone(
            _check_signature_consistency(ref, spec, idx=1, optional=set())
        )

    def test_incompatible_dtype_mismatch_raises(self) -> None:
        """fp32 vs i32 still raises (not a compatible width pair)."""
        ref = _make_kernel_spec(signature={1: "fp32"})
        spec = _make_kernel_spec(signature={1: "i32"})
        with self.assertRaisesRegex(ValueError, r"dtype mismatch"):
            _check_signature_consistency(ref, spec, idx=1, optional=set())

    def test_partition_difference_allowed(self) -> None:
        # An arg in signature in one spec and absent (in constants) in
        # another is allowed — annotation-as-variant pattern.
        ref = _make_kernel_spec(signature={1: "i32"})
        spec = _make_kernel_spec(signature={})
        self.assertIsNone(
            _check_signature_consistency(ref, spec, idx=1, optional=set())
        )

    def test_pointer_args_skip_check(self) -> None:
        # Pointer dtypes may differ across specs — covered by Phase 4 variants
        # or upstream constraints, not by this consistency check.
        ref = _make_kernel_spec(signature={0: "*fp32"})
        spec = _make_kernel_spec(signature={0: "*bf16"})
        self.assertIsNone(
            _check_signature_consistency(ref, spec, idx=1, optional=set())
        )

    def test_optional_args_skip_check(self) -> None:
        # Optional tensor args legitimately differ (pointer vs None).
        ref = _make_kernel_spec(signature={0: "*fp32"})
        spec = _make_kernel_spec(signature={})  # None side has no signature entry
        self.assertIsNone(_check_signature_consistency(ref, spec, idx=1, optional={0}))


class CheckConstantsConsistencyTest(unittest.TestCase):
    """Tests for _check_constants_consistency() — cross-spec constant type invariance."""

    def test_same_type_passes(self) -> None:
        ref = _make_kernel_spec(constants={3: 64, 4: True})
        spec = _make_kernel_spec(
            constants={3: 128, 4: False}
        )  # diff values, same types
        self.assertIsNone(
            _check_constants_consistency(ref, spec, idx=1, optional=set())
        )

    def test_type_mismatch_raises(self) -> None:
        ref = _make_kernel_spec(constants={3: 64})  # int
        spec = _make_kernel_spec(constants={3: 64.0})  # float
        with self.assertRaisesRegex(ValueError, r"constant type mismatch"):
            _check_constants_consistency(ref, spec, idx=1, optional=set())

    def test_missing_in_one_spec_silently_skipped(self) -> None:
        # Missing key on one side resolves to None via dict.get(); the
        # check skips comparison.  Constant partition is enforced upstream
        # by Triton's tl.constexpr declarations being identical across
        # call sites of the same kernel.
        ref = _make_kernel_spec(constants={3: 64})
        spec = _make_kernel_spec(constants={})
        self.assertIsNone(
            _check_constants_consistency(ref, spec, idx=1, optional=set())
        )

    def test_none_constant_skips_check(self) -> None:
        # None on either side (optional tensor) is handled by
        # _detect_optional_args / _check_optional_consistency, not by this check.
        ref = _make_kernel_spec(constants={3: None})
        spec = _make_kernel_spec(constants={3: 64})
        self.assertIsNone(
            _check_constants_consistency(ref, spec, idx=1, optional=set())
        )

    def test_optional_args_skip_check(self) -> None:
        ref = _make_kernel_spec(constants={3: 64})
        spec = _make_kernel_spec(constants={3: 64.0})  # type mismatch but optional
        self.assertIsNone(_check_constants_consistency(ref, spec, idx=1, optional={3}))


class ConvertSpecsFP8Test(unittest.TestCase):
    """Tests for OpsUnit.from_raw_specs() FP8 dtype handling across GPU targets."""

    @parameterized.expand(
        [
            (
                "native_fp8_on_gfx942",
                "*fp8e4b8",
                GPUTarget("hip", "gfx942", 64),
                "*fp8e4b8",
            ),
            (
                "nv_fp8_converted_on_gfx942",
                "*fp8e4nv",
                GPUTarget("hip", "gfx942", 64),
                "*fp8e4b8",
            ),
            (
                "nv_fp8_kept_on_mi350x",
                "*fp8e4nv",
                GPUTarget("hip", "gfx950", 64),
                "*fp8e4nv",
            ),
            (
                "fp8_falls_back_to_bf16_on_sm80",
                "*fp8e4nv",
                GPUTarget("cuda", 80, 32),
                "*bf16",
            ),
            (
                "fp8_no_fallback_on_sm90",
                "*fp8e4nv",
                GPUTarget("cuda", 90, 32),
                "*fp8e4nv",
            ),
        ]
    )
    def test_fp8_dtype_conversion(
        self,
        _name: str,
        input_dtype: str,
        gpu_target: GPUTarget,
        expected_dtype: str,
    ) -> None:
        base_specs: list[RawKernelSpec] = [{"signature": [(input_dtype, 16)]}]
        unit = OpsUnit.from_raw_specs(base_specs, gpu_target)
        self.assertEqual(unit.specs[0].signature[0], expected_dtype)


class ConvertSpecsTest(unittest.TestCase):
    """Tests for OpsUnit.from_raw_specs() edge cases."""

    def test_single_spec_none_constant(self) -> None:
        """Single spec with None arg: constants, pointer_args, signature all correct."""
        base_specs: list[RawKernelSpec] = [
            {"signature": [("*fp32", 16), None, ("i32", 16)]}
        ]
        unit = OpsUnit.from_raw_specs(base_specs, GPUTarget("cuda", 90, 32))
        spec = unit.specs[0]

        self.assertNotIn(1, spec.signature)
        self.assertIn(1, spec.constants)
        self.assertIsNone(spec.constants[1])
        self.assertIn(1, unit.pointer_args)
        self.assertIn(0, unit.pointer_args)
        self.assertNotIn(2, unit.pointer_args)
        self.assertEqual(spec.signature[0], "*fp32")
        self.assertEqual(spec.signature[2], "i32")

    @parameterized.expand(
        [
            ("nvidia_sm80", GPUTarget("cuda", 80, 32), 80),
            ("amd_gfx942", GPUTarget("hip", "gfx942", 64), "gfx942"),
        ]
    )
    def test_cc_set_from_gpu_target(
        self, _name: str, gpu_target: GPUTarget, expected_cc: object
    ) -> None:
        """OpsUnit.cc comes from gpu_target.arch."""
        unit = OpsUnit.from_raw_specs([{"signature": [("*fp32", 16)]}], gpu_target)
        self.assertEqual(unit.cc, expected_cc)

    def test_optional_set_from_3tuple(self) -> None:
        """OpsUnit.optional populated from 3-tuple signature elements."""
        base_specs: list[RawKernelSpec] = [
            {"signature": [("*fp32", 16, True), ("i32", 16)]},
            {"signature": [("*fp32", 16, False), ("i32", 16)]},
        ]
        unit = OpsUnit.from_raw_specs(base_specs, GPUTarget("cuda", 90, 32))
        self.assertIn(0, unit.optional)
        self.assertNotIn(1, unit.optional)

    def test_scalar_dtypes_computed(self) -> None:
        """scalar_dtypes maps non-pointer signature args to their dtypes."""
        base_specs: list[RawKernelSpec] = [
            {"signature": [("*fp32", 16), ("i32", 16), ("fp32", 8)]}
        ]
        unit = OpsUnit.from_raw_specs(base_specs, GPUTarget("cuda", 90, 32))
        self.assertEqual(unit.scalar_dtypes[1], "i32")
        self.assertEqual(unit.scalar_dtypes[2], "fp32")
        self.assertNotIn(0, unit.scalar_dtypes)

    def test_constant_types_computed(self) -> None:
        """constant_types maps constant arg indices to their Python types."""
        base_specs: list[RawKernelSpec] = [
            {"signature": [("*fp32", 16), ("i32", 16), 128, "relu"]}
        ]
        unit = OpsUnit.from_raw_specs(base_specs, GPUTarget("cuda", 90, 32))
        self.assertEqual(unit.constant_types[2], int)
        self.assertEqual(unit.constant_types[3], str)


class DedupSpecsTest(unittest.TestCase):
    """Tests for dedup_specs()."""

    def test_identical_specs_deduped(self) -> None:
        """Duplicate specs are removed."""
        spec = KernelSpec(
            signature={0: "*fp32"},
            constants={1: 64},
            divisible_by_16={0},
            divisible_by_8=set(),
        )
        result = _dedup_specs([spec, spec, spec])
        self.assertEqual(len(result), 1)

    def test_different_specs_kept(self) -> None:
        """Distinct specs are all kept."""
        spec_a = KernelSpec(
            signature={0: "*fp32"},
            constants={1: 64},
            divisible_by_16={0},
            divisible_by_8=set(),
        )
        spec_b = KernelSpec(
            signature={0: "*fp16"},
            constants={1: 128},
            divisible_by_16={0},
            divisible_by_8=set(),
        )
        result = _dedup_specs([spec_a, spec_b])
        self.assertEqual(len(result), 2)

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        self.assertEqual(_dedup_specs([]), [])


class SpecConsistencyTest(unittest.TestCase):
    """Ensure KernelSpec/OpsUnit stay in sync with SignatureConstraints.

    KernelSpec absorbs fields from SignatureConstraints (shared/spec_conversion.py).
    If that contract changes without a corresponding KernelSpec/OpsUnit update,
    these tests break.
    """

    def test_convert_specs_maps_all_constraint_fields(self) -> None:
        """from_raw_specs must propagate all SignatureConstraints fields to KernelSpec/OpsUnit."""
        sig: list[SignatureElement] = [
            ("*fp32", 16),  # idx 0: pointer, div16, div8
            ("i32", 1),  # idx 1: scalar, value=1 → constant
            ("*fp32", 16, True),  # idx 2: optional, has_value, div16
            ("*fp32", 16, False),  # idx 3: optional, no value → none_arg
        ]
        constraints = collect_constraints(sig)
        unit = OpsUnit.from_raw_specs([{"signature": sig}], GPUTarget("cuda", 90, 32))
        spec = unit.specs[0]

        self.assertEqual(spec.divisible_by_16, constraints.divisible_by_16)
        self.assertEqual(spec.divisible_by_8, constraints.divisible_by_8)
        self.assertEqual(unit.optional, constraints.optional_args)

    def test_signature_constraints_fields_covered(self) -> None:
        """Every SignatureConstraints field must have a KernelSpec or OpsUnit counterpart."""
        constraint_fields = {f.name for f in dataclasses.fields(SignatureConstraints)}
        spec_fields = {f.name for f in dataclasses.fields(KernelSpec)}
        unit_fields = {f.name for f in dataclasses.fields(OpsUnit)}
        all_target_fields = spec_fields | unit_fields

        consumed_during_conversion = {"none_args", "has_fp8", "equal_to_1"}

        field_mapping = {
            "divisible_by_16": "divisible_by_16",
            "divisible_by_8": "divisible_by_8",
            "optional_args": "optional",
        }

        for c_field in constraint_fields - consumed_during_conversion:
            mapped = field_mapping.get(c_field, c_field)
            self.assertIn(
                mapped,
                all_target_fields,
                f"SignatureConstraints.{c_field} → {mapped} is missing from "
                "both KernelSpec and OpsUnit.",
            )


class AutotuneAttrsTest(unittest.TestCase):
    """Tests for ``AutotuneAttrs`` (SoT for autotune backend opt fields)."""

    def test_autotune_fields_declaration_order(self) -> None:
        self.assertEqual(
            [f.name for f in dataclasses.fields(AutotuneAttrs)],
            [
                "num_warps",
                "num_stages",
                "matrix_instr_nonkdim",
                "waves_per_eu",
                "kpack",
                "num_ctas",
                "auto_tma",
            ],
        )

    def test_autotune_fields_partition_by_platform(self) -> None:
        """Every field belongs to exactly one of COMMON / AMD_ONLY / NVIDIA_ONLY."""
        all_fields = {f.name for f in dataclasses.fields(AutotuneAttrs)}
        partition = (
            AutotuneAttrs.COMMON_FIELDS
            | AutotuneAttrs.AMD_ONLY_FIELDS
            | AutotuneAttrs.NVIDIA_ONLY_FIELDS
        )
        self.assertEqual(all_fields, partition)
        self.assertEqual(
            AutotuneAttrs.COMMON_FIELDS & AutotuneAttrs.AMD_ONLY_FIELDS, set()
        )
        self.assertEqual(
            AutotuneAttrs.COMMON_FIELDS & AutotuneAttrs.NVIDIA_ONLY_FIELDS, set()
        )
        self.assertEqual(
            AutotuneAttrs.AMD_ONLY_FIELDS & AutotuneAttrs.NVIDIA_ONLY_FIELDS, set()
        )

    @parameterized.expand(
        [
            ("cuda", AutotuneAttrs.COMMON_FIELDS | AutotuneAttrs.NVIDIA_ONLY_FIELDS),
            ("hip", AutotuneAttrs.COMMON_FIELDS | AutotuneAttrs.AMD_ONLY_FIELDS),
        ]
    )
    def test_fields_for_backend(self, backend: str, expected: frozenset[str]) -> None:
        names = {f.name for f in AutotuneAttrs.fields_for(backend)}
        self.assertEqual(names, expected)

    def test_fields_for_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            AutotuneAttrs.fields_for("rocm")

    def test_all_autotune_fields_have_defaults(self) -> None:
        """Required for ``_autotune_specs`` to materialize via ``AutotuneAttrs(**values)``."""
        for f in dataclasses.fields(AutotuneAttrs):
            has_default = (
                f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING
            )
            self.assertTrue(has_default, f"{f.name!r} missing default")

    def test_all_autotune_fields_are_int_or_bool(self) -> None:
        """Codegen derives C++ types via ``AutotuneAttrs.field_python_types()``
        (annotation-driven). Currently ``int`` (->int64_t) and ``bool`` (->bool)
        are supported; adding another type requires updating
        ``PY_TYPES_TO_CPP_TYPES`` + default rendering in ``compile/codegen.py``."""
        for name, py_type in AutotuneAttrs.field_python_types().items():
            self.assertIn(py_type, (int, bool), name)

    def test_all_autotune_fields_have_cubin_short(self) -> None:
        """``gen_kernel_name`` reads ``f.metadata['cubin_short']`` per field;
        a missing entry would KeyError at codegen time."""
        for f in dataclasses.fields(AutotuneAttrs):
            self.assertIn("cubin_short", f.metadata, f.name)
            self.assertIsInstance(f.metadata["cubin_short"], str)

    def test_field_python_types_covers_all_fields(self) -> None:
        """``field_python_types()`` must expose every declared field."""
        annotated = AutotuneAttrs.field_python_types()
        for f in dataclasses.fields(AutotuneAttrs):
            self.assertIn(f.name, annotated)

    def test_kernel_spec_autotune_is_per_instance(self) -> None:
        """Guard the default-factory: each KernelSpec must own its AutotuneAttrs."""
        a = KernelSpec(
            signature={}, constants={}, divisible_by_16=set(), divisible_by_8=set()
        )
        b = KernelSpec(
            signature={}, constants={}, divisible_by_16=set(), divisible_by_8=set()
        )
        self.assertIsNot(a.autotune, b.autotune)


class KernelParamNamesTest(unittest.TestCase):
    """Tests for ``kernel_param_names()``."""

    def test_kernel_param_names_plain_jit(self) -> None:
        params = kernel_param_names(_addmm_fwd)
        self.assertIn("x_ptr", params)
        self.assertIn("BLOCK_M", params)
        self.assertIn("BROADCAST_Y", params)
        # Backend opts (NVIDIA + AMD) are NOT kernel signature params.
        for backend_opt in (
            "num_warps",
            "num_stages",
            "num_ctas",
            "matrix_instr_nonkdim",
            "waves_per_eu",
            "kpack",
        ):
            self.assertNotIn(backend_opt, params)

    @parameterized.expand(
        [
            (
                "autotuner",
                triton.autotune(
                    configs=[
                        triton.Config(
                            {
                                "BLOCK_M": 32,
                                "BLOCK_N": 32,
                                "BLOCK_K": 32,
                                "GROUP_M": 4,
                            }
                        )
                    ],
                    key=["M", "N"],
                ),
            ),
            (
                "heuristics",
                triton.heuristics({"BROADCAST_Y": lambda _meta: 1}),
            ),
            (
                "autotune_over_heuristics",
                lambda fn: triton.autotune(
                    configs=[
                        triton.Config(
                            {
                                "BLOCK_M": 32,
                                "BLOCK_N": 32,
                                "BLOCK_K": 32,
                                "GROUP_M": 4,
                            }
                        )
                    ],
                    key=["M", "N"],
                )(triton.heuristics({"BROADCAST_Y": lambda _meta: 1})(fn)),
            ),
        ]
    )
    def test_unwraps_decorator_wrappers(self, name: str, wrap: Any) -> None:
        """Autotune / heuristics (and nested) wrappers peel back to JITFunction."""
        self.assertEqual(
            kernel_param_names(wrap(_addmm_fwd)), kernel_param_names(_addmm_fwd)
        )


class AutotuneAttrsFromCfgTest(unittest.TestCase):
    """Tests for ``AutotuneAttrs.from_cfg()``."""

    def test_from_cfg_amd_opts_from_kwargs(self) -> None:
        cfg = triton.Config(
            {"BLOCK_M": 64, "matrix_instr_nonkdim": 16, "waves_per_eu": 2, "kpack": 2},
            num_warps=4,
            num_stages=3,
        )
        attrs = AutotuneAttrs.from_cfg(cfg, AutotuneAttrs.fields_for("hip"))
        self.assertEqual(attrs.matrix_instr_nonkdim, 16)
        self.assertEqual(attrs.waves_per_eu, 2)
        self.assertEqual(attrs.kpack, 2)
        self.assertEqual(attrs.num_warps, 4)
        self.assertEqual(attrs.num_stages, 3)

    def test_from_cfg_nvidia_opts_from_attrs(self) -> None:
        cfg = triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=5)
        attrs = AutotuneAttrs.from_cfg(cfg, AutotuneAttrs.fields_for("cuda"))
        self.assertEqual(attrs.num_warps, 8)
        self.assertEqual(attrs.num_stages, 5)

    def test_from_cfg_missing_field_uses_default(self) -> None:
        """Missing AMD field on hip backend falls back to AutotuneAttrs default."""
        cfg = triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3)
        attrs = AutotuneAttrs.from_cfg(cfg, AutotuneAttrs.fields_for("hip"))
        self.assertEqual(attrs.matrix_instr_nonkdim, 0)  # AutotuneAttrs default
        self.assertEqual(attrs.waves_per_eu, 1)
        self.assertEqual(attrs.kpack, 1)

    def test_from_cfg_kwargs_wins_over_attr(self) -> None:
        """When the same name appears in both kwargs and attr, kwargs takes priority."""
        cfg = triton.Config({"num_warps": 16}, num_warps=4)
        self.assertEqual(
            AutotuneAttrs.from_cfg(cfg, AutotuneAttrs.fields_for("cuda")).num_warps,
            16,
        )

    def test_from_cfg_cuda_ignores_amd_keys(self) -> None:
        """AMD-only keys present in cfg.kwargs are dropped for cuda backend."""
        cfg = triton.Config(
            {"BLOCK_M": 64, "matrix_instr_nonkdim": 16}, num_warps=4, num_stages=2
        )
        attrs = AutotuneAttrs.from_cfg(cfg, AutotuneAttrs.fields_for("cuda"))
        # AMD field stays at AutotuneAttrs default (NOT 16 from cfg).
        self.assertEqual(attrs.matrix_instr_nonkdim, 0)

    @parameterized.expand(
        [
            # name, cfg_kwargs, cfg_attr_kwargs, backend, expected
            (
                "from_kwargs",
                {"BLOCK_M": 64, "num_ctas": 2},
                {"num_warps": 4, "num_stages": 3},
                "cuda",
                2,
            ),
            # ``triton.Config(..., num_ctas=N)`` exposes num_ctas as a cfg
            # attr; from_cfg picks it up via the getattr fallback.
            (
                "from_attr",
                {"BLOCK_M": 64},
                {"num_warps": 4, "num_stages": 3, "num_ctas": 2},
                "cuda",
                2,
            ),
            (
                "defaults_to_one",
                {"BLOCK_M": 64},
                {"num_warps": 4, "num_stages": 3},
                "cuda",
                1,
            ),
            # num_ctas is NVIDIA-only — HIP fields_for() omits it.
            (
                "hip_drops_num_ctas",
                {"BLOCK_M": 64},
                {"num_warps": 4, "num_stages": 3, "num_ctas": 2},
                "hip",
                1,
            ),
        ]
    )
    def test_from_cfg_num_ctas(
        self,
        _name: str,
        cfg_kwargs: dict[str, int],
        cfg_attr_kwargs: dict[str, int],
        backend: str,
        expected_num_ctas: int,
    ) -> None:
        cfg = triton.Config(cfg_kwargs, **cfg_attr_kwargs)
        attrs = AutotuneAttrs.from_cfg(cfg, AutotuneAttrs.fields_for(backend))
        self.assertEqual(attrs.num_ctas, expected_num_ctas)


class AutotuneSpecsTest(unittest.TestCase):
    """Tests for ``_autotune_specs()``."""

    def _make_mock_autotuner(self) -> MagicMock:
        # ``spec=Autotuner`` makes ``isinstance(mock, KernelInterface)`` return
        # True, so ``kernel_param_names`` peels via ``.fn`` to the underlying
        # JITFunction instead of returning MagicMock's ``__call__`` signature.
        func = MagicMock(spec=triton.runtime.autotuner.Autotuner)
        func.fn = _addmm_fwd
        func.arg_names = list(_addmm_fwd.arg_names)
        func.cache = {
            (256, 1024): triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=2,
                num_stages=5,
            ),
            (128, 256): triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=4,
                num_stages=3,
            ),
        }
        func.key_idx = [5, 6]
        return func

    def test_produces_expected_specs(self) -> None:
        """_autotune_specs expands 1 base spec x 2 configs into 2 fully-specified specs."""
        # Arg indices: 15=BLOCK_M, 16=BLOCK_N, 17=BLOCK_K, 18=GROUP_M, 19=ALLOW_TF32
        base_spec = KernelSpec(
            signature={0: "*fp32", 4: "i32", 5: "i32", 6: "i32"},
            constants={19: 0},
            divisible_by_16={0},
            divisible_by_8=set(),
        )
        func = self._make_mock_autotuner()
        result = _autotune_specs(
            func,
            _compute_constexpr_keys(func),
            AutotuneAttrs.fields_for("cuda"),
            [base_spec],
        )

        expected = [
            KernelSpec(
                signature={0: "*fp32", 4: "i32", 5: "i32", 6: "i32"},
                constants={15: 64, 16: 32, 17: 32, 18: 8, 19: 0},
                divisible_by_16={0},
                divisible_by_8=set(),
                autotune=AutotuneAttrs(num_warps=2, num_stages=5),
            ),
            KernelSpec(
                signature={0: "*fp32", 4: "i32", 5: "i32", 6: "i32"},
                constants={15: 32, 16: 64, 17: 32, 18: 8, 19: 0},
                divisible_by_16={0},
                divisible_by_8=set(),
                autotune=AutotuneAttrs(num_warps=4, num_stages=3),
            ),
        ]
        self.assertEqual(result, expected)

    def test_from_raw_specs_with_tuned_func(self) -> None:
        """from_raw_specs with tuned_func produces expanded, deduped specs."""
        base_specs: list[RawKernelSpec] = [
            {
                "signature": [
                    ("*fp32", 16),  # x_ptr
                    ("*fp32", 16),  # w_ptr
                    ("*fp32", 16),  # y_ptr
                    ("*fp32", 16),  # z_ptr
                    ("i32", 16),  # M
                    ("i32", 16),  # N
                    ("i32", 16),  # K
                    ("i32", 16),  # stride_xm
                    1,  # stride_xk
                    ("i32", 16),  # stride_wk
                    1,  # stride_wn
                    ("i32", 16),  # stride_ym
                    1,  # stride_yn
                    ("i32", 16),  # stride_zm
                    1,  # stride_zn
                    64,  # BLOCK_M
                    32,  # BLOCK_N
                    32,  # BLOCK_K
                    8,  # GROUP_M
                    0,  # ALLOW_TF32
                    1,  # BROADCAST_Y
                ]
            }
        ]
        func = self._make_mock_autotuner()
        unit = OpsUnit.from_raw_specs(
            base_specs, GPUTarget("cuda", 90, 32), tuned_func=func
        )

        # 1 base spec × 2 configs = 2 tuned specs
        self.assertEqual(len(unit.specs), 2)
        self.assertEqual({s.autotune.num_warps for s in unit.specs}, {2, 4})
        self.assertEqual({s.autotune.num_stages for s in unit.specs}, {3, 5})

    @parameterized.expand(
        [
            # hip: AMD opts in cfg.kwargs flow into spec.autotune
            ("hip", "gfx942", 16, 2, 2),
            # cuda: AMD opts dropped (not in fields_for("cuda")) -> AutotuneAttrs defaults
            ("cuda", 90, 0, 1, 1),
        ]
    )
    def test_amd_backend_opt_routing(
        self,
        backend: str,
        arch: int | str,
        expected_matrix: int,
        expected_waves: int,
        expected_kpack: int,
    ) -> None:
        """BLOCK_M -> constants; matrix_instr_nonkdim -> autotune (hip) or default (cuda)."""
        func = MagicMock(spec=triton.runtime.autotuner.Autotuner)
        func.fn = _addmm_fwd
        func.arg_names = list(_addmm_fwd.arg_names)
        func.cache = {
            (1,): triton.Config(
                {
                    "BLOCK_M": 64,
                    "matrix_instr_nonkdim": 16,
                    "waves_per_eu": 2,
                    "kpack": 2,
                },
                num_warps=4,
                num_stages=2,
            ),
        }
        base = KernelSpec(
            signature={0: "*fp32"},
            constants={},
            divisible_by_16=set(),
            divisible_by_8=set(),
        )
        [tuned] = _autotune_specs(
            func,
            _compute_constexpr_keys(func),
            AutotuneAttrs.fields_for(backend),
            [base],
        )

        block_m_idx = func.arg_names.index("BLOCK_M")
        self.assertEqual(tuned.constants[block_m_idx], 64)
        self.assertEqual(tuned.autotune.matrix_instr_nonkdim, expected_matrix)
        self.assertEqual(tuned.autotune.waves_per_eu, expected_waves)
        self.assertEqual(tuned.autotune.kpack, expected_kpack)
        self.assertEqual(tuned.autotune.num_warps, 4)
        self.assertEqual(tuned.autotune.num_stages, 2)


class AutotuneSpecsCrossProductBugTest(unittest.TestCase):
    """Regression: ``_autotune_specs`` cross-products base_specs x autotuner.cache.

    Breaks for kernels with per-call non-autotune'd constexpr (e.g. ``N_BLOCK``
    in PPO ``_gemm_reduction_kernel``): synthesizes ``(N_BLOCK=1024,
    BLOCK_K=64)`` from never-co-bench'd (call_A_spec, call_B_cfg), exceeds H100
    SMEM cap, ``cuLaunchKernel`` returns ``CUDA_ERROR_INVALID_VALUE``.
    ``OpsUnit.drop_oversize_specs`` is the SMEM-side fix.
    """

    def _make_mock_autotuner(self) -> MagicMock:
        # N_BLOCK is *not* in cfg.kwargs -- launcher sets it per call site.
        func = MagicMock(spec=triton.runtime.autotuner.Autotuner)
        func.fn = _reduction_like_fwd
        func.arg_names = list(_reduction_like_fwd.arg_names)
        func.cache = {
            (64,): triton.Config(
                {"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=4, num_stages=3
            ),
            (1024,): triton.Config(
                {"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=8, num_stages=3
            ),
        }
        return func

    def test_cross_product_emits_unverified_n_block_cfg_pairing(self) -> None:
        # arg layout: 0:a 1:b 2:c 3:M 4:N 5:K 6:BLOCK_M 7:BLOCK_K 8:N_BLOCK
        BLOCK_K_IDX, N_BLOCK_IDX = 7, 8

        base_small_n = KernelSpec(
            signature={
                0: "*fp32",
                1: "*fp32",
                2: "*fp32",
                3: "i32",
                4: "i32",
                5: "i32",
            },
            constants={N_BLOCK_IDX: 64},
            divisible_by_16=set(),
            divisible_by_8=set(),
        )
        base_large_n = KernelSpec(
            signature={
                0: "*fp32",
                1: "*fp32",
                2: "*fp32",
                3: "i32",
                4: "i32",
                5: "i32",
            },
            constants={N_BLOCK_IDX: 1024},
            divisible_by_16=set(),
            divisible_by_8=set(),
        )

        func = self._make_mock_autotuner()
        result = _autotune_specs(
            func,
            _compute_constexpr_keys(func),
            AutotuneAttrs.fields_for("cuda"),
            [base_small_n, base_large_n],
        )

        # 2 base specs x 2 cached cfgs = 4 cross-product cells
        self.assertEqual(len(result), 4)

        bench_pairs: set[tuple[int, int]] = {(64, 64), (1024, 32)}
        emitted_pairs: set[tuple[int, int]] = {
            (s.constants[N_BLOCK_IDX], s.constants[BLOCK_K_IDX]) for s in result
        }
        synthesized = emitted_pairs - bench_pairs
        # (1024, 64) is the over-SMEM cell that crashes H100.
        self.assertIn((1024, 64), synthesized)
        self.assertIn((64, 32), synthesized)

        # Each cell carries the cfg's num_warps/num_stages.
        cfg_by_block_k = {64: (4, 3), 32: (8, 3)}
        for s in result:
            warps, stages = cfg_by_block_k[s.constants[BLOCK_K_IDX]]
            self.assertEqual(s.autotune.num_warps, warps)
            self.assertEqual(s.autotune.num_stages, stages)


class DetectOptionalArgsTest(unittest.TestCase):
    """Tests for _detect_optional_args()."""

    def test_single_spec_returns_empty(self) -> None:
        """Single spec has no cross-spec comparison to do."""
        spec = _make_kernel_spec(signature={0: "*fp32"}, constants={1: None})
        self.assertEqual(_detect_optional_args([spec]), set())

    def test_cross_spec_pointer_and_none(self) -> None:
        """Pointer in one spec + None constant in another → optional."""
        spec1 = _make_kernel_spec(signature={0: "*fp32", 1: "i32"})
        spec2 = _make_kernel_spec(signature={1: "i32"}, constants={0: None})
        self.assertEqual(_detect_optional_args([spec1, spec2]), {0})

    def test_non_optional_not_detected(self) -> None:
        """Same pointer in both specs → not optional."""
        spec1 = _make_kernel_spec(signature={0: "*fp32", 1: "i32"})
        spec2 = _make_kernel_spec(signature={0: "*bf16", 1: "i32"})
        self.assertEqual(_detect_optional_args([spec1, spec2]), set())

    def test_multiple_optional(self) -> None:
        """Multiple positions can be optional independently."""
        spec1 = _make_kernel_spec(signature={0: "*fp32", 1: "*i64"})
        spec2 = _make_kernel_spec(constants={0: None, 1: None})
        self.assertEqual(_detect_optional_args([spec1, spec2]), {0, 1})

    def test_from_raw_specs_detects_optional_without_unify(self) -> None:
        """from_raw_specs detects optional from bare None via _detect_optional_args."""
        base_specs: list[RawKernelSpec] = [
            {"signature": [("*fp32", 16), ("i32", 16)]},
            {"signature": [None, ("i32", 16)]},
        ]
        unit = OpsUnit.from_raw_specs(base_specs, GPUTarget("cuda", 90, 32))
        self.assertIn(0, unit.optional)
        self.assertIn(0, unit.pointer_args)


class WiderTypeTest(unittest.TestCase):
    """Tests for _wider_type()."""

    @parameterized.expand(
        [
            ("same_i32", "i32", "i32", "i32"),
            ("same_fp32", "fp32", "fp32", "fp32"),
            ("i32_widens_to_i64", "i32", "i64", "i64"),
            ("i64_stays_i64", "i64", "i32", "i64"),
        ]
    )
    def test_wider_type(self, _name: str, t1: str, t2: str, expected: str) -> None:
        self.assertEqual(_wider_type(t1, t2), expected)

    def test_incompatible_raises(self) -> None:
        with self.assertRaises(ValueError):
            _wider_type("fp32", "i32")

    def test_compute_invariants_picks_widest(self) -> None:
        """OpsUnit.scalar_dtypes uses widest type across specs."""
        base_specs: list[RawKernelSpec] = [
            {"signature": [("*fp32", 16), ("i32", 16)]},
            {"signature": [("*fp32", 16), "i64"]},
        ]
        unit = OpsUnit.from_raw_specs(base_specs, GPUTarget("cuda", 90, 32))
        self.assertEqual(unit.scalar_dtypes[1], "i64")


class ComputeAutotuneParamNamesTest(unittest.TestCase):
    """``compute_autotune_param_names`` flattens (constexpr_keys, autotune_fields)
    into the name list the transform pipeline needs (no ``OpsUnit`` available
    at that stage).
    """

    def test_no_autotuner_returns_only_backend_field_names(self) -> None:
        names = compute_autotune_param_names(None, "cuda")
        self.assertEqual(names, [f.name for f in AutotuneAttrs.fields_for("cuda")])

    def test_autotuner_prepends_constexpr_keys(self) -> None:
        func = MagicMock(spec=triton.runtime.autotuner.Autotuner)
        func.fn = _addmm_fwd
        func.arg_names = list(_addmm_fwd.arg_names)
        func.cache = {
            (1,): triton.Config(
                {"BLOCK_N": 32, "BLOCK_M": 64, "matrix_instr_nonkdim": 16},
                num_warps=4,
                num_stages=2,
            ),
        }
        names = compute_autotune_param_names(func, "hip")
        # Constexpr keys come first, then autotune fields (incl. AMD-only).
        self.assertEqual(
            names,
            ["BLOCK_N", "BLOCK_M"] + [f.name for f in AutotuneAttrs.fields_for("hip")],
        )

    def test_disagreeing_cfgs_raise(self) -> None:
        func = MagicMock(spec=triton.runtime.autotuner.Autotuner)
        func.fn = _addmm_fwd
        func.arg_names = list(_addmm_fwd.arg_names)
        func.cache = {
            (1,): triton.Config({"BLOCK_M": 64}),
            (2,): triton.Config({"BLOCK_N": 32}),
        }
        with self.assertRaisesRegex(ValueError, "disagree on constexpr keys"):
            _ = compute_autotune_param_names(func, "cuda")


class ComputeConstexprKeysTest(unittest.TestCase):
    """``_compute_constexpr_keys`` cfg-source fallback paths.

    ``compute_autotune_param_names`` and ``OpsUnit.from_raw_specs`` cover
    the warmed-cache happy path; these exercise the empty-cache branches.
    """

    def test_empty_cache_falls_back_to_configs(self) -> None:
        """Pre-warmup autotuner (cache empty) still yields keys from configs."""
        func = MagicMock(spec=triton.runtime.autotuner.Autotuner)
        func.fn = _addmm_fwd
        func.arg_names = list(_addmm_fwd.arg_names)
        func.cache = {}
        func.configs = [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        ]
        self.assertEqual(_compute_constexpr_keys(func), ("BLOCK_M", "BLOCK_N"))

    def test_empty_cache_and_configs_returns_empty(self) -> None:
        func = MagicMock(spec=triton.runtime.autotuner.Autotuner)
        func.fn = _addmm_fwd
        func.arg_names = list(_addmm_fwd.arg_names)
        func.cache = {}
        func.configs = []
        self.assertEqual(_compute_constexpr_keys(func), ())


class OpsUnitConstexprKeysTest(unittest.TestCase):
    """``OpsUnit.from_raw_specs`` populates ``constexpr_keys`` +
    ``autotune_fields`` once — the 4-face contract anchor for codegen.
    """

    _RAW_SPECS: list[RawKernelSpec] = [
        {
            "signature": [
                ("*fp32", 16),
                ("*fp32", 16),
                ("*fp32", 16),
                ("*fp32", 16),
                ("i32", 16),
                ("i32", 16),
                ("i32", 16),
                ("i32", 16),
                1,
                ("i32", 16),
                1,
                ("i32", 16),
                1,
                ("i32", 16),
                1,
                64,
                32,
                32,
                8,
                0,
                1,
            ]
        }
    ]

    def test_no_autotuner_empty_constexpr_keys_cuda_fields(self) -> None:
        unit = OpsUnit.from_raw_specs(self._RAW_SPECS, GPUTarget("cuda", 90, 32))
        self.assertEqual(unit.constexpr_keys, ())
        self.assertEqual(unit.autotune_fields, AutotuneAttrs.fields_for("cuda"))

    def test_autotuner_populates_constexpr_keys(self) -> None:
        func = MagicMock(spec=triton.runtime.autotuner.Autotuner)
        func.fn = _addmm_fwd
        func.arg_names = list(_addmm_fwd.arg_names)
        func.cache = {
            (1,): triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=4,
                num_stages=2,
            ),
        }
        unit = OpsUnit.from_raw_specs(
            self._RAW_SPECS, GPUTarget("cuda", 90, 32), tuned_func=func
        )
        self.assertEqual(
            unit.constexpr_keys, ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M")
        )
        self.assertEqual(unit.autotune_fields, AutotuneAttrs.fields_for("cuda"))

    def test_autotune_fields_track_backend(self) -> None:
        unit = OpsUnit.from_raw_specs(self._RAW_SPECS, GPUTarget("hip", "gfx942", 64))
        self.assertEqual(unit.autotune_fields, AutotuneAttrs.fields_for("hip"))

    def test_autotuner_populates_schema_and_expands_specs(self) -> None:
        """End-to-end: ``from_raw_specs(..., tuned_func=)`` produces a unit where
        schema (``constexpr_keys`` + ``autotune_fields``) and per-variant data
        (``specs`` with ``KernelSpec.autotune`` filled) are *jointly* correct.

        Downstream codegen reads from all three; a regression that filled
        only the schema or only the specs would silently break codegen.
        """
        func = MagicMock(spec=triton.runtime.autotuner.Autotuner)
        func.fn = _addmm_fwd
        func.arg_names = list(_addmm_fwd.arg_names)
        func.cache = {
            (1,): triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=2,
                num_stages=5,
            ),
            (2,): triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=4,
                num_stages=3,
            ),
        }
        unit = OpsUnit.from_raw_specs(
            self._RAW_SPECS, GPUTarget("cuda", 90, 32), tuned_func=func
        )

        # Schema
        self.assertEqual(
            unit.constexpr_keys, ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M")
        )
        self.assertEqual(unit.autotune_fields, AutotuneAttrs.fields_for("cuda"))

        # Per-variant specs expanded one-per-cfg, autotune values flow in.
        self.assertEqual(len(unit.specs), 2)
        self.assertEqual({s.autotune.num_warps for s in unit.specs}, {2, 4})
        self.assertEqual({s.autotune.num_stages for s in unit.specs}, {3, 5})


class MaxDynamicSharedPerBlockOptinTest(unittest.TestCase):
    """Host-driver SMEM cap query (assumes arch already validated upstream)."""

    @parameterized.expand(
        [
            # name, gpu_target, cuda_available, expected
            ("matching_arch", GPUTarget("cuda", 90, 32), True, 232448),
            ("non_cuda_backend", GPUTarget("hip", "gfx942", 64), True, sys.maxsize),
            ("no_cuda_host", GPUTarget("cuda", 90, 32), False, sys.maxsize),
        ]
    )
    def test_returns_expected_value(
        self,
        _name: str,
        gpu_target: GPUTarget,
        cuda_available: bool,
        expected: int,
    ) -> None:
        with (
            patch(
                "aot_tensor.compile.triton.spec_processing.torch.cuda.is_available",
                return_value=cuda_available,
            ),
            patch(
                "aot_tensor.compile.triton.spec_processing.max_shared_mem",
                return_value=232448,
            ),
        ):
            self.assertEqual(
                _max_dynamic_shared_per_block_optin(gpu_target),
                expected,
            )


class OpsUnitDropOversizeSpecsTest(unittest.TestCase):
    """``OpsUnit.drop_oversize_specs`` policy + warning."""

    def _make_unit_with_n_specs(self, n: int, smem_cap: int) -> OpsUnit:
        specs = [
            KernelSpec(
                signature={0: "*fp32"},
                constants={1: 16, 2: 64 * (i + 1)},
                divisible_by_16=set(),
                divisible_by_8=set(),
                autotune=AutotuneAttrs(num_warps=4, num_stages=3),
            )
            for i in range(n)
        ]
        return OpsUnit(
            cc=90,
            optional=set(),
            pointer_args={0},
            scalar_dtypes={},
            constant_types={1: int, 2: int},
            specs=specs,
            smem_cap=smem_cap,
        )

    def test_partitions_specs_by_cap_and_warns_on_drop(self) -> None:
        unit = self._make_unit_with_n_specs(3, smem_cap=232_448)
        generated_specs = [
            ("code_a", 100_000),
            ("code_b", 232_448),  # boundary == cap, kept
            ("code_c", 266_240),  # over cap, dropped
        ]
        with self.assertLogs(
            "aot_tensor.compile.triton.spec_processing", level=logging.WARNING
        ) as captured:
            narrowed, filtered_specs = unit.drop_oversize_specs(generated_specs)
        self.assertEqual(len(narrowed.specs), 2)
        self.assertEqual(filtered_specs, ["code_a", "code_b"])
        joined = "\n".join(captured.output)
        self.assertIn("266240", joined)
        self.assertIn("232448", joined)

    def test_no_drops_returns_full_unit(self) -> None:
        unit = self._make_unit_with_n_specs(2, smem_cap=232_448)
        narrowed, filtered_specs = unit.drop_oversize_specs([("a", 1024), ("b", 2048)])
        self.assertEqual(narrowed.specs, unit.specs)
        self.assertEqual(filtered_specs, ["a", "b"])

    def test_all_dropped_raises(self) -> None:
        unit = self._make_unit_with_n_specs(2, smem_cap=232_448)
        with self.assertRaisesRegex(RuntimeError, "All 2 specs exceeded SMEM cap"):
            unit.drop_oversize_specs([("a", 300_000), ("b", 400_000)])

    def test_preserves_invariants_across_filter(self) -> None:
        unit = self._make_unit_with_n_specs(3, smem_cap=232_448)
        narrowed, _ = unit.drop_oversize_specs([("a", 100), ("b", 300_000), ("c", 200)])
        self.assertEqual(narrowed.pointer_args, unit.pointer_args)
        self.assertEqual(narrowed.scalar_dtypes, unit.scalar_dtypes)
        self.assertEqual(narrowed.constant_types, unit.constant_types)
        self.assertEqual(narrowed.cc, unit.cc)
        self.assertEqual(narrowed.smem_cap, unit.smem_cap)

    def test_default_smem_cap_is_maxsize(self) -> None:
        unit = OpsUnit(
            cc=90,
            optional=set(),
            pointer_args={0},
            scalar_dtypes={},
            constant_types={},
            specs=[
                KernelSpec(
                    signature={0: "*fp32"},
                    constants={},
                    divisible_by_16=set(),
                    divisible_by_8=set(),
                    autotune=AutotuneAttrs(),
                )
            ],
        )
        self.assertEqual(unit.smem_cap, sys.maxsize)
        narrowed, codes = unit.drop_oversize_specs([("c", 999_999_999)])
        self.assertEqual(len(narrowed.specs), 1)
        self.assertEqual(codes, ["c"])
