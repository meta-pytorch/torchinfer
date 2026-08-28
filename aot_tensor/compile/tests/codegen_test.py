# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors[2]: triton func without type

import dataclasses
import re
import types
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# @manual=//triton:triton
import triton
import triton.language as tl
from aot_tensor.compile.tests.mocks import MockAutotuner
from aot_tensor.compile.triton.arg_descriptor import (
    ArgDescriptor,
    build_arg_descriptors,
    ConstantArg,
    PointerArg,
    ScalarArg,
)
from aot_tensor.compile.triton.codegen import (
    _as_cpp_string_literals,
    _CONT,
    _INDENT,
    gen_cpp_op_params,
    gen_failure_msg,
    gen_guarded_calls,
    gen_kernel_name,
    gen_launcher,
    gen_launcher_call_args,
    gen_launcher_params,
    gen_selector_params,
    gen_selector_proto,
    gen_torch_op_params,
    gen_torch_op_schema,
    gen_tuner_meta_cpp,
    gen_tuner_meta_py,
    generate_header_content,
    generate_kernel_cpp_content,
    generate_torch_op_content,
    is_non_empty_mapping_of_type,
    validate_unique_kernel_names,
)
from aot_tensor.compile.triton.spec_processing import AutotuneAttrs, KernelSpec, OpsUnit
from parameterized import parameterized
from triton.runtime import JITFunction

_CPP_STRING_LITERAL = r'"((?:[^"\\]|\\.)*)"'

# Between today's peak (140) and the shortest regression this catches (231).
# The floor is the ~130-char kernel symbol, not anything reformattable.
_MAX_GENERATED_LINE = 200


def _schema_from_registration(content: str) -> str:
    """Reassemble the schema from the ``m.def(...)`` registration.

    Codegen splits it across adjacent C++ string literals, which the compiler
    concatenates back into one string.
    """
    match = re.search(rf"m\.def\(\s*((?:{_CPP_STRING_LITERAL}\s*)+)\)", content)
    assert match is not None, f"no m.def(...) registration found in:\n{content}"
    return "".join(re.findall(_CPP_STRING_LITERAL, match.group(1)))


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


class CompilerTest(unittest.TestCase):
    def _create_mock_func(self) -> JITFunction:
        return _addmm_fwd

    def _create_mock_tuned_func(self) -> Any:
        """Build a fake autotuner using ``MockAutotuner`` so ``unwrap_to_jit``
        (which uses ``isinstance(Autotuner)``) and ``is_autotuner`` (MRO walk)
        both succeed.  ``key_idx`` is attached separately as it is
        codegen-test-specific, not part of the ``MockAutotuner`` contract.
        """
        tuned_func = MockAutotuner(
            fn=_addmm_fwd,
            arg_names=list(_addmm_fwd.arg_names),
            cache={
                (256, 1024): triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 32,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_warps=2,
                    num_stages=5,
                ),
                (128, 256): triton.Config(
                    {
                        "BLOCK_M": 32,
                        "BLOCK_N": 64,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_warps=2,
                    num_stages=5,
                ),
            },
        )
        tuned_func.key_idx = [5, 6]  # indices of N and K in arg_names
        return tuned_func

    def _create_mock_unit(self) -> OpsUnit:
        """Create mock OpsUnit for _addmm_fwd (cuda-flavored)."""
        return OpsUnit(
            cc=80,
            optional=set(),
            pointer_args={0, 1, 2, 3},
            scalar_dtypes={
                4: "i32",
                5: "i32",
                6: "i32",
                7: "i32",
                9: "i32",
                11: "i32",
                13: "i32",
            },
            constant_types={
                8: int,
                10: int,
                12: int,
                14: int,
                15: int,
                16: int,
                17: int,
                18: int,
                19: int,
                20: int,
            },
            specs=[
                KernelSpec(
                    signature={
                        0: "*fp32",  # x_ptr
                        1: "*fp32",  # w_ptr
                        2: "*fp32",  # y_ptr
                        3: "*fp32",  # z_ptr
                        4: "i32",  # M
                        5: "i32",  # N
                        6: "i32",  # K
                        7: "i32",  # stride_xm
                        9: "i32",  # stride_wk
                        11: "i32",  # stride_ym
                        13: "i32",  # stride_zm
                    },
                    constants={
                        8: 1,  # stride_xk
                        10: 1,  # stride_wn
                        12: 1,  # stride_yn
                        14: 1,  # stride_zn
                        15: 64,  # BLOCK_M
                        16: 32,  # BLOCK_N
                        17: 32,  # BLOCK_K
                        18: 8,  # GROUP_M
                        19: 0,  # ALLOW_TF32
                        20: 1,  # BROADCAST_Y
                    },
                    divisible_by_16={0, 1, 2, 3, 5, 6, 7, 9, 11, 13},
                    divisible_by_8={5, 6, 7, 9, 11, 13},
                    autotune=AutotuneAttrs(
                        num_warps=2,
                        num_stages=5,
                        matrix_instr_nonkdim=0,
                        waves_per_eu=0,
                        kpack=1,
                    ),
                )
            ],
            constexpr_keys=("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"),
            autotune_fields=AutotuneAttrs.fields_for("cuda"),
        )

    def _create_descriptors(self) -> list[ArgDescriptor]:
        """Build descriptors from mock func + unit."""
        return build_arg_descriptors(self._create_mock_func(), self._create_mock_unit())

    def test_gen_selector_proto(self) -> None:
        """Test that gen_selector_proto generates expected function signature."""
        descriptors = self._create_descriptors()

        result = gen_selector_proto(
            descriptors, "_addmm_fwd", AutotuneAttrs.fields_for("cuda")
        )

        expected_strs = [
            "void _addmm_fwd(",
            "gridDims grid",
            "const std::optional<torch::stable::Tensor>& x_ptr",
            "int32_t M",
            "num_warps=4",
            "num_stages=3",
        ]
        for s in expected_strs:
            self.assertIn(s, result)

    def test_gen_tuner_meta_cpp(self) -> None:
        """Test that gen_tuner_meta_cpp generates expected C++ tuner function."""
        func = self._create_mock_tuned_func()
        unit = self._create_mock_unit()

        result = gen_tuner_meta_cpp(
            func,
            tuner_fallback=True,
            constant_types=unit.constant_types,
        )

        expected_strs = [
            "inline std::tuple<",
            "_addmm_fwd_meta(",
            "int64_t N",
            "int64_t K",
            "N == 256 && K == 1024",
            "N == 128 && K == 256",
            "std::make_tuple",
        ]
        for s in expected_strs:
            self.assertIn(s, result)

    @parameterized.expand(
        [
            (
                "with_fallback",
                True,
                "return ",
            ),
            (
                "no_fallback_raises",
                False,
                "raise RuntimeError",
            ),
        ]
    )
    def test_gen_tuner_meta_py_autotuned(
        self,
        _name: str,
        tuner_fallback: bool,
        expected_str: str,
    ) -> None:
        """Test that gen_tuner_meta_py generates correct Python meta for autotuned kernel."""
        func = self._create_mock_tuned_func()
        unit = self._create_mock_unit()

        result = gen_tuner_meta_py(
            func,
            tuner_fallback=tuner_fallback,
            unit=unit,
        )

        common_strs = [
            "def _addmm_fwd_meta(",
            "N: int",
            "K: int",
            "# Returns: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages, num_ctas, auto_tma)",
            "if N == 256 and K == 1024:",
            "if N == 128 and K == 256:",
        ]
        for s in common_strs:
            self.assertIn(s, result)
        self.assertIn(expected_str, result)

    @parameterized.expand(
        [
            # (name, fallback, expected_str, backend, returns_line)
            (
                "cuda_with_fallback",
                True,
                "return ",
                "cuda",
                "# Returns: (num_warps, num_stages, num_ctas, auto_tma)",
            ),
            (
                "cuda_no_fallback",
                False,
                "raise RuntimeError",
                "cuda",
                "# Returns: (num_warps, num_stages, num_ctas, auto_tma)",
            ),
            (
                "hip_with_fallback",
                True,
                "return ",
                "hip",
                "# Returns: (num_warps, num_stages, matrix_instr_nonkdim, waves_per_eu, kpack)",
            ),
        ]
    )
    def test_gen_tuner_meta_py_default(
        self,
        _name: str,
        tuner_fallback: bool,
        expected_str: str,
        backend: str,
        returns_line: str,
    ) -> None:
        """gen_tuner_meta_py for non-autotuned kernel: Returns line reflects
        backend-correct AutotuneAttrs field set."""
        unit = dataclasses.replace(
            self._create_mock_unit(),
            constexpr_keys=(),
            autotune_fields=AutotuneAttrs.fields_for(backend),
        )
        result = gen_tuner_meta_py(
            self._create_mock_func(),
            tuner_fallback=tuner_fallback,
            unit=unit,
        )
        self.assertIn("def _addmm_fwd_meta(", result)
        self.assertIn(returns_line, result)
        self.assertIn(expected_str, result)

    def _create_mock_tuned_func_amd(self) -> Any:
        """AMD-flavor autotuner: ``cfg.kwargs`` mixes kernel constexprs
        (``BLOCK_*``) with AMD backend opts (``matrix_instr_nonkdim``,
        ``waves_per_eu``) — exercises the ``cfg_constexpr_keys`` split."""
        tuned_func = MockAutotuner(
            fn=_addmm_fwd,
            arg_names=list(_addmm_fwd.arg_names),
            cache={
                (256, 1024): triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 32,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                        "matrix_instr_nonkdim": 16,
                        "waves_per_eu": 2,
                    },
                    num_warps=2,
                    num_stages=5,
                ),
            },
        )
        tuned_func.key_idx = [5, 6]
        return tuned_func

    def test_gen_tuner_meta_py_autotuned_amd_splits_kwargs(self) -> None:
        """AMD: ``cfg.kwargs`` containing both kernel constexprs and AMD
        backend opts must split — constexprs in the return prefix, backend
        opts in the AutotuneAttrs tail."""
        tuned = self._create_mock_tuned_func_amd()
        unit = dataclasses.replace(
            self._create_mock_unit(),
            autotune_fields=AutotuneAttrs.fields_for("hip"),
        )
        result = gen_tuner_meta_py(
            tuned,
            tuner_fallback=True,
            unit=unit,
        )
        self.assertIn(
            "# Returns: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, "
            "num_warps, num_stages, matrix_instr_nonkdim, waves_per_eu, kpack)",
            result,
        )

    def test_generate_header_content(self) -> None:
        """Test that generate_header_content produces expected structure and content."""
        func = self._create_mock_func()
        tuned_func = self._create_mock_tuned_func()
        unit = self._create_mock_unit()
        descriptors = self._create_descriptors()

        result = generate_header_content(
            tuned_func=tuned_func,
            func=func,
            unit=unit,
            descriptors=descriptors,
            tuner_fallback=True,
            autotune_fields=AutotuneAttrs.fields_for("cuda"),
        )

        expected_strs = [
            "#pragma once",
            "#include <torch/csrc/stable/tensor.h>",
            "#include <cuda.h>",
            "#include <cuda_runtime.h>",
            "namespace triton {",
            "namespace aot {",
            "} // namespace aot",
            "} // namespace triton",
            "struct gridDims {",
            "GRID_DIM_DEFINED_MACRO",
            "int x = 1;",
            "cudaStream_t stream = 0;",
            "_addmm_fwd_meta(int64_t N, int64_t K)",
            "N == 256 && K == 1024",
            "N == 128 && K == 256",
            # The proto is emitted one parameter per line, so the signature is
            # no longer a single contiguous string.
            "void _addmm_fwd(",
            "gridDims grid",
            "const std::optional<torch::stable::Tensor>& x_ptr",
            "int num_warps=4",
        ]
        for s in expected_strs:
            self.assertIn(s, result)

    def test_generate_header_content_without_tuned_func(self) -> None:
        """Test generate_header_content without autotuning."""
        func = self._create_mock_func()
        unit = self._create_mock_unit()
        descriptors = self._create_descriptors()

        result = generate_header_content(
            tuned_func=None,
            func=func,
            unit=unit,
            descriptors=descriptors,
            tuner_fallback=False,
            autotune_fields=AutotuneAttrs.fields_for("cuda"),
        )

        self.assertNotIn("_addmm_fwd_meta", result)
        self.assertIn("void _addmm_fwd(", result)

    @parameterized.expand(
        [
            ("default_prefix", "_addmm_fwd"),
            ("custom_prefix", "my_custom_kernel"),
        ]
    )
    def test_generate_kernel_cpp_content(self, _name: str, prefix: str) -> None:
        """Test that generate_kernel_cpp_content produces expected structure."""
        func = self._create_mock_func()
        unit = self._create_mock_unit()
        descriptors = self._create_descriptors()
        generated_specs = ["// mock generated specs from spec_gen"]

        result = generate_kernel_cpp_content(
            func,
            unit,
            descriptors,
            prefix,
            generated_specs,
            AutotuneAttrs.fields_for("cuda"),
            backend="cuda",
        )

        expected_strs = [
            f'#include "{prefix}.h"',
            "namespace triton {",
            "namespace aot {",
            "int compute_capability()",
            "void enable_large_smem_or_throw(int shared, CUfunction func)",
            generated_specs[0],
            f"void {func.__name__}(",
        ]
        for s in expected_strs:
            self.assertIn(s, result)

        # D4: no ATen headers remain
        self.assertNotIn("<ATen/Tensor.h>", result)
        self.assertNotIn("<torch/library.h>", result)
        self.assertIn("torch/headeronly/core/ScalarType.h", result)

        # D1: error handling uses TRITON_AOT_CU_CHECK / std::runtime_error
        self.assertIn("TRITON_AOT_CU_CHECK", result)
        self.assertIn("std::runtime_error", result)
        self.assertNotIn("AT_CUDA_DRIVER_CHECK", result)
        self.assertNotIn("c10::Error", result)

        # D2: compute_capability uses CUDA Driver API, not ATen
        self.assertIn("cuDeviceGetAttribute", result)
        self.assertNotIn("at::cuda::getCurrentDeviceProperties", result)

        # D3: device/stream uses stable ABI, not c10::cuda
        self.assertIn("triton_aot_get_current_stream", result)
        self.assertNotIn("c10::cuda::getCurrentCUDAStream", result)
        self.assertNotIn("c10::cuda::current_device", result)

        # D5: enable_large_smem_or_throw uses cuCtxGetDevice (NOT hardcoded
        # device 0) and hard-rejects over-SMEM specs at load time.
        self.assertIn("cuCtxGetDevice", result)
        self.assertNotIn("int device = 0", result)
        self.assertIn("shared > shared_optin", result)
        self.assertIn("max opt-in", result)

        # D6: deprecated check_errors name is gone (renamed to
        # enable_large_smem_or_throw so the function name reflects what
        # it actually does -- opt into >48KB SMEM + validate).
        self.assertNotIn("check_errors", result)

    @parameterized.expand(
        [
            # hip: kernel_specs + selector → 2 hipify calls
            ("hip_hipifies", "hip", 2),
            # cuda: no hipify
            ("cuda_skips", "cuda", 0),
        ]
    )
    def test_generate_kernel_cpp_content_hipify_routing(
        self,
        _name: str,
        backend: str,
        expected_hipify_calls: int,
    ) -> None:
        """``backend`` routes generated code through ``maybe_hipify_code_wrapper``
        iff ``backend == "hip"`` (two callees: KERNEL_SPECS + SELECTOR)."""
        with patch(
            "torch._inductor.codegen.aoti_hipify_utils.maybe_hipify_code_wrapper",
            side_effect=lambda s, **_: f"<HIPIFIED>{s}",
        ) as mock_hipify:
            result = generate_kernel_cpp_content(
                self._create_mock_func(),
                self._create_mock_unit(),
                self._create_descriptors(),
                "_addmm_fwd",
                ["// mock generated specs"],
                AutotuneAttrs.fields_for(backend),
                backend=backend,
            )
        self.assertEqual(mock_hipify.call_count, expected_hipify_calls)
        if expected_hipify_calls:
            self.assertIn("<HIPIFIED>", result)
        else:
            self.assertNotIn("<HIPIFIED>", result)

    @parameterized.expand(
        [
            ("default_prefix", "_addmm_fwd"),
            ("custom_prefix", "my_custom_kernel"),
        ]
    )
    def test_generate_torch_op_content(self, _name: str, prefix: str) -> None:
        """Test that generate_torch_op_content produces expected structure."""
        func = self._create_mock_func()
        descriptors = self._create_descriptors()

        result = generate_torch_op_content(
            func, descriptors, prefix, {}, AutotuneAttrs.fields_for("cuda")
        )

        expected_strs = [
            f'#include "{prefix}.h"',
            f"void {func.__name__}_op(",
            f"void {func.__name__}_dummy_op(",
            "STABLE_TORCH_LIBRARY_FRAGMENT(triton_aot, m)",
            "m.def(",
            "STABLE_TORCH_LIBRARY_IMPL(triton_aot, CUDA, m)",
            "STABLE_TORCH_LIBRARY_IMPL(triton_aot, CPU, m)",
            "STABLE_TORCH_LIBRARY_IMPL(triton_aot, Meta, m)",
            "TORCH_BOX",
            "non-optional but use Tensor?",
            # Both pin a position, not just a presence. Which brace carries
            # the namespace comment is the point, and the second string is the
            # only thing covering the blank line before the CUDA block -- the
            # first one spans a separator that already existed.
            "}\n} // namespace\n\nSTABLE_TORCH_LIBRARY_FRAGMENT",
            "}\n\nSTABLE_TORCH_LIBRARY_IMPL(triton_aot, CUDA, m)",
        ]
        for s in expected_strs:
            self.assertIn(s, result)

        self.assertTrue(
            _schema_from_registration(result).startswith(
                f"{func.__name__}(int[] grid, "
            )
        )

        # D4: no unstable ATen references
        self.assertNotIn("<ATen/Tensor.h>", result)
        self.assertNotIn(
            "TORCH_LIBRARY_FRAGMENT",
            result.replace("STABLE_TORCH_LIBRARY_FRAGMENT", ""),
        )
        self.assertNotIn(
            "TORCH_LIBRARY_IMPL", result.replace("STABLE_TORCH_LIBRARY_IMPL", "")
        )

    @parameterized.expand(
        [
            ("int_default", {"M": 128}, "int M = 128"),
            ("string_default", {"M": "test_string"}, '\\"test_string\\"'),
        ]
    )
    def test_generate_torch_op_content_with_default_values(
        self, name: str, default_values: Dict[str, Any], expected_str: str
    ) -> None:
        """Test generate_torch_op_content with default values (tests gen_str_wrap)."""
        func = self._create_mock_func()
        descriptors = self._create_descriptors()
        prefix = "_addmm_fwd"

        result = generate_torch_op_content(
            func, descriptors, prefix, default_values, AutotuneAttrs.fields_for("cuda")
        )

        self.assertIn(expected_str, result)

    def test_generated_cpp_has_no_overlong_lines(self) -> None:
        """Catches an emitter going back to ``", ".join(...)``, which is how the
        selector became a single 4,000-char line."""
        func = self._create_mock_func()
        unit = self._create_mock_unit()
        descriptors = self._create_descriptors()
        autotune_fields = AutotuneAttrs.fields_for("cuda")

        contents = {
            "header": generate_header_content(
                tuned_func=self._create_mock_tuned_func(),
                func=func,
                unit=unit,
                descriptors=descriptors,
                tuner_fallback=True,
                autotune_fields=autotune_fields,
            ),
            "kernel.cpp": generate_kernel_cpp_content(
                func,
                unit,
                descriptors,
                "_addmm_fwd",
                ["// mock generated specs from spec_gen"],
                autotune_fields,
                backend="cuda",
            ),
            "torch_op.cpp": generate_torch_op_content(
                func, descriptors, "_addmm_fwd", {}, autotune_fields
            ),
        }

        longest = {
            name: max(len(line) for line in content.splitlines())
            for name, content in contents.items()
        }
        self.assertLessEqual(
            max(longest.values()),
            _MAX_GENERATED_LINE,
            f"generated C++ has an overlong line ({longest}); something was "
            "re-joined onto one line",
        )

    def test_gen_torch_op_schema_matches_cpp_registration(self) -> None:
        func = self._create_mock_func()
        descriptors = self._create_descriptors()

        schema = gen_torch_op_schema(
            func, descriptors, {"BLOCK_M": 128}, AutotuneAttrs.fields_for("cuda")
        )
        content = generate_torch_op_content(
            func,
            descriptors,
            "_addmm_fwd",
            {"BLOCK_M": 128},
            AutotuneAttrs.fields_for("cuda"),
        )

        # The registration splits the schema across adjacent string literals,
        # which the compiler concatenates. Reassemble rather than substring
        # match, so this still proves the two are byte-identical.
        self.assertEqual(_schema_from_registration(content), schema)
        self.assertEqual(schema.split("(", 1)[0], func.__name__)
        self.assertIn("int[] grid", schema)
        self.assertIn("Tensor(a!)? x_ptr", schema)
        self.assertIn("int BLOCK_M = 128", schema)
        self.assertTrue(schema.endswith(") -> ()"))

    def test_gen_torch_op_schema_includes_python_defaults(self) -> None:
        func = self._create_mock_func()
        descriptors = self._create_descriptors()
        defaults = {"allow_tf32": False, "BLOCK_M": 128}

        schema = gen_torch_op_schema(
            func, descriptors, defaults, AutotuneAttrs.fields_for("cuda")
        )

        self.assertIn("int BLOCK_M = 128", schema)


class GenKernelNameTest(unittest.TestCase):
    """Tests for gen_kernel_name()."""

    @parameterized.expand(
        [
            (
                "nvidia_sm90",
                90,
                ["_addmm_fwd", "sm90", "pfp32", "i32", "64", "w4", "s3"],
            ),
            (
                "amd_gfx942",
                "gfx942",
                ["_addmm_fwd", "smgfx942", "pfp32"],
            ),
        ]
    )
    def test_kernel_name_contains_expected_parts(
        self,
        _name: str,
        cc: int | str,
        expected_substrings: List[str],
    ) -> None:
        """Kernel name includes func name, cc, sig, constants, autotune attrs."""
        spec = KernelSpec(
            signature={0: "*fp32", 1: "i32"},
            constants={2: 64},
            divisible_by_16={0},
            divisible_by_8=set(),
            autotune=AutotuneAttrs(num_warps=4, num_stages=3),
        )
        name = gen_kernel_name(_addmm_fwd, spec, cc, AutotuneAttrs.fields_for("cuda"))
        for s in expected_substrings:
            self.assertIn(s, name)

    def test_suffix_marks_use_original_arg_indices_with_folded_args(self) -> None:
        """The divisibility suffix must track ORIGINAL arg indices (the
        space the constraint sets live in), not dict position. Real-world
        repro shape (gemm_pointwise, P2467214344): equal-to-1 strides fold
        into constants, so positional iteration shifted every later mark
        onto the wrong arg and dropped marks past the fold boundary --
        symbol names misdescribed their cubins, and two variant specs
        differing only in those marks could collide on cubin filename."""
        spec = KernelSpec(
            # Args 1 and 2 are folded equal-to-1 strides; marks sit on the
            # pointer (0) and on scalars AFTER the fold gap (4, 5).
            signature={0: "*fp16", 3: "i32", 4: "i32", 5: "i32"},
            constants={1: 1, 2: 1},
            divisible_by_16={0, 4, 5},
            divisible_by_8=set(),
            autotune=AutotuneAttrs(),
        )
        name = gen_kernel_name(_addmm_fwd, spec, 90, AutotuneAttrs.fields_for("cuda"))
        suffix = name.rsplit("_", 1)[-1]
        # Correct: marks on original indices 0, 4, 5. The positional bug
        # produced "0d123" (mark shifted to position 0 only, 4/5 dropped).
        self.assertEqual(suffix, "0d34d5d")

    def test_validate_unique_kernel_names_rejects_collision_pre_compile(
        self,
    ) -> None:
        """The pre-compile validator (runs before compile_specs_parallel,
        i.e. before any Triton compile cost is paid) rejects two specs
        mapping to one name; distinct specs pass."""
        spec = KernelSpec(
            signature={0: "*fp16", 3: "i32"},
            constants={1: 1, 2: 1},
            divisible_by_16={0},
            divisible_by_8=set(),
            autotune=AutotuneAttrs(),
        )
        fields = AutotuneAttrs.fields_for("cuda")
        distinct = dataclasses.replace(spec, divisible_by_16={0, 3})
        # Distinct specs -> distinct names -> no raise.
        validate_unique_kernel_names(_addmm_fwd, [spec, distinct], 90, fields)
        with self.assertRaisesRegex(RuntimeError, "kernel name collision"):
            validate_unique_kernel_names(_addmm_fwd, [spec, spec], 90, fields)

    def test_kernel_name_segments_per_backend(self) -> None:
        """Cubin name segments are platform-correct and complete:
        NVIDIA emits ``w/s/cta`` (common + NVIDIA-only); AMD emits
        ``w/s/matrix/wave/kpack`` (common + AMD-only). Dropping a segment
        from either set would silently collide cubins across autotune
        configs that only differ in that field (e.g. two ``num_ctas``
        variants would map to the same .cubin filename)."""
        spec = KernelSpec(
            signature={0: "*fp32", 1: "i32"},
            constants={2: 64},
            divisible_by_16={0},
            divisible_by_8=set(),
            autotune=AutotuneAttrs(
                num_warps=8,
                num_stages=4,
                matrix_instr_nonkdim=16,
                waves_per_eu=2,
                kpack=2,
                num_ctas=2,
            ),
        )
        cases = {
            "cuda": ["w8", "s4", "cta2"],
            "hip": ["w8", "s4", "matrix16", "wave2", "kpack2"],
        }
        for backend, expected in cases.items():
            with self.subTest(backend=backend):
                name = gen_kernel_name(
                    _addmm_fwd, spec, 90, AutotuneAttrs.fields_for(backend)
                )
                for segment in expected:
                    self.assertIn(segment, name)
                # Negative checks: backend-only segments must NOT leak across.
                if backend == "cuda":
                    for amd_only in ("matrix", "wave", "kpack"):
                        self.assertNotIn(amd_only, name)
                else:
                    self.assertNotIn("cta", name)


class AsCppStringLiteralsTest(unittest.TestCase):
    """Splitting a schema across adjacent literals must not change its value.

    String defaults (``gen_str_wrap`` emits ``\\"value\\"``) are the risky
    shape; no schema in tree has one yet, so cover it here.
    """

    @parameterized.expand(
        [
            ("no_string_default", "_k(int[] grid, int M, int num_warps=4) -> ()"),
            ("string_default", '_k(int[] grid, str mode=\\"foo\\", int y=1) -> ()'),
            (
                "comma_space_inside",
                '_k(int[] grid, str mode=\\"a, b\\", int y=1) -> ()',
            ),
            ("two_string_defaults", '_k(str a=\\"x\\", str b=\\"y\\", int c=3) -> ()'),
        ]
    )
    def test_round_trips(self, _name: str, schema: str) -> None:
        rendered = _as_cpp_string_literals(schema, indent=4, width=24)
        self.assertGreater(rendered.count("\n"), 0, "expected a multi-line split")
        # Concatenating the literal bodies must reproduce the input exactly, and
        # no chunk may end in a backslash that would escape its closing quote.
        bodies = re.findall(_CPP_STRING_LITERAL, rendered)
        self.assertEqual("".join(bodies), schema)
        for body in bodies:
            trailing = len(body) - len(body.rstrip("\\"))
            self.assertEqual(
                trailing % 2, 0, f"chunk ends in odd backslashes: {body!r}"
            )


class GenTorchOpParamsTest(unittest.TestCase):
    """Tests for ``gen_torch_op_params`` schema reflection."""

    def _trivial_descriptors(self) -> List[ArgDescriptor]:
        # One pointer, one scalar — enough to exercise the iteration; the
        # interesting bit is the autotune-fields tail.
        return [
            PointerArg(name="x_ptr", index=0, is_optional=False),
            ScalarArg(name="M", index=1, triton_dtype="i32"),
        ]

    @parameterized.expand([("cuda",), ("hip",)])
    def test_includes_all_autotune_fields(self, backend: str) -> None:
        """Torch schema includes every common + backend-specific AutotuneAttrs
        field as a trailing kwarg. Drift would surface as a missing field."""
        result = gen_torch_op_params(
            self._trivial_descriptors(), {}, AutotuneAttrs.fields_for(backend)
        )
        for f in AutotuneAttrs.fields_for(backend):
            self.assertIn(f"{f.name}={f.default}", result)


class GenSelectorParamsTest(unittest.TestCase):
    """Tests for ``gen_selector_params`` (direct, not via gen_selector_proto)."""

    _DESCRIPTORS: List[ArgDescriptor] = [
        PointerArg(name="x_ptr", index=0, is_optional=False),
    ]

    def test_no_defaults_by_default(self) -> None:
        result = gen_selector_params(
            self._DESCRIPTORS, AutotuneAttrs.fields_for("cuda")
        )
        for f in AutotuneAttrs.fields_for("cuda"):
            self.assertIn(f" {f.name}", result)
            self.assertNotIn(f"{f.name}=", result)

    def test_with_defaults_emits_value(self) -> None:
        result = gen_selector_params(
            self._DESCRIPTORS,
            AutotuneAttrs.fields_for("cuda"),
            with_defaults=True,
        )
        for f in AutotuneAttrs.fields_for("cuda"):
            # C++ bool defaults render lowercase (true/false), unlike Python repr.
            dv = (
                ("true" if f.default else "false")
                if isinstance(f.default, bool)
                else f.default
            )
            self.assertIn(f"{f.name}={dv}", result)


class BuildArgDescriptorsTest(unittest.TestCase):
    """Tests for build_arg_descriptors()."""

    def test_classifies_all_args(self) -> None:
        """Every func arg gets a descriptor with correct kind."""

        @triton.jit
        def _kernel(a, b, C: tl.constexpr) -> None:  # noqa: N802
            pass

        unit = OpsUnit(
            cc=90,
            optional=set(),
            pointer_args={0},
            scalar_dtypes={1: "i32"},
            constant_types={2: int},
            specs=[
                KernelSpec(
                    signature={0: "*fp32", 1: "i32"},
                    constants={2: 64},
                    divisible_by_16={0},
                    divisible_by_8=set(),
                )
            ],
        )
        descs = build_arg_descriptors(_kernel, unit)

        self.assertEqual(len(descs), 3)
        d0 = descs[0]
        assert isinstance(d0, PointerArg)
        self.assertEqual(d0.name, "a")
        self.assertFalse(d0.is_optional)

        d1 = descs[1]
        assert isinstance(d1, ScalarArg)
        self.assertEqual(d1.triton_dtype, "i32")

        d2 = descs[2]
        assert isinstance(d2, ConstantArg)
        self.assertEqual(d2.python_type, int)

    def test_optional_tensor(self) -> None:
        """Optional tensor arg has is_optional=True."""

        @triton.jit
        def _kernel(a, b) -> None:
            pass

        unit = OpsUnit(
            cc=90,
            optional={1},
            pointer_args={0, 1},
            scalar_dtypes={},
            constant_types={},
            specs=[
                KernelSpec(
                    signature={0: "*fp32", 1: "*fp32"},
                    constants={},
                    divisible_by_16=set(),
                    divisible_by_8=set(),
                )
            ],
        )
        descs = build_arg_descriptors(_kernel, unit)

        self.assertEqual(len(descs), 2)
        d0 = descs[0]
        assert isinstance(d0, PointerArg)
        self.assertFalse(d0.is_optional)
        d1 = descs[1]
        assert isinstance(d1, PointerArg)
        self.assertTrue(d1.is_optional)

    def test_unclassified_arg_raises(self) -> None:
        """Arg not in any OpsUnit dict raises ValueError."""

        @triton.jit
        def _kernel(a, b) -> None:
            pass

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
                )
            ],
        )
        with self.assertRaises(ValueError):
            build_arg_descriptors(_kernel, unit)

    def test_descriptor_order_matches_arg_names(self) -> None:
        """Descriptors are in func.arg_names order."""

        @triton.jit
        def _kernel(x, N, C: tl.constexpr) -> None:  # noqa: N802
            pass

        unit = OpsUnit(
            cc=90,
            optional=set(),
            pointer_args={0},
            scalar_dtypes={1: "fp32"},
            constant_types={2: bool},
            specs=[
                KernelSpec(
                    signature={0: "*fp32", 1: "fp32"},
                    constants={2: True},
                    divisible_by_16=set(),
                    divisible_by_8=set(),
                )
            ],
        )
        descs = build_arg_descriptors(_kernel, unit)

        self.assertEqual([d.name for d in descs], ["x", "N", "C"])
        self.assertEqual([d.index for d in descs], [0, 1, 2])


class OptionalTensorCodegenTest(unittest.TestCase):
    """Tests for optional tensor handling in generated C++ code."""

    def _create_optional_unit(self) -> OpsUnit:
        """Create OpsUnit with an optional tensor arg."""
        return OpsUnit(
            cc=90,
            optional={2},
            pointer_args={0, 2},
            scalar_dtypes={1: "i32"},
            constant_types={3: int},
            specs=[
                # Spec with optional tensor present
                KernelSpec(
                    signature={0: "*fp32", 1: "i32", 2: "*fp32"},
                    constants={3: 64},
                    divisible_by_16={0, 2},
                    divisible_by_8=set(),
                ),
                # Spec with optional tensor absent
                KernelSpec(
                    signature={0: "*fp32", 1: "i32"},
                    constants={2: None, 3: 64},
                    divisible_by_16={0},
                    divisible_by_8=set(),
                ),
            ],
        )

    def test_selector_params_optional_tensor(self) -> None:
        """Optional tensor arg generates correct C++ types; pointer_args covers None constants."""

        @triton.jit
        def _kernel(a, b, c, D: tl.constexpr) -> None:  # noqa: N802
            pass

        unit = self._create_optional_unit()
        self.assertIn(2, unit.pointer_args)

        descriptors = build_arg_descriptors(_kernel, unit)
        result = gen_selector_params(descriptors, AutotuneAttrs.fields_for("cuda"))
        self.assertIn("const std::optional<torch::stable::Tensor>& c", result)
        self.assertIn("const std::optional<torch::stable::Tensor>& a", result)
        self.assertIn("int32_t b", result)

    def test_guarded_calls_none_guard(self) -> None:
        """Absent optional tensor generates !arg.has_value() guard."""

        @triton.jit
        def _kernel(a, b, c, D: tl.constexpr) -> None:  # noqa: N802
            pass

        unit = self._create_optional_unit()
        descriptors = build_arg_descriptors(_kernel, unit)
        result = gen_guarded_calls(
            _kernel, unit, descriptors, AutotuneAttrs.fields_for("cuda")
        )

        self.assertIn("!c.has_value()", result)
        self.assertIn("c.has_value()", result)

    def test_guarded_calls_reject_kernel_name_collision(self) -> None:
        """Kernel names are cubin identity (hash_kernel_name keys the binary
        files and embedded symbols): two distinct specs mapping to one name
        would silently alias each other's cubins under the guards, so the
        codegen must fail loudly instead."""

        @triton.jit
        def _kernel(a, b, c, D: tl.constexpr) -> None:  # noqa: N802
            pass

        unit = self._create_optional_unit()
        # Duplicate the first spec: same signature/constants/marks/autotune
        # -> identical generated name.
        unit = dataclasses.replace(unit, specs=[unit.specs[0], unit.specs[0]])
        descriptors = build_arg_descriptors(_kernel, unit)
        with self.assertRaisesRegex(RuntimeError, "kernel name collision"):
            gen_guarded_calls(
                _kernel, unit, descriptors, AutotuneAttrs.fields_for("cuda")
            )


class TypeMappingTest(unittest.TestCase):
    """Tests for type-mapping across gen_selector_params, gen_cpp_op_params, gen_torch_op_params."""

    _MIXED_DESCRIPTORS: list[ArgDescriptor] = [
        ScalarArg(name="N", index=0, triton_dtype="i32"),
        ScalarArg(name="lr", index=1, triton_dtype="fp32"),
        ConstantArg(name="mode", index=2, python_type=str),
    ]

    @parameterized.expand(
        [
            (
                "string_constant",
                [ConstantArg(name="mode", index=0, python_type=str)],
                "const std::string& mode",
            ),
            (
                "float_scalar",
                [ScalarArg(name="lr", index=0, triton_dtype="fp32")],
                "float lr",
            ),
            (
                "bool_constant",
                [ConstantArg(name="flag", index=0, python_type=bool)],
                "bool flag",
            ),
        ]
    )
    def test_selector_params(
        self,
        _name: str,
        descriptors: list[ArgDescriptor],
        expected: str,
    ) -> None:
        result = gen_selector_params(descriptors, AutotuneAttrs.fields_for("cuda"))
        self.assertIn(expected, result)

    def test_cpp_op_widens_types(self) -> None:
        """cpp_op params widen: i32→int64_t, fp32→double, str→const std::string&."""
        result = gen_cpp_op_params(
            self._MIXED_DESCRIPTORS, AutotuneAttrs.fields_for("cuda")
        )
        self.assertIn("int64_t N", result)
        self.assertIn("double lr", result)
        self.assertIn("const std::string& mode", result)

    def test_torch_schema_types(self) -> None:
        """torch_op params: i32→int, fp32→float, str→str."""
        result = gen_torch_op_params(
            self._MIXED_DESCRIPTORS, {}, AutotuneAttrs.fields_for("cuda")
        )
        self.assertIn("int N", result)
        self.assertIn("float lr", result)
        self.assertIn("str mode", result)


class LauncherDescriptorTest(unittest.TestCase):
    """Tests for launcher functions with descriptors."""

    _DESCRIPTORS: list[ArgDescriptor] = [
        PointerArg(name="x", index=0, is_optional=False),
        ScalarArg(name="N", index=1, triton_dtype="i32"),
    ]
    _SIGNATURE: dict[int, str] = {0: "*fp32", 1: "i32"}

    def test_launcher_params(self) -> None:
        """Pointer → void*, scalar → CTYPES[dtype]."""
        result = gen_launcher_params(self._DESCRIPTORS, self._SIGNATURE)
        self.assertIn("void* x", result)
        self.assertIn("int32_t N", result)

    def test_launcher_params_skips_constants(self) -> None:
        """Constants not in signature are skipped."""
        descriptors: list[ArgDescriptor] = [
            PointerArg(name="x", index=0, is_optional=False),
            ConstantArg(name="BLK", index=1, python_type=int),
        ]
        result = gen_launcher_params(descriptors, {0: "*fp32"})
        self.assertIn("void* x", result)
        self.assertNotIn("BLK", result)

    def test_launcher_call_args(self) -> None:
        """Pointer → .value().data_ptr(), scalar → pass-through."""
        result = gen_launcher_call_args(self._DESCRIPTORS, self._SIGNATURE)
        self.assertIn("x.value().data_ptr()", result)
        self.assertIn("N", result)

    def test_launcher_call_args_static_cast_narrower_type(self) -> None:
        """Scalar with narrower spec type emits static_cast."""
        descriptors: list[ArgDescriptor] = [
            PointerArg(name="x", index=0, is_optional=False),
            ScalarArg(name="N", index=1, triton_dtype="i64"),
        ]
        # Spec uses i32 (narrower than selector's i64)
        signature: dict[int, str] = {0: "*fp32", 1: "i32"}
        result = gen_launcher_call_args(descriptors, signature)
        self.assertIn("static_cast<int32_t>(N)", result)

    def test_launcher_call_args_same_type_no_cast(self) -> None:
        """Scalar matching spec type has no static_cast."""
        descriptors: list[ArgDescriptor] = [
            PointerArg(name="x", index=0, is_optional=False),
            ScalarArg(name="N", index=1, triton_dtype="i64"),
        ]
        signature: dict[int, str] = {0: "*fp32", 1: "i64"}
        result = gen_launcher_call_args(descriptors, signature)
        self.assertNotIn("static_cast", result)
        self.assertIn("N", result)


class IntWidthCodegenTest(unittest.TestCase):
    """Tests for i32/i64 coexistence codegen (fits_i32 guard + static_cast)."""

    def _create_mixed_int_unit(self) -> OpsUnit:
        """OpsUnit with two specs: one i32, one i64 for scalar arg N."""
        return OpsUnit(
            cc=90,
            optional=set(),
            pointer_args={0},
            scalar_dtypes={1: "i64"},
            constant_types={2: int},
            specs=[
                KernelSpec(
                    signature={0: "*fp32", 1: "i32"},
                    constants={2: 64},
                    divisible_by_16={0},
                    divisible_by_8=set(),
                ),
                KernelSpec(
                    signature={0: "*fp32", 1: "i64"},
                    constants={2: 64},
                    divisible_by_16={0},
                    divisible_by_8=set(),
                ),
            ],
        )

    def test_guarded_calls_i32_i64_coexistence(self) -> None:
        """i32 spec emits fits_i32 guard + static_cast; i64 spec does not."""

        @triton.jit
        def _kernel(x, N, BLK: tl.constexpr) -> None:  # noqa: N802
            pass

        unit = self._create_mixed_int_unit()
        descriptors = build_arg_descriptors(_kernel, unit)
        result = gen_guarded_calls(
            _kernel, unit, descriptors, AutotuneAttrs.fields_for("cuda")
        )

        # Guards are emitted one per line, so a spec is a block of lines rather
        # than a single line; blocks are separated by a blank line.
        specs = result.strip().split("\n\n")
        self.assertEqual(len(specs), 2)
        # First spec (i32): fits_i32 guard + static_cast
        self.assertIn("fits_i32(N)", specs[0])
        self.assertIn("static_cast<int32_t>(N)", specs[0])
        # Second spec (i64): neither fits_i32 nor static_cast
        self.assertNotIn("fits_i32", specs[1])
        self.assertNotIn("static_cast", specs[1])


class IsNonEmptyMappingOfTypeTest(unittest.TestCase):
    """Tests for is_non_empty_mapping_of_type()."""

    @parameterized.expand(
        [
            ("non_dict_list", [1, 2], int, False),
            ("non_dict_str", "abc", str, False),
            ("non_dict_int", 42, int, False),
            ("empty_dict", {}, int, False),
            ("matching_multi", {"a": 1, "b": 2}, int, True),
            ("matching_single", {"x": "hello"}, str, True),
            ("mixed_types", {"a": 1, "b": "two"}, int, False),
            ("single_match", {"k": 3.14}, float, True),
            ("single_mismatch", {"k": 3.14}, int, False),
        ]
    )
    def test_is_non_empty_mapping_of_type(
        self,
        _name: str,
        obj: object,
        value_type: type[object],
        expected: bool,
    ) -> None:
        self.assertEqual(is_non_empty_mapping_of_type(obj, value_type), expected)


class FailureMsgTest(unittest.TestCase):
    """Tests for gen_failure_msg (dispatch-failure error message)."""

    _DESCRIPTORS: list[ArgDescriptor] = [
        PointerArg(name="x", index=0, is_optional=False),
        ScalarArg(name="N", index=1, triton_dtype="i32"),
        ConstantArg(name="BLK", index=2, python_type=int),
    ]

    @parameterized.expand(
        [
            ("tensors_section", "Tensors:"),
            ("scalars_section", "Scalars:"),
            ("constants_section", "Constants:"),
            ("autotune_section", "Autotune:"),
            ("device_section", "Device: cc="),
            ("dtype_via_headeronly", "torch::headeronly::toString"),
            ("aligned16_status", "aligned16="),
            ("alignment_check", "uintptr_t"),
        ]
    )
    def test_output_contains(self, _name: str, expected: str) -> None:
        result = gen_failure_msg(self._DESCRIPTORS, AutotuneAttrs.fields_for("cuda"))
        self.assertIn(expected, result)

    def test_no_raw_pointer_addresses(self) -> None:
        """Error message must not leak raw data_ptr expressions as labels."""
        result = gen_failure_msg(self._DESCRIPTORS, AutotuneAttrs.fields_for("cuda"))
        self.assertNotIn("data_ptr :", result)


class GenLauncherLevel1Test(unittest.TestCase):
    """Tests for gen_launcher with Level 1 launcher_src fast path."""

    def setUp(self) -> None:
        # gen_launcher only takes the Level-1 path when launch.h can be vendored
        # into the build (_launch_header_available). That probe is False in this
        # unit-test environment (no installed launch.h), which would route these
        # Level-1 codegen assertions onto the legacy fallback. Force it True so
        # the Level-1 path under test is exercised.
        patcher = patch(
            "aot_tensor.compile.triton.codegen._launch_header_available",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    _FAKE_LAUNCHER_SRC: str = (
        '#include "nvidia/backend/launch.h"\n'
        "\n"
        "typedef struct { CUdeviceptr A; CUdeviceptr B; int32_t N; } test_kernel_0d1d2_args_t;\n"
        "\n"
        # Real make_launcher_src ABI: 6 params, with global_scratch /
        # profile_scratch as separate trailing args (NOT folded into args_t).
        "static inline CUresult triton_launch_test_kernel_0d1d2(\n"
        "    const uint32_t grid[3], CUstream stream, CUfunction function,\n"
        "    test_kernel_0d1d2_args_t *args,\n"
        "    CUdeviceptr global_scratch, CUdeviceptr profile_scratch) {\n"
        "  /* placeholder */\n"
        "  return CUDA_SUCCESS;\n"
        "}\n"
    )

    def _make_spec(self) -> KernelSpec:
        return KernelSpec(
            signature={0: "*fp32", 1: "*fp32", 2: "i32"},
            constants={3: 1024},
            divisible_by_16={0, 1},
            divisible_by_8=set(),
            autotune=AutotuneAttrs(num_warps=4, num_stages=3),
        )

    def _make_descriptors(self) -> list[ArgDescriptor]:
        return [
            PointerArg(index=0, name="A", is_optional=False),
            PointerArg(index=1, name="B", is_optional=False),
            ScalarArg(index=2, name="N", triton_dtype="i32"),
            ConstantArg(index=3, name="BLOCK", python_type=int),
        ]

    def _make_kernel(self, *, with_launcher_src: bool) -> MagicMock:
        kernel = MagicMock()
        kernel.metadata.name = "test_kernel_0d1d2"
        kernel.metadata.shared = 0
        kernel.metadata.global_scratch_size = 0
        # Explicit trivial cluster: MagicMock would otherwise auto-create a
        # truthy ``cluster_dims`` attr, defeating the ``getattr(..., None)``
        # fallback in ``_get_cluster_dims`` and pushing this legacy-path test
        # onto the cuLaunchKernelEx branch.
        kernel.metadata.cluster_dims = (1, 1, 1)
        if with_launcher_src:
            kernel.asm = {"launcher_src": self._FAKE_LAUNCHER_SRC, "cubin": b""}
        else:
            kernel.asm = {}
        return kernel

    @staticmethod
    @triton.jit
    def _simple_kernel(A, B, N, BLOCK: tl.constexpr):  # pyre-ignore[2,3]
        pass

    @parameterized.expand(
        [
            (
                "level1_fast_path",
                True,
                ["triton_launch_test_kernel_0d1d2", "_args_t args"],
                ["void *args[] ="],
            ),
            (
                "level1_strips_include_uses_stable_abi",
                True,
                ["TRITON_AOT_CU_CHECK", "triton_aot_get_current_stream"],
                ['#include "nvidia/backend/launch.h"', "AT_CUDA_DRIVER_CHECK"],
            ),
            (
                "fallback_no_launcher_src",
                False,
                ["cuLaunchKernel"],
                ["triton_launch_"],
            ),
            (
                "level1_casts_pointers",
                True,
                ["(CUdeviceptr)A", "(CUdeviceptr)B"],
                [],
            ),
        ]
    )
    def test_gen_launcher(
        self,
        _name: str,
        with_launcher_src: bool,
        expected_in: list[str],
        expected_not_in: list[str],
    ) -> None:
        """gen_launcher produces correct output for Level 1 and fallback paths."""
        result = gen_launcher(
            kernel_name="test_kernel_sm80",
            func=self._simple_kernel,
            kernel=self._make_kernel(with_launcher_src=with_launcher_src),
            shared=0,
            warp_size=32,
            spec=self._make_spec(),
            descriptors=self._make_descriptors(),
            backend="cuda",
        )
        for s in expected_in:
            self.assertIn(s, result)
        for s in expected_not_in:
            self.assertNotIn(s, result)

        # Guards both launcher paths against a re-joined parameter list.
        # A length bound would not work here: the real driver of length is the
        # ~130-char kernel symbol, and this mock's name is short, so a re-join
        # would still fit. Assert the structure instead.
        self.assertIn(
            "void test_kernel_sm80(\n",
            result,
            f"launcher (launcher_src={with_launcher_src}) put its parameter "
            "list on the signature line",
        )

    def test_fast_path_scales_scratch_by_num_ctas(self) -> None:
        """Fast (Level 1) path still routes scratch sizing through
        ``get_scratch_parameters``, so ``num_ctas`` (product of cluster_dims)
        must scale ``global_scratch`` even though the launch mechanics come
        from the compiler-generated ``launcher_src``. Previously only the
        fallback path (``GenLauncherClusterTest``) exercised this scaling.
        """
        kernel = self._make_kernel(with_launcher_src=True)
        kernel.metadata.global_scratch_size = 256
        kernel.metadata.cluster_dims = (2, 2, 1)  # num_ctas = 4
        result = gen_launcher(
            kernel_name="test_kernel_sm90",
            func=self._simple_kernel,
            kernel=kernel,
            shared=0,
            warp_size=32,
            spec=self._make_spec(),
            descriptors=self._make_descriptors(),
            backend="cuda",
        )
        # Fast path (compiler launcher_src), not the legacy fallback.
        self.assertIn("triton_launch_test_kernel_0d1d2", result)
        self.assertNotIn("cuLaunchKernel(", result)
        # Our scratch sizing, scaled by num_ctas, forwarded to the launcher.
        self.assertIn("constexpr int64_t global_scratch_per_program = 256;", result)
        self.assertIn("constexpr int64_t global_scratch_num_ctas = 4;", result)
        self.assertIn("global_scratch_per_program * global_scratch_num_ctas", result)
        self.assertIn("global_scratch", result)

    def test_fast_path_call_arity_matches_wrapper(self) -> None:
        """Regression: the generated call to ``triton_launch_<name>`` must pass
        the same number of arguments as the wrapper ``make_launcher_src`` emits,
        including the trailing ``global_scratch`` / ``profile_scratch`` params.

        A stale 4-arg call against the 6-param launch.h ABI silently fails to
        compile (the AOT-T fast path regressed this way after D103308999, while
        TritonCC stayed correct). Scratch must be forwarded as separate wrapper
        args, never folded into ``args_t``.
        """
        result = gen_launcher(
            kernel_name="test_kernel_sm80",
            func=self._simple_kernel,
            kernel=self._make_kernel(with_launcher_src=True),
            shared=0,
            warp_size=32,
            spec=self._make_spec(),
            descriptors=self._make_descriptors(),
            backend="cuda",
        )
        # Param count from the wrapper signature in the launcher_src.
        wrapper_params = self._FAKE_LAUNCHER_SRC.split(
            "triton_launch_test_kernel_0d1d2(", 1
        )[1].split(")", 1)[0]
        n_params = len([p for p in wrapper_params.split(",") if p.strip()])
        # Arg count from the generated call site (only the call is wrapped in
        # TRITON_AOT_CU_CHECK; the embedded definition is not).
        call_args = result.split(
            "TRITON_AOT_CU_CHECK(triton_launch_test_kernel_0d1d2(", 1
        )[1].split(")", 1)[0]
        n_args = len([a for a in call_args.split(",") if a.strip()])
        self.assertEqual(
            n_args,
            n_params,
            f"call passes {n_args} args but wrapper takes {n_params}\n{result}",
        )
        self.assertEqual(n_params, 6)  # grid, stream, function, args, 2x scratch
        # Scratch is forwarded as separate params, not folded into args_t.
        self.assertIn("&args, global_scratch, profile_scratch", result)
        struct_line = next(
            line for line in result.splitlines() if "_args_t args = {" in line
        )
        self.assertNotIn("global_scratch", struct_line)
        self.assertNotIn("profile_scratch", struct_line)


class GenLauncherClusterTest(unittest.TestCase):
    """Unit coverage for the cluster vs legacy branch in ``gen_launcher``.

    Avoids the H100 nvi test being the sole signal for cluster codegen.
    """

    def _mock_kernel(
        self,
        cluster_dims: tuple[int, int, int] | None = (1, 1, 1),
        global_scratch_size: int = 0,
    ) -> Any:
        # ``metadata`` is a SimpleNamespace (not MagicMock) so the
        # ``getattr(..., (1,1,1))`` fallback isn't silently defeated by
        # MagicMock's auto-attribute creation.
        kernel = unittest.mock.MagicMock()
        attrs: dict[str, Any] = {
            "global_scratch_size": global_scratch_size,
            "profile_scratch_size": 0,
        }
        if cluster_dims is not None:
            attrs["cluster_dims"] = cluster_dims
        kernel.metadata = types.SimpleNamespace(**attrs)
        return kernel

    def _gen_for(
        self,
        cluster_dims: tuple[int, int, int] | None = (1, 1, 1),
        backend: str = "cuda",
        global_scratch_size: int = 0,
    ) -> str:
        compiler = CompilerTest()
        spec = compiler._create_mock_unit().specs[0]
        self._spec = spec
        return gen_launcher(
            kernel_name="_addmm_fwd",
            func=compiler._create_mock_func(),
            kernel=self._mock_kernel(cluster_dims, global_scratch_size),
            shared=1024,
            warp_size=32,
            spec=spec,
            descriptors=compiler._create_descriptors(),
            backend=backend,
        )

    @parameterized.expand(
        [
            ("trivial_111_cuda", (1, 1, 1), "cuda"),
            # HIP has no CGA: even non-trivial cluster_dims must stay legacy.
            ("hip_non_trivial_dims_2x1x1", (2, 1, 1), "hip"),
            # Missing ``cluster_dims`` (older Triton / AMD fork) → fallback.
            ("cluster_dims_attr_missing_cuda", None, "cuda"),
        ]
    )
    def test_legacy_cuLaunchKernel_path(
        self,
        _name: str,
        cluster_dims: tuple[int, int, int] | None,
        backend: str,
    ) -> None:
        cpp = self._gen_for(cluster_dims, backend=backend)
        self.assertIn("cuLaunchKernel(", cpp)
        self.assertNotIn("cuLaunchKernelEx", cpp)
        self.assertNotIn("CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION", cpp)
        # Parity with cluster path: ``warp_size * num_warps``.
        self.assertIn(f"32 * {self._spec.autotune.num_warps}", cpp)

    @parameterized.expand(
        [
            ("x_only_2x1x1", (2, 1, 1)),
            ("y_only_1x2x1", (1, 2, 1)),
            ("z_only_1x1x2", (1, 1, 2)),
            ("2x2x1", (2, 2, 1)),
        ]
    )
    def test_non_trivial_cluster_emits_cuLaunchKernelEx(
        self, _name: str, cluster_dims: tuple[int, int, int]
    ) -> None:
        cpp = self._gen_for(cluster_dims)
        cx, cy, cz = cluster_dims
        # Grid-multiply marker is the regression guard for the program→CTA
        # conversion (see _gen_launch_call doc).
        for marker in (
            "cuLaunchKernelEx(&_launch_config",
            "CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION",
            f"_cluster_attrs[0].value.clusterDim = {{ {cx}, {cy}, {cz} }};",
            "_launch_config.numAttrs = 1;",
            f"_launch_config.gridDimX = grid.x * {cx};",
            f"_launch_config.gridDimY = grid.y * {cy};",
            f"_launch_config.gridDimZ = grid.z * {cz};",
            # blockDimX uses the same warp_size * num_warps as the legacy path.
            f"_launch_config.blockDimX = 32 * {self._spec.autotune.num_warps};",
        ):
            self.assertIn(marker, cpp, f"missing `{marker}`")
        self.assertNotIn("cuLaunchKernel(", cpp)

    def test_cluster_plus_scratch_emits_both_sizing_and_launchEx(self) -> None:
        """gen_launcher-level guard that ``_gen_launch_call`` (cluster) and
        ``get_scratch_parameters`` (per-CTA sizing) compose correctly in the
        same .cpp. Previously only the H100 nvi e2e test exercised this.
        """
        cpp = self._gen_for(
            cluster_dims=(2, 2, 1), backend="cuda", global_scratch_size=128
        )
        for marker in (
            "cuLaunchKernelEx(&_launch_config",
            "constexpr int64_t global_scratch_per_program = 128;",
            "constexpr int64_t global_scratch_num_ctas = 4;",
            "global_scratch_per_program * global_scratch_num_ctas",
            "_launch_config.gridDimX = grid.x * 2;",
        ):
            self.assertIn(marker, cpp, f"missing `{marker}`")
        self.assertNotIn("cuLaunchKernel(", cpp)

        # Both of these are multi-line values interpolated into the launcher
        # template, and both are joined at an indent their producer hardcodes.
        # A line deeper than the template's nesting means one of those joins
        # went stale. The deepest legitimate line here is the `void *args[]`
        # entries at _INDENT + _CONT; a body that drifted back to 4-space would
        # push them past this.
        over_indented = [
            line
            for line in cpp.splitlines()
            if line.strip() and len(line) - len(line.lstrip()) > _INDENT + _CONT
        ]
        self.assertEqual(over_indented, [], f"over-indented: {over_indented[:3]}")
