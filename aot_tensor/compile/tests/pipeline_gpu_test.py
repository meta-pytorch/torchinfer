# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-ignore-all-errors[2]: triton func without type

import importlib
import tempfile
import unittest
from typing import List

# @manual=//triton:triton
import triton
import triton.language as tl
from aot_tensor.compile.triton.arg_descriptor import (
    ArgDescriptor,
    ConstantArg,
    PointerArg,
    ScalarArg,
)
from aot_tensor.compile.triton.pipeline import compile_specs_parallel
from aot_tensor.compile.triton.spec_processing import AutotuneAttrs, KernelSpec
from triton.backends.compiler import GPUTarget


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


class CompileSpecsParallelTest(unittest.TestCase):
    def _create_valid_specs(self) -> List[KernelSpec]:
        """Create valid specs for _addmm_fwd that can be compiled."""
        return [
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
        ]

    def _create_valid_descriptors(self) -> list[ArgDescriptor]:
        """Create arg descriptors matching _addmm_fwd specs."""
        return [
            PointerArg(name="x_ptr", index=0, is_optional=False),
            PointerArg(name="w_ptr", index=1, is_optional=False),
            PointerArg(name="y_ptr", index=2, is_optional=False),
            PointerArg(name="z_ptr", index=3, is_optional=False),
            ScalarArg(name="M", index=4, triton_dtype="i32"),
            ScalarArg(name="N", index=5, triton_dtype="i32"),
            ScalarArg(name="K", index=6, triton_dtype="i32"),
            ScalarArg(name="stride_xm", index=7, triton_dtype="i32"),
            ConstantArg(name="stride_xk", index=8, python_type=int),
            ScalarArg(name="stride_wk", index=9, triton_dtype="i32"),
            ConstantArg(name="stride_wn", index=10, python_type=int),
            ScalarArg(name="stride_ym", index=11, triton_dtype="i32"),
            ConstantArg(name="stride_yn", index=12, python_type=int),
            ScalarArg(name="stride_zm", index=13, triton_dtype="i32"),
            ConstantArg(name="stride_zn", index=14, python_type=int),
            ConstantArg(name="BLOCK_M", index=15, python_type=int),
            ConstantArg(name="BLOCK_N", index=16, python_type=int),
            ConstantArg(name="BLOCK_K", index=17, python_type=int),
            ConstantArg(name="GROUP_M", index=18, python_type=int),
            ConstantArg(name="ALLOW_TF32", index=19, python_type=int),
            ConstantArg(name="BROADCAST_Y", index=20, python_type=int),
        ]

    def test_compile_specs_parallel_generates_code(self) -> None:
        """Test that compile_specs_parallel generates cubin, loader, and launcher code."""
        specs = self._create_valid_specs()
        descriptors = self._create_valid_descriptors()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = compile_specs_parallel(
                specs=specs,
                install_dir=tmpdir,
                module="aot_tensor.compile.tests.pipeline_gpu_test",
                name="_addmm_fwd",
                gpu_target=GPUTarget(backend="cuda", arch=80, warp_size=32),
                import_module=importlib.import_module,
                descriptors=descriptors,
            )

            # result is a list of (code, shared) tuples, one per spec
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), len(specs))

            # Each element should contain cubin, loader, and launcher code
            for spec_code, shared in result:
                self.assertIn("_addmm_fwd", spec_code)
                self.assertIn("cubin", spec_code)
                self.assertIn("load_", spec_code)
                self.assertIn("CUfunction", spec_code)
                self.assertEqual(shared, 0)
