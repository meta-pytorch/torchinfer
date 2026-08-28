# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""Test that AMD templates contain correct HIP APIs.

This test validates the AMD (HIP/ROCm) template resources by checking that
they contain the expected HIP-specific APIs and headers. It runs on CPU
machines without requiring GPU hardware since it only reads template files.
"""

import unittest

from aot_tensor.compile.template_utils import TRITON_TEMPLATES


class TemplateAmdApiTest(unittest.TestCase):
    """Verify AMD templates contain correct HIP APIs."""

    def test_kernel_cpp_has_hip_apis(self) -> None:
        """Test kernel.cpp contains HIP-specific headers and APIs."""
        kernel_cpp = TRITON_TEMPLATES.load("kernel.cpp")

        self.assertIn("torch/csrc/stable/accelerator.h", kernel_cpp)
        self.assertIn("hipFunction_t", kernel_cpp)
        self.assertIn("hipDeviceGetAttribute", kernel_cpp)
        self.assertIn("TRITON_AOT_CU_CHECK", kernel_cpp)
        self.assertIn("triton_aot_get_current_stream", kernel_cpp)

        self.assertNotIn("cuda/CUDAContext.h", kernel_cpp)
        self.assertNotIn("c10/cuda/CUDAStream.h", kernel_cpp)
        self.assertNotIn("CUfunction", kernel_cpp)
        self.assertNotIn("AT_CUDA_DRIVER_CHECK", kernel_cpp)

        # D4: no ATen headers remain
        self.assertNotIn("ATen/Tensor.h", kernel_cpp)
        self.assertNotIn("torch/library.h", kernel_cpp)

        # D5: scratch alloc uses stable Tensor, not AOTI runtime RAII handle
        self.assertNotIn("torch/csrc/inductor/aoti_runtime/utils.h", kernel_cpp)
        self.assertNotIn("RAIIAtenTensorHandle", kernel_cpp)

        # D6: enable_large_smem_or_throw is the renamed check_errors. AMD
        # variant is a no-op (HIP handles dynamic SMEM differently), but the
        # symbol must still exist so codegen ``load_*()`` calls link.
        self.assertIn("enable_large_smem_or_throw", kernel_cpp)
        self.assertNotIn("void check_errors", kernel_cpp)

    def test_kernel_h_has_hip_apis(self) -> None:
        """Test kernel.h contains HIP-specific types."""
        kernel_h = TRITON_TEMPLATES.load("kernel.h")

        self.assertIn("hipStream_t", kernel_h)

        self.assertNotIn("cudaStream_t", kernel_h)
        self.assertNotIn("<cuda.h>", kernel_h)

        # D4: stable ABI headers replace ATen
        self.assertNotIn("ATen/Tensor.h", kernel_h)

    def test_torch_op_cpp_has_stable_abi(self) -> None:
        """Test torch_op.cpp uses stable ABI registration."""
        torch_op = TRITON_TEMPLATES.load("torch_op.cpp")

        self.assertIn("STABLE_TORCH_LIBRARY_FRAGMENT", torch_op)
        self.assertIn("STABLE_TORCH_LIBRARY_IMPL", torch_op)
        self.assertIn("TORCH_BOX", torch_op)

        self.assertNotIn("ATen/Tensor.h", torch_op)
        self.assertNotIn("torch/library.h", torch_op)
