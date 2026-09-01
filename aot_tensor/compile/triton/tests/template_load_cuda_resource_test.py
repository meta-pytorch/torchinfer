# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""Test that CUDA templates contain correct NVIDIA APIs.

This test validates the CUDA (NVIDIA) template resources by checking that
they contain the expected CUDA-specific APIs and headers. It runs on CPU
machines without requiring GPU hardware since it only reads template files.
"""

import unittest

# @dep=//aot_tensor/compile/triton/templates:triton_templates
from aot_tensor.compile.template_utils import TRITON_TEMPLATES


class TemplateCudaApiTest(unittest.TestCase):
    def test_kernel_cpp_has_cuda_apis(self) -> None:
        """Test kernel.cpp contains CUDA-specific headers and APIs."""
        kernel_cpp = TRITON_TEMPLATES.load("kernel.cpp")

        self.assertIn("torch/csrc/stable/accelerator.h", kernel_cpp)
        self.assertIn("TRITON_AOT_CU_CHECK", kernel_cpp)
        self.assertIn("cuDeviceGetAttribute", kernel_cpp)
        self.assertIn("CUfunction", kernel_cpp)

        # Verify HIP headers are NOT present.
        # Does not include hipFunction_t which already defined under USE_ROCM
        self.assertNotIn("hip/HIPContext.h", kernel_cpp)
        self.assertNotIn("HIPStreamMasqueradingAsCUDA.h", kernel_cpp)

        # D2: device/stream uses stable ABI
        self.assertIn("triton_aot_get_current_stream", kernel_cpp)
        self.assertNotIn("c10/cuda/CUDAStream.h", kernel_cpp)
        self.assertNotIn("c10::cuda::current_device", kernel_cpp)

        # D1: error handling uses TRITON_AOT_CU_CHECK, not ATen macros
        self.assertIn("TRITON_AOT_CU_CHECK", kernel_cpp)
        self.assertNotIn("AT_CUDA_DRIVER_CHECK", kernel_cpp)

        # D4: no ATen headers remain
        self.assertNotIn("ATen/Tensor.h", kernel_cpp)
        self.assertNotIn("torch/library.h", kernel_cpp)
        self.assertIn("torch/headeronly/core/ScalarType.h", kernel_cpp)

        # D5: scratch alloc uses stable Tensor, not AOTI runtime RAII handle
        self.assertIn("torch/csrc/stable/tensor.h", kernel_cpp)
        self.assertIn("torch/csrc/stable/macros.h", kernel_cpp)
        self.assertNotIn("torch/csrc/inductor/aoti_runtime/utils.h", kernel_cpp)
        self.assertNotIn("RAIIAtenTensorHandle", kernel_cpp)

        # D6: enable_large_smem_or_throw replaces the legacy ``check_errors``
        # (which only did SMEM opt-in setup despite the name). Validates
        # both the rename and the new hard-reject for over-SMEM specs.
        self.assertIn("enable_large_smem_or_throw", kernel_cpp)
        self.assertNotIn("void check_errors", kernel_cpp)
        # Uses cuCtxGetDevice for the current device (NOT hardcoded 0).
        self.assertIn("cuCtxGetDevice", kernel_cpp)
        self.assertNotIn("int device = 0", kernel_cpp)
        # Hard-rejects specs whose ``shared`` exceeds the device opt-in
        # cap with a clear error message at load time, instead of letting
        # downstream cuLaunchKernel fail with CUDA_ERROR_INVALID_VALUE.
        self.assertIn("shared > shared_optin", kernel_cpp)
        self.assertIn("max opt-in", kernel_cpp)

    def test_kernel_h_has_cuda_apis(self) -> None:
        """Test kernel.h contains CUDA-specific types."""
        kernel_h = TRITON_TEMPLATES.load("kernel.h")

        self.assertIn("cudaStream_t", kernel_h)

        self.assertNotIn("hipStream_t", kernel_h)
        self.assertNotIn("hip/hip_runtime.h", kernel_h)

        # D4: stable ABI headers replace ATen
        self.assertIn("torch/csrc/stable/tensor.h", kernel_h)
        self.assertNotIn("ATen/Tensor.h", kernel_h)
        self.assertNotIn("torch/types.h", kernel_h)

    def test_torch_op_cpp_has_stable_abi(self) -> None:
        """Test torch_op.cpp uses stable ABI registration."""
        torch_op = TRITON_TEMPLATES.load("torch_op.cpp")

        self.assertIn("STABLE_TORCH_LIBRARY_FRAGMENT", torch_op)
        self.assertIn("STABLE_TORCH_LIBRARY_IMPL", torch_op)
        self.assertIn("TORCH_BOX", torch_op)
        self.assertIn("torch/csrc/stable/library.h", torch_op)

        self.assertNotIn("ATen/Tensor.h", torch_op)
        self.assertNotIn(
            "TORCH_LIBRARY_FRAGMENT",
            torch_op.replace("STABLE_TORCH_LIBRARY_FRAGMENT", ""),
        )
