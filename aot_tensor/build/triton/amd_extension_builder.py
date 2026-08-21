# pyre-strict
"""
AMD HIP (ROCm) extension builder for Triton AOT kernels.

This module contains the AmdExtensionBuilder class that inherits from
NvidiaExtensionBuilder and overrides GPU-specific methods for AMD/HIP compilation.
"""

import os

from aot_tensor.build.triton import cubin_embedder
from aot_tensor.build.triton.nvidia_extension_builder import NvidiaExtensionBuilder
from torch.utils.cpp_extension import COMMON_HIP_FLAGS, ROCM_HOME


# Fallback ROCm include path for missing headers in certain ROCm versions (e.g., 6.2.x)
DEFAULT_ROCM_INCLUDE: str = "/opt/rocm/include"


class AmdExtensionBuilder(NvidiaExtensionBuilder):
    """
    Extension builder for AMD HIP (ROCm) backend.

    Templates are pre-hipified at Buck build time, and compiler.py generates
    HIP code directly (hipFunction_t, hipModuleLaunchKernel, etc.).
    Thus we need ROCm, but no runtime hipification is needed here.
    """

    def _default_gpu_toolkit_path(self) -> str:
        if ROCM_HOME is None:
            raise RuntimeError(
                "ROCM_HOME/HIP_HOME is not set. Install ROCm toolkit or set ROCM_HOME."
            )
        return ROCM_HOME

    def get_gpu_include_dirs(self) -> list[str]:
        """
        Return HIP include directory paths, validated.

        Uses ROCM_HOME/include as the primary include directory.
        Attempts to find additional headers from /opt/rocm.
        """
        include_dir = os.path.join(self.gpu_toolkit_path, "include")
        hip_header = os.path.join(include_dir, "hip", "hip_runtime.h")
        if not os.path.exists(hip_header):
            raise RuntimeError(
                f"HIP header not found at {hip_header}. ROCm Toolkit must be installed."
            )

        include_dirs = [include_dir]

        # ROCm 6.2.x is missing hipblas-common headers. If not found, try /opt/rocm.
        hipblas_common = os.path.join(include_dir, "hipblas-common", "hipblas-common.h")
        if not os.path.exists(hipblas_common):
            if os.path.exists(
                os.path.join(DEFAULT_ROCM_INCLUDE, "hipblas-common", "hipblas-common.h")
            ):
                include_dirs.insert(0, DEFAULT_ROCM_INCLUDE)

        return include_dirs

    def get_gpu_library_dirs(self) -> list[str]:
        """
        Return directories containing HIP libraries (libamdhip64.so).
        """
        candidates = [
            *self.build_config.extra_gpu_library_dirs,
            os.path.join(self.gpu_toolkit_path, "lib"),
            os.path.join(self.gpu_toolkit_path, "lib64"),
            os.path.join(self.gpu_toolkit_path, "hip", "lib"),
        ]

        result = [d for d in candidates if os.path.isdir(d)]
        if not result:
            raise RuntimeError(
                f"No HIP library directories found in {self.gpu_toolkit_path}. Searched: {candidates}"
            )
        return result

    def get_libraries(self) -> list[str]:
        """Return HIP libraries to link against."""
        return ["amdhip64"]

    def get_extra_compile_args(self) -> list[str]:
        """Return HIP-specific compiler arguments."""
        # HIP keeps -DUSE_CUDA (from CudaExtensionBuilder) on purpose so shim.h's CUDA-stream declarations stay visible
        args = super().get_extra_compile_args()
        args.extend(COMMON_HIP_FLAGS)

        # The c10/cuda/impl/cuda_cmake_macros.h is not generated for the
        # hip build yet.
        args.append("-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE")
        return args

    def generate_embedded_kernels(
        self, output_filename: str, kernel_variants: list[str]
    ) -> None:
        """Generate embedded hsaco cpp file."""
        cubin_embedder.generate_cpp_for_kernel_binaries(
            output_filename=output_filename,
            kernel_variants=kernel_variants,
            binary_dir=self.source_dir,
            is_amd=True,
        )

    def get_torch_device_type(self) -> str:
        """Return the torch device type for include path lookup."""
        return "hip"
