# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Router module for Triton AOT extension builders.

This module provides the main entry point for building Triton AOT kernels.
It automatically selects the appropriate builder (NVIDIA or AMD) based on
the PyTorch build configuration.
"""

from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.build.gpu_backend import is_amd
from aot_tensor.build.triton.amd_extension_builder import AmdExtensionBuilder
from aot_tensor.build.triton.nvidia_extension_builder import NvidiaExtensionBuilder


def build_triton_aot_extension(
    source_dir: str,
    kernel_name: str,
    output_dir: str,
    build_config: ExtensionBuildConfig | None = None,
) -> str:
    """
    Build a Triton AOT kernel as a PyTorch C++ extension.

    This function compiles Triton AOT generated C++ sources into a shared library
    that can be loaded by PyTorch.

    Supports both NVIDIA CUDA and AMD HIP (ROCm) backends. The backend is
    automatically detected based on the PyTorch build configuration.

    Args:
        source_dir: Directory containing the generated C++ sources and cubin/hsaco files.
        kernel_name: Name of the kernel (e.g., "_addmm_fwd").
        output_dir: Directory to place the built .so file.
        build_config: Optional compiler, include-path, and library-path overrides.

    Returns:
        Path to the built .so file.

    Raises:
        RuntimeError: If CUDA/HIP is not properly configured or build fails.
        AssertionError: If required source files are missing.
    """
    if is_amd():
        builder = AmdExtensionBuilder(
            source_dir, kernel_name, output_dir, build_config=build_config
        )
    else:
        builder = NvidiaExtensionBuilder(
            source_dir, kernel_name, output_dir, build_config=build_config
        )

    return builder.build()
