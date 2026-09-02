# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# pyre-strict

import os
import re
import tempfile
import unittest
from unittest.mock import patch

from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.build.triton import amd_extension_builder, nvidia_extension_builder


class GetGpuIncludeDirsTest(unittest.TestCase):
    def test_raises_when_rocm_home_not_set(self) -> None:
        with patch.object(amd_extension_builder, "ROCM_HOME", None):
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "ROCM_HOME/HIP_HOME is not set. Install ROCm toolkit or set "
                    "ROCM_HOME."
                ),
            ):
                amd_extension_builder.AmdExtensionBuilder(
                    source_dir="/tmp",
                    kernel_name="_test_kernel",
                    output_dir="/tmp",
                )

    def test_returns_include_dir_when_hip_header_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            include_dir = os.path.join(tmpdir, "include")
            hip_dir = os.path.join(include_dir, "hip")
            hipblas_common_dir = os.path.join(include_dir, "hipblas-common")
            os.makedirs(hip_dir)
            os.makedirs(hipblas_common_dir)
            with open(os.path.join(hip_dir, "hip_runtime.h"), "w") as f:
                f.write("// HIP runtime header stub")
            with open(os.path.join(hipblas_common_dir, "hipblas-common.h"), "w") as f:
                f.write("// hipBLAS common header stub")

            builder = amd_extension_builder.AmdExtensionBuilder(
                source_dir="/tmp",
                kernel_name="_test_kernel",
                output_dir="/tmp",
                build_config=ExtensionBuildConfig(gpu_toolkit_path=tmpdir),
            )
            result = builder.get_gpu_include_dirs()

            self.assertEqual([include_dir], result)


class GetGpuLibraryDirsTest(unittest.TestCase):
    def test_appends_explicit_gpu_library_dirs_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extra_library_dir = os.path.join(tmpdir, "extra")
            toolkit_library_dir = os.path.join(tmpdir, "lib")
            os.makedirs(extra_library_dir)
            os.makedirs(toolkit_library_dir)
            builder = amd_extension_builder.AmdExtensionBuilder(
                source_dir="/tmp",
                kernel_name="_test_kernel",
                output_dir="/tmp",
                build_config=ExtensionBuildConfig(
                    gpu_toolkit_path=tmpdir,
                    extra_gpu_library_dirs=(extra_library_dir,),
                ),
            )

            result = builder.get_gpu_library_dirs()

            self.assertEqual([extra_library_dir, toolkit_library_dir], result)


class GetExtraCompileArgsTest(unittest.TestCase):
    def test_inherits_the_embed_flags_from_nvidia(self) -> None:
        build_config = ExtensionBuildConfig(gpu_toolkit_path="/tmp")
        nvidia_builder = nvidia_extension_builder.NvidiaExtensionBuilder(
            source_dir="/some/source/dir",
            kernel_name="_test_kernel",
            output_dir="/tmp",
            build_config=build_config,
        )
        builder = amd_extension_builder.AmdExtensionBuilder(
            source_dir="/some/source/dir",
            kernel_name="_test_kernel",
            output_dir="/tmp",
            build_config=build_config,
        )

        nvidia_args = nvidia_builder.get_extra_compile_args()
        args = builder.get_extra_compile_args()

        self.assertIn("--embed-dir=/some/source/dir", nvidia_args)
        self.assertIn("-Wno-c23-extensions", nvidia_args)
        self.assertEqual(
            [
                *nvidia_args,
                *amd_extension_builder.COMMON_HIP_FLAGS,
                "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
            ],
            args,
        )
