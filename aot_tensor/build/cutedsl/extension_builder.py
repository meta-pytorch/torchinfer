# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import logging
import os

from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.build.extension_builder_base import (
    CudaExtensionBuilder,
    SoBuildExtension,
)
from aot_tensor.build.gpu_backend import is_amd
from setuptools import Extension, setup


logger: logging.Logger = logging.getLogger(__name__)

GENERATED_CUDA_LIBRARY_API_MARKER: str = "cudaLibrary_t"
TOOLKIT_CUDA_LIBRARY_API_MARKER: str = "cudaLibraryLoadData"


class CutedslExtensionBuilder(CudaExtensionBuilder):
    """Builds CuTeDSL AOT sidecar and PyTorch op extensions.

    The CuTeDSL-generated object is linked only into the sidecar extension.
    The PyTorch op extension uses dlopen/dlsym to call that sidecar at runtime.
    """

    def _validate_cutedsl_source_files(self) -> tuple[str, str, str]:
        header_file = os.path.join(self.source_dir, f"{self.kernel_name}.h")
        object_file = os.path.join(self.source_dir, f"{self.kernel_name}.o")
        entry_file = os.path.join(self.source_dir, f"{self.kernel_name}_entry.cpp")
        torch_op_file = os.path.join(
            self.source_dir, f"{self.kernel_name}_torch_op.cpp"
        )
        assert os.path.exists(header_file), f"CuTe header not found: {header_file}"
        assert os.path.exists(object_file), f"CuTe object not found: {object_file}"
        assert os.path.exists(entry_file), f"CuTe entry source not found: {entry_file}"
        assert os.path.exists(torch_op_file), (
            f"CuTe torch op source not found: {torch_op_file}"
        )
        return object_file, entry_file, torch_op_file

    def _file_contains(self, path: str, marker: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            return marker in fp.read()

    def _generated_header_requires_cuda_library_api(self) -> bool:
        header_file = os.path.join(self.source_dir, f"{self.kernel_name}.h")
        return self._file_contains(header_file, GENERATED_CUDA_LIBRARY_API_MARKER)

    def _toolkit_supports_cuda_library_api(self, toolkit_path: str) -> bool:
        cuda_runtime_api = os.path.join(toolkit_path, "include", "cuda_runtime_api.h")
        return self._file_contains(cuda_runtime_api, TOOLKIT_CUDA_LIBRARY_API_MARKER)

    def _select_cutedsl_gpu_toolkit_path(self) -> None:
        if not self._generated_header_requires_cuda_library_api():
            return
        if self._toolkit_supports_cuda_library_api(self.gpu_toolkit_path):
            return

        for candidate in self._candidate_cuda_toolkits():
            if candidate == self.gpu_toolkit_path:
                continue
            if self._toolkit_supports_cuda_library_api(candidate):
                logger.info(
                    "Using CUDA toolkit %s for CuTeDSL AOT because %s does not expose cudaLibrary* APIs",
                    candidate,
                    self.gpu_toolkit_path,
                )
                self.gpu_toolkit_path = candidate
                return

        raise RuntimeError(
            "CuTeDSL AOT generated cudaLibrary* API calls, but the selected CUDA "
            f"toolkit at {self.gpu_toolkit_path} does not expose those APIs. "
            "Use a CUDA toolkit with cudaLibraryLoadData support."
        )

    def build(self) -> str:
        object_file, entry_file, torch_op_file = self._validate_cutedsl_source_files()
        self._select_cutedsl_gpu_toolkit_path()

        gpu_include_dirs = self.get_gpu_include_dirs()
        gpu_lib_dirs = self.get_gpu_library_dirs()
        torch_include_dirs = self.get_torch_include_dirs()
        extra_compile_args = self.get_extra_compile_args()

        sidecar_name = f"{self.ext_name}_cutedsl_impl"
        sidecar_extension = Extension(
            name=sidecar_name,
            sources=[entry_file],
            include_dirs=[self.source_dir] + gpu_include_dirs,
            library_dirs=gpu_lib_dirs,
            libraries=["cuda", "cudart", "dl"],
            extra_compile_args=extra_compile_args,
            extra_objects=[object_file],
            language="c++",
        )

        torch_op_extension = Extension(
            name=self.ext_name,
            sources=[torch_op_file],
            include_dirs=gpu_include_dirs + torch_include_dirs,
            library_dirs=gpu_lib_dirs,
            libraries=["cuda", "dl"],
            extra_compile_args=extra_compile_args,
            language="c++",
        )

        script_args = [
            "build_ext",
            f"--build-lib={self.output_dir}",
            f"--build-temp={self.output_dir}/build_temp",
        ]
        if self.build_config.compiler_path is not None:
            script_args.append(f"--compiler-path={self.build_config.compiler_path}")

        setup(
            name=self.ext_name,
            ext_modules=[sidecar_extension, torch_op_extension],
            script_args=script_args,
            cmdclass={"build_ext": SoBuildExtension},
        )

        sidecar_output = os.path.join(self.output_dir, f"{sidecar_name}.so")
        assert os.path.exists(sidecar_output), (
            f"Expected built CuTe sidecar extension: {sidecar_output}"
        )
        output = os.path.join(self.output_dir, f"{self.ext_name}.so")
        assert os.path.exists(output), f"Expected built extension: {output}"
        return output


def build_cutedsl_aot_extension(
    source_dir: str,
    kernel_name: str,
    output_dir: str,
    build_config: ExtensionBuildConfig | None = None,
) -> str:
    """Build a CuTeDSL AOT op extension plus its dynamically loaded sidecar."""
    if is_amd():
        raise NotImplementedError("CuTeDSL AOT is only supported on NVIDIA CUDA.")

    return CutedslExtensionBuilder(
        source_dir,
        kernel_name,
        output_dir,
        build_config=build_config,
    ).build()
