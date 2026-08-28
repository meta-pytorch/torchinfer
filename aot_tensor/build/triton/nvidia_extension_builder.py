"""
Triton-NVIDIA extension builder: the CUDA cubin-embedding build() flow.

`NvidiaExtensionBuilder` adds the Triton cubin-embedding `build()` (and its
launch-header vendoring + embedded-cubin generation) on top of the CUDA
hardware hooks in `CudaExtensionBuilder`. AMD subclasses this for the HIP cubin
path; CuTeDSL does NOT (it has its own sidecar build()).
"""

import logging
import os
import re
import shutil

from aot_tensor.build.extension_builder_base import (
    CudaExtensionBuilder,
    SoBuildExtension,
)
from aot_tensor.build.triton import cubin_embedder
from aot_tensor.compile.triton.launch_header import find_launch_header
from setuptools import Extension, setup


logger: logging.Logger = logging.getLogger(__name__)

# Generated files, for cubin embedder
EMBEDDED_CUBIN_FILENAME: str = "embedded_kernels_autogen.cpp"

# Regex pattern for extracting kernel names from cubin variable declarations
CUBIN_VAR_PATTERN: re.Pattern[str] = re.compile(r"extern unsigned char (\w+)_cubin\[\]")


def extract_kernel_variants_from_cpp_files(source_dir: str) -> list[str]:
    """
    Extract kernel variant names from .cpp files by finding cubin variable declarations.

    Matches pattern: extern unsigned char xxx_cubin[];
    Returns kernel variant names (e.g., '_addmm_fwd_sm80_pfp32_pfp32_pfp32_pfp32_i32_')
    with '_cubin' suffix removed.
    """
    kernel_variants = []
    cpp_files = [f for f in os.listdir(source_dir) if f.endswith(".cpp")]

    for cpp_file in cpp_files:
        cpp_path = os.path.join(source_dir, cpp_file)
        with open(cpp_path, "r") as f:
            content = f.read()
            matches = CUBIN_VAR_PATTERN.findall(content)
            kernel_variants.extend(matches)

    return kernel_variants


class NvidiaExtensionBuilder(CudaExtensionBuilder):
    """
    Triton extension builder for the NVIDIA CUDA backend.

    Embeds pre-compiled cubins into the generated C++ and compiles the kernel +
    torch-op sources into a single `.so`.
    """

    def _vendor_launch_header(self) -> None:
        """Copy launch.h into ``source_dir/nvidia/backend/launch.h``.

        The generated kernel.cpp includes "nvidia/backend/launch.h" at file
        scope (the shared Level-1 launch core used by KERNEL_SPECS). This
        standalone setuptools compile has no triton_launch_h dep and does not
        add the triton package to ``-I``; vendoring the header next to the
        generated sources lets the quoted include resolve (the compiler searches
        the including file's directory). Mirrors the vendored copy driver.py
        inlines for the JIT runtime compile.
        """
        src = find_launch_header()
        if src is None:
            return  # let the compiler emit a clear error if it's actually needed
        dst_dir = os.path.join(self.source_dir, "nvidia", "backend")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src, os.path.join(dst_dir, "launch.h"))

    def get_extra_compile_args(self) -> list[str]:
        """Flags the generated `#embed` TU needs. Not on the base: only this
        class, and AmdExtensionBuilder which inherits it, emits `#embed`."""
        args = super().get_extra_compile_args()
        # clang resolves #embed through its own search path, not -I.
        args.append(f"--embed-dir={self.source_dir}")
        # clang calls #embed a C23 extension in every C++ mode, c++26 included,
        # so no -std bump removes the need for this.
        args.append("-Wno-c23-extensions")
        return args

    def generate_embedded_kernels(
        self, output_filename: str, kernel_variants: list[str]
    ) -> None:
        """Generate embedded cubin cpp file."""
        cubin_embedder.generate_cpp_for_kernel_binaries(
            output_filename=output_filename,
            kernel_variants=kernel_variants,
            binary_dir=self.source_dir,
        )

    def validate_source_files(self) -> tuple[str, str]:
        """Validate that required source files exist and return their paths."""
        cpp_file = os.path.join(self.source_dir, f"{self.kernel_name}.cpp")
        torch_op_file = os.path.join(
            self.source_dir, f"{self.kernel_name}_torch_op.cpp"
        )
        assert os.path.exists(cpp_file), f"Kernel source not found: {cpp_file}"
        assert os.path.exists(torch_op_file), (
            f"Torch op source not found: {torch_op_file}"
        )
        return cpp_file, torch_op_file

    def build(self) -> str:
        """
        Build a Triton AOT kernel as a PyTorch C++ extension.

        Returns:
            Path to the built .so file.

        Raises:
            RuntimeError: If CUDA/HIP is not properly configured or build fails.
            AssertionError: If required source files are missing.
        """
        # Validate source files
        cpp_file, torch_op_file = self.validate_source_files()
        # Subclasses may contribute additional TUs (e.g. kernel foundry's
        # split wrapper-function sources) compiled into the same extension.
        extra_sources: list[str] = list(getattr(self, "extra_sources", []))

        # Vendor launch.h next to the generated sources so the file-scope
        # #include "nvidia/backend/launch.h" (Level-1 launcher) resolves here.
        self._vendor_launch_header()

        kernel_variants = extract_kernel_variants_from_cpp_files(self.source_dir)
        assert kernel_variants, f"No cubin references found in {self.source_dir}/*.cpp"

        # Generate embedded cubin/hsaco cpp file
        embedded_cubin_filename = f"{self.kernel_name}_embedded_kernels_autogen.cpp"
        embedded_cubin_cpp = os.path.join(self.output_dir, embedded_cubin_filename)
        self.generate_embedded_kernels(embedded_cubin_cpp, kernel_variants)

        # Get all include and library directories
        gpu_include_dirs = self.get_gpu_include_dirs()
        gpu_lib_dirs = self.get_gpu_library_dirs()
        torch_include_dirs = self.get_torch_include_dirs()
        libraries = self.get_libraries()
        extra_compile_args = self.get_extra_compile_args()

        ext_module = Extension(
            name=self.ext_name,
            sources=[cpp_file, torch_op_file, *extra_sources, embedded_cubin_cpp],
            include_dirs=gpu_include_dirs + torch_include_dirs,
            library_dirs=gpu_lib_dirs,
            libraries=libraries,
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
            ext_modules=[ext_module],
            script_args=script_args,
            cmdclass={"build_ext": SoBuildExtension},
        )

        so_path = os.path.join(self.output_dir, f"{self.ext_name}.so")
        assert os.path.exists(so_path), f"Build failed: .so file not found at {so_path}"

        return so_path
