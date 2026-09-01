# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# pyre-strict

import os
import tempfile
import unittest
from dataclasses import replace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from aot_tensor.build import extension_builder_base
from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.build.triton import nvidia_extension_builder
from parameterized import parameterized
from setuptools import Distribution


class ExtractKernelVariantsFromCppFilesTest(unittest.TestCase):
    def test_extracts_kernel_variants(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir:
            with open(os.path.join(source_dir, "kernel.cpp"), "w") as f:
                f.write(
                    "extern unsigned char first_kernel_cubin[];\n"
                    "extern const void* volatile first_kernel_cubin_ptr;"
                )
            with open(os.path.join(source_dir, "second_kernel.cpp"), "w") as f:
                f.write("extern unsigned char second_kernel_cubin[];")
            with open(os.path.join(source_dir, "ignored.h"), "w") as f:
                f.write("extern unsigned char ignored_kernel_cubin[];")

            result = nvidia_extension_builder.extract_kernel_variants_from_cpp_files(
                source_dir
            )

        self.assertIsInstance(result, list)
        self.assertCountEqual(["first_kernel", "second_kernel"], result)
        for name in result:
            self.assertFalse(name.endswith("_cubin"))

    @parameterized.expand(
        [
            ("empty_directory", False),
            ("cpp_without_cubin_refs", True),
        ]
    )
    def test_returns_empty_when_no_cubin_refs(
        self, _name: str, write_cpp: bool
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            if write_cpp:
                with open(os.path.join(tmpdir, "test.cpp"), "w") as f:
                    f.write("int main() { return 0; }")

            result = nvidia_extension_builder.extract_kernel_variants_from_cpp_files(
                tmpdir
            )
            self.assertEqual([], result)


class GetExtraCompileArgsTest(unittest.TestCase):
    """Nothing on the CPU side compiles the generated `#embed` TU, so dropping
    either flag would otherwise only surface in the GPU end-to-end build."""

    def test_emits_expected_compile_args_in_order(self) -> None:
        builder = nvidia_extension_builder.NvidiaExtensionBuilder(
            source_dir="/some/source/dir",
            kernel_name="_test_kernel",
            output_dir="/tmp",
            build_config=ExtensionBuildConfig(gpu_toolkit_path="/tmp"),
        )

        self.assertEqual(
            [
                f"-std={extension_builder_base.CPP_STANDARD}",
                "-fPIC",
                f"-DTORCH_TARGET_VERSION={extension_builder_base.TORCH_TARGET_VERSION}",
                "-v",
                "-DUSE_CUDA",
                "--embed-dir=/some/source/dir",
                "-Wno-c23-extensions",
            ],
            builder.get_extra_compile_args(),
        )


class SoBuildExtensionOptionsTest(unittest.TestCase):
    def test_defaults_to_setuptools_compiler(self) -> None:
        command = extension_builder_base.SoBuildExtension(Distribution())

        self.assertIsNone(command.compiler_path)

    def test_parses_explicit_compiler_argument(self) -> None:
        compiler_path = "/opt/clang/bin/clang"
        distribution = Distribution(
            {
                "cmdclass": {"build_ext": extension_builder_base.SoBuildExtension},
            }
        )
        distribution.script_args = [
            "build_ext",
            f"--compiler-path={compiler_path}",
        ]

        self.assertTrue(distribution.parse_command_line())

        command = cast(
            extension_builder_base.SoBuildExtension,
            distribution.get_command_obj("build_ext"),
        )

        self.assertEqual(compiler_path, command.compiler_path)

    def test_applies_explicit_compiler_to_build(self) -> None:
        compiler_path = "/opt/clang/bin/clang"
        command = extension_builder_base.SoBuildExtension(Distribution())
        command.compiler_path = compiler_path
        compiler = MagicMock()

        with (
            patch.object(command, "compiler", compiler),
            patch.object(
                extension_builder_base.build_ext, "build_extensions"
            ) as parent_build_extensions,
        ):
            command.build_extensions()

        compiler.set_executables.assert_called_once_with(
            compiler=compiler_path,
            compiler_so=compiler_path,
            compiler_cxx=compiler_path,
            linker_so=f"{compiler_path} -shared",
            linker_exe=compiler_path,
        )
        parent_build_extensions.assert_called_once_with()


class GpuToolkitPathTest(unittest.TestCase):
    def test_uses_explicit_gpu_toolkit_path(self) -> None:
        with tempfile.TemporaryDirectory() as gpu_toolkit_path:
            builder = nvidia_extension_builder.NvidiaExtensionBuilder(
                source_dir="/tmp",
                kernel_name="_test_kernel",
                build_config=ExtensionBuildConfig(gpu_toolkit_path=gpu_toolkit_path),
            )

            self.assertEqual(gpu_toolkit_path, builder.gpu_toolkit_path)

    def test_uses_backend_default_when_gpu_toolkit_path_is_not_configured(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as gpu_toolkit_path:
            with patch.object(extension_builder_base, "CUDA_HOME", gpu_toolkit_path):
                builder = nvidia_extension_builder.NvidiaExtensionBuilder(
                    source_dir="/tmp",
                    kernel_name="_test_kernel",
                )

        self.assertEqual(gpu_toolkit_path, builder.gpu_toolkit_path)


class BuildScriptArgsTest(unittest.TestCase):
    def _setup_kwargs(
        self, build_config: ExtensionBuildConfig | None = None
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel_name = "_test_kernel"
            for filename in [f"{kernel_name}.cpp", f"{kernel_name}_torch_op.cpp"]:
                with open(os.path.join(tmpdir, filename), "w") as f:
                    f.write("")
            with open(os.path.join(tmpdir, "test_kernel.so"), "w") as f:
                f.write("")

            resolved_build_config = replace(
                build_config or ExtensionBuildConfig(),
                gpu_toolkit_path=tmpdir,
            )
            builder = nvidia_extension_builder.NvidiaExtensionBuilder(
                source_dir=tmpdir,
                kernel_name=kernel_name,
                output_dir=tmpdir,
                build_config=resolved_build_config,
            )
            with (
                patch.object(builder, "_vendor_launch_header"),
                patch.object(builder, "generate_embedded_kernels"),
                patch.object(builder, "get_gpu_include_dirs", return_value=[]),
                patch.object(builder, "get_gpu_library_dirs", return_value=[]),
                patch.object(builder, "get_torch_include_dirs", return_value=[]),
                patch.object(
                    nvidia_extension_builder,
                    "extract_kernel_variants_from_cpp_files",
                    return_value=["variant"],
                ),
                patch.object(nvidia_extension_builder, "setup") as setup_mock,
            ):
                builder.build()

        return dict(setup_mock.call_args.kwargs)

    def test_omits_compiler_argument_when_not_configured(self) -> None:
        setup_kwargs = self._setup_kwargs()
        script_args = cast(list[str], setup_kwargs["script_args"])

        self.assertNotIn("options", setup_kwargs)
        self.assertFalse(
            any(argument.startswith("--compiler-path=") for argument in script_args)
        )

    def test_passes_explicit_compiler_path_as_script_argument(self) -> None:
        compiler_path = "/opt/clang/bin/clang"
        setup_kwargs = self._setup_kwargs(
            ExtensionBuildConfig(compiler_path=compiler_path)
        )
        script_args = cast(list[str], setup_kwargs["script_args"])

        self.assertNotIn("options", setup_kwargs)
        self.assertEqual(f"--compiler-path={compiler_path}", script_args[-1])


class GetTorchIncludeDirsTest(unittest.TestCase):
    def test_appends_explicit_include_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            standard_include = os.path.join(tmpdir, "standard")
            include_root = os.path.join(tmpdir, "bundled")
            api_include = os.path.join(include_root, "torch", "csrc", "api", "include")
            aten_src = os.path.join(include_root, "aten", "src")
            os.makedirs(standard_include)
            os.makedirs(api_include)
            os.makedirs(aten_src)
            builder = nvidia_extension_builder.NvidiaExtensionBuilder(
                source_dir="/tmp",
                kernel_name="_test_kernel",
                output_dir="/tmp",
                build_config=ExtensionBuildConfig(
                    gpu_toolkit_path="/tmp",
                    extra_torch_include_dirs=(include_root, aten_src),
                ),
            )
            with patch.object(
                extension_builder_base.cpp_extension,
                "include_paths",
                return_value=[standard_include],
            ) as include_paths:
                result = builder.get_torch_include_dirs()

        include_paths.assert_called_once_with("cuda")
        self.assertEqual(
            [standard_include, include_root, api_include, aten_src], result
        )

    def test_skips_missing_include_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            standard_include = os.path.join(tmpdir, "standard")
            include_root = os.path.join(tmpdir, "include_root")
            missing_dir = os.path.join(tmpdir, "missing")
            os.makedirs(standard_include)
            os.makedirs(include_root)
            builder = nvidia_extension_builder.NvidiaExtensionBuilder(
                source_dir="/tmp",
                kernel_name="_test_kernel",
                output_dir="/tmp",
                build_config=ExtensionBuildConfig(
                    gpu_toolkit_path="/tmp",
                    extra_torch_include_dirs=(include_root, missing_dir),
                ),
            )
            with patch.object(
                extension_builder_base.cpp_extension,
                "include_paths",
                return_value=[standard_include, missing_dir],
            ):
                result = builder.get_torch_include_dirs()

        self.assertEqual([standard_include, include_root], result)


class GetGpuLibraryDirsTest(unittest.TestCase):
    def test_appends_explicit_gpu_library_dirs_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = os.path.join(tmpdir, "runtime")
            stubs_dir = os.path.join(tmpdir, "stubs")
            os.makedirs(runtime_dir)
            os.makedirs(stubs_dir)
            builder = nvidia_extension_builder.NvidiaExtensionBuilder(
                source_dir="/tmp",
                kernel_name="_k",
                build_config=ExtensionBuildConfig(
                    gpu_toolkit_path=tmpdir,
                    extra_gpu_library_dirs=(runtime_dir, stubs_dir),
                ),
            )

            result = builder.get_gpu_library_dirs()

        self.assertEqual([runtime_dir, stubs_dir], result)

    def test_uses_standard_cuda_library_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            standard_dir = os.path.join(tmpdir, "lib64", "stubs")
            standard_parent_dir = os.path.join(tmpdir, "lib64")
            os.makedirs(standard_dir)
            builder = nvidia_extension_builder.NvidiaExtensionBuilder(
                source_dir="/tmp",
                kernel_name="_k",
                build_config=ExtensionBuildConfig(gpu_toolkit_path=tmpdir),
            )

            result = builder.get_gpu_library_dirs()

        self.assertEqual([standard_dir, standard_parent_dir], result)

    def test_skips_missing_explicit_gpu_library_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_dir = os.path.join(tmpdir, "missing")
            standard_dir = os.path.join(tmpdir, "lib64")
            os.makedirs(standard_dir)
            builder = nvidia_extension_builder.NvidiaExtensionBuilder(
                source_dir="/tmp",
                kernel_name="_k",
                build_config=ExtensionBuildConfig(
                    gpu_toolkit_path=tmpdir, extra_gpu_library_dirs=(missing_dir,)
                ),
            )

            result = builder.get_gpu_library_dirs()

        self.assertEqual([standard_dir], result)
