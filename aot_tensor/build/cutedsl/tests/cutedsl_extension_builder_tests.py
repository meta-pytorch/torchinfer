# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# pyre-strict

"""CPU-only tests for the CuTeDSL AOT extension builder.

Fixtures are snapshotted intermediate artifacts under ``resources/``. Setuptools
is mocked so the tests can validate extension construction without a CUDA
toolkit or GPU.
"""

import functools
import os
import tempfile
import unittest
from unittest import mock

from aot_tensor.build import extension_builder_base
from aot_tensor.build.cutedsl import extension_builder
from aot_tensor.build.cutedsl.extension_builder import CutedslExtensionBuilder
from aot_tensor.build.extension_build_config import ExtensionBuildConfig

_RESOURCE_NAME: str = "cutedsl_VectorAddKernel"
_KERNEL_NAME: str = "_cutedsl_VectorAddKernel"


class CutedslExtensionBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_dir = os.path.join(
            os.path.dirname(__file__),
            "resources",
            _RESOURCE_NAME,
        )

    def _fake_toolkit(self, path: str, *, supports: bool) -> None:
        include = os.path.join(path, "include")
        os.makedirs(include, exist_ok=True)
        with open(os.path.join(include, "cuda.h"), "w") as f:
            f.write("// cuda")
        marker = "cudaLibraryLoadData" if supports else "// no library api"
        with open(os.path.join(include, "cuda_runtime_api.h"), "w") as f:
            f.write(marker)
        os.makedirs(os.path.join(path, "lib64"), exist_ok=True)

    def _write_build_outputs(
        self,
        output_dir: str,
        extension_names: tuple[str, ...],
        **_setup_kwargs: object,
    ) -> None:
        for extension_name in extension_names:
            with open(os.path.join(output_dir, f"{extension_name}.so"), "w") as output:
                output.write("")

    def test_build_configures_so_and_sidecar(self) -> None:
        with (
            tempfile.TemporaryDirectory() as toolkit,
            tempfile.TemporaryDirectory() as output_dir,
            tempfile.TemporaryDirectory() as torch_include_dir,
        ):
            self._fake_toolkit(toolkit, supports=True)
            builder = CutedslExtensionBuilder(
                source_dir=self.source_dir,
                kernel_name=_KERNEL_NAME,
                output_dir=output_dir,
                build_config=ExtensionBuildConfig(
                    compiler_path="/compiler",
                    gpu_toolkit_path=toolkit,
                ),
            )
            ext_name = _KERNEL_NAME.lstrip("_")
            output = os.path.join(output_dir, f"{ext_name}.so")
            with (
                mock.patch.object(
                    extension_builder_base.cpp_extension,
                    "include_paths",
                    return_value=[torch_include_dir],
                ),
                mock.patch.object(
                    extension_builder,
                    "setup",
                    side_effect=functools.partial(
                        self._write_build_outputs,
                        output_dir,
                        (ext_name, f"{ext_name}_cutedsl_impl"),
                    ),
                ) as setup,
            ):
                result = builder.build()

            self.assertEqual(result, output)
            self.assertEqual(builder.gpu_toolkit_path, toolkit)
            setup.assert_called_once()
            setup_kwargs = setup.call_args.kwargs
            self.assertEqual(
                setup_kwargs["script_args"],
                [
                    "build_ext",
                    f"--build-lib={output_dir}",
                    f"--build-temp={output_dir}/build_temp",
                    "--compiler-path=/compiler",
                ],
            )
            extensions = setup_kwargs["ext_modules"]
            self.assertEqual(
                [extension.name for extension in extensions],
                [f"{ext_name}_cutedsl_impl", ext_name],
            )
            self.assertEqual(
                extensions[0].extra_objects,
                [os.path.join(self.source_dir, f"{_KERNEL_NAME}.o")],
            )
            self.assertFalse(extensions[1].extra_objects)
            self.assertEqual(
                extensions[0].include_dirs,
                [self.source_dir, os.path.join(toolkit, "include")],
            )
            self.assertEqual(
                extensions[1].include_dirs,
                [os.path.join(toolkit, "include"), torch_include_dir],
            )

    def test_build_raises_when_generated_source_is_missing(self) -> None:
        artifacts = (
            (f"{_KERNEL_NAME}.h", "CuTe header not found"),
            (f"{_KERNEL_NAME}.o", "CuTe object not found"),
            (f"{_KERNEL_NAME}_entry.cpp", "CuTe entry source not found"),
            (f"{_KERNEL_NAME}_torch_op.cpp", "CuTe torch op source not found"),
        )
        for missing_artifact, expected_error in artifacts:
            with self.subTest(missing_artifact=missing_artifact):
                with tempfile.TemporaryDirectory() as source_dir:
                    for artifact, _ in artifacts:
                        if artifact == missing_artifact:
                            continue
                        with open(os.path.join(source_dir, artifact), "w") as output:
                            output.write("")

                    builder = CutedslExtensionBuilder(
                        source_dir=source_dir,
                        kernel_name=_KERNEL_NAME,
                        build_config=ExtensionBuildConfig(gpu_toolkit_path=source_dir),
                    )
                    with self.assertRaisesRegex(AssertionError, expected_error):
                        builder.build()

    def test_build_raises_when_expected_output_is_missing(self) -> None:
        ext_name = _KERNEL_NAME.lstrip("_")
        cases = (
            ("sidecar", (), "Expected built CuTe sidecar extension"),
            ("torch_op", (f"{ext_name}_cutedsl_impl",), "Expected built extension"),
        )
        for name, produced_extensions, expected_error in cases:
            with self.subTest(name=name):
                with (
                    tempfile.TemporaryDirectory() as toolkit,
                    tempfile.TemporaryDirectory() as output_dir,
                ):
                    self._fake_toolkit(toolkit, supports=True)
                    builder = CutedslExtensionBuilder(
                        source_dir=self.source_dir,
                        kernel_name=_KERNEL_NAME,
                        output_dir=output_dir,
                        build_config=ExtensionBuildConfig(gpu_toolkit_path=toolkit),
                    )
                    with (
                        mock.patch.object(
                            extension_builder_base.cpp_extension,
                            "include_paths",
                            return_value=[],
                        ),
                        mock.patch.object(
                            extension_builder,
                            "setup",
                            side_effect=functools.partial(
                                self._write_build_outputs,
                                output_dir,
                                produced_extensions,
                            ),
                        ),
                    ):
                        with self.assertRaisesRegex(AssertionError, expected_error):
                            builder.build()

    def test_build_selects_supporting_toolkit_fallback(self) -> None:
        with (
            tempfile.TemporaryDirectory() as bad,
            tempfile.TemporaryDirectory() as good,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            self._fake_toolkit(bad, supports=False)
            self._fake_toolkit(good, supports=True)
            builder = CutedslExtensionBuilder(
                source_dir=self.source_dir,
                kernel_name=_KERNEL_NAME,
                output_dir=output_dir,
                build_config=ExtensionBuildConfig(gpu_toolkit_path=bad),
            )
            ext_name = _KERNEL_NAME.lstrip("_")
            output = os.path.join(output_dir, f"{ext_name}.so")
            with (
                mock.patch.dict(os.environ, {"CUDA_HOME": good, "CUDA_PATH": ""}),
                mock.patch.object(
                    extension_builder_base, "DEFAULT_SYSTEM_CUDA_HOME", bad
                ),
                mock.patch.object(
                    extension_builder_base.cpp_extension,
                    "include_paths",
                    return_value=[],
                ),
                mock.patch.object(
                    extension_builder,
                    "setup",
                    side_effect=functools.partial(
                        self._write_build_outputs,
                        output_dir,
                        (ext_name, f"{ext_name}_cutedsl_impl"),
                    ),
                ) as setup,
            ):
                result = builder.build()
            setup.assert_called_once()
            self.assertEqual(result, output)
            self.assertEqual(builder.gpu_toolkit_path, good)

    def test_build_raises_when_toolkits_lack_cuda_library_api(self) -> None:
        with tempfile.TemporaryDirectory() as bad:
            self._fake_toolkit(bad, supports=False)
            builder = CutedslExtensionBuilder(
                source_dir=self.source_dir,
                kernel_name=_KERNEL_NAME,
                build_config=ExtensionBuildConfig(gpu_toolkit_path=bad),
            )
            with (
                mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": ""}),
                mock.patch.object(
                    extension_builder_base, "DEFAULT_SYSTEM_CUDA_HOME", bad
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "does not expose those APIs",
                ):
                    builder.build()


class BuildCutedslAotExtensionTest(unittest.TestCase):
    def test_rejects_amd(self) -> None:
        with (
            mock.patch.object(extension_builder, "is_amd", return_value=True),
            mock.patch.object(
                extension_builder,
                "CutedslExtensionBuilder",
            ) as builder,
        ):
            with self.assertRaisesRegex(
                NotImplementedError,
                "only supported on NVIDIA CUDA",
            ):
                extension_builder.build_cutedsl_aot_extension(
                    source_dir="/source",
                    kernel_name="_kernel",
                    output_dir="/output",
                )

        builder.assert_not_called()

    def test_forwards_build_config(self) -> None:
        build_config = ExtensionBuildConfig(compiler_path="/compiler")
        with (
            mock.patch.object(extension_builder, "is_amd", return_value=False),
            mock.patch.object(
                extension_builder,
                "CutedslExtensionBuilder",
            ) as builder,
        ):
            builder.return_value.build.return_value = "/output/kernel.so"
            output = extension_builder.build_cutedsl_aot_extension(
                source_dir="/source",
                kernel_name="_kernel",
                output_dir="/output",
                build_config=build_config,
            )

        self.assertEqual(output, "/output/kernel.so")
        builder.assert_called_once_with(
            "/source",
            "_kernel",
            "/output",
            build_config=build_config,
        )
        builder.return_value.build.assert_called_once_with()
