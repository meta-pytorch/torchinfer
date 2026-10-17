# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import os
import tempfile
import unittest
from unittest import mock

from aot_tensor.build import extension_builder_base
from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.build.flydsl import extension_builder
from aot_tensor.build.flydsl.extension_builder import (
    build_flydsl_aot_extension,
    FlydslExtensionBuilder,
)
from parameterized import parameterized

_KERNEL = "_flydsl_vec_launch"


def _touch(path: str) -> str:
    with open(path, "w") as f:
        f.write("")
    return path


class FlydslExtensionBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.toolkit: str = self.enterContext(tempfile.TemporaryDirectory())
        os.makedirs(os.path.join(self.toolkit, "include", "hip"))
        _touch(os.path.join(self.toolkit, "include", "hip", "hip_runtime.h"))
        os.makedirs(os.path.join(self.toolkit, "lib"))
        self.kernel_dir: str = self.enterContext(tempfile.TemporaryDirectory())
        _touch(os.path.join(self.kernel_dir, f"{_KERNEL}_torch_op.cpp"))
        self.objects: list[str] = [
            _touch(os.path.join(self.kernel_dir, f"{_KERNEL}__{tag}.o"))
            for tag in ("aaaa", "bbbb")
        ]

    def _builder(self, objects: list[str] | None = None) -> FlydslExtensionBuilder:
        return FlydslExtensionBuilder(
            self.kernel_dir,
            _KERNEL,
            self.kernel_dir,
            self.objects if objects is None else objects,
            build_config=ExtensionBuildConfig(gpu_toolkit_path=self.toolkit),
        )

    def test_links_objects_into_a_self_contained_so_exporting_nothing(self) -> None:
        def fake_setup(**_kwargs: object) -> None:
            _touch(os.path.join(self.kernel_dir, "flydsl_vec_launch.so"))

        with (
            mock.patch.object(
                extension_builder_base.cpp_extension, "include_paths", return_value=[]
            ),
            mock.patch.object(
                extension_builder, "setup", side_effect=fake_setup
            ) as setup,
        ):
            result = self._builder().build()

        self.assertEqual(result, os.path.join(self.kernel_dir, "flydsl_vec_launch.so"))
        (extension,) = setup.call_args.kwargs["ext_modules"]
        self.assertEqual(
            extension.sources,
            [os.path.join(self.kernel_dir, f"{_KERNEL}_torch_op.cpp")],
        )
        self.assertEqual(extension.extra_objects, self.objects)
        (link_arg,) = extension.extra_link_args
        self.assertTrue(link_arg.startswith("-Wl,--version-script="))
        with open(link_arg.split("=", 1)[1]) as fp:
            self.assertEqual(fp.read().split(), ["{", "local:", "*;", "};"])
        self.assertIn("-O3", extension.extra_compile_args)
        self.assertIn("amdhip64", extension.libraries)

    @parameterized.expand(
        [
            ("missing_input", ["/no/such.o"], FileNotFoundError, r"such\.o"),
            ("no_objects", [], ValueError, "at least one launcher object"),
        ]
    )
    def test_rejects_bad_objects(
        self,
        _name: str,
        objects: list[str],
        error: type[Exception],
        message: str,
    ) -> None:
        with self.assertRaisesRegex(error, message):
            self._builder(objects).build()

    def test_nvidia_is_rejected(self) -> None:
        with mock.patch.object(extension_builder, "is_amd", return_value=False):
            with self.assertRaisesRegex(NotImplementedError, "AMD ROCm"):
                build_flydsl_aot_extension(
                    self.kernel_dir, _KERNEL, self.kernel_dir, self.objects
                )
