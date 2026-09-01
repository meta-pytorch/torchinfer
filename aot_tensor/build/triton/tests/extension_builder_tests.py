# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.build.triton import extension_builder


class BuildTritonAotExtensionTest(unittest.TestCase):
    def test_uses_amd_builder_when_is_amd_true(self) -> None:
        build_config = ExtensionBuildConfig(compiler_path="/opt/clang/bin/clang")
        builder = MagicMock()
        builder.build.return_value = "/tmp/kernel.so"

        with (
            patch.object(extension_builder, "is_amd", return_value=True),
            patch.object(
                extension_builder,
                "AmdExtensionBuilder",
                return_value=builder,
            ) as amd_builder,
            patch.object(extension_builder, "NvidiaExtensionBuilder") as nvidia_builder,
        ):
            result = extension_builder.build_triton_aot_extension(
                source_dir="/tmp/source",
                kernel_name="_kernel",
                output_dir="/tmp/output",
                build_config=build_config,
            )

        amd_builder.assert_called_once_with(
            "/tmp/source",
            "_kernel",
            "/tmp/output",
            build_config=build_config,
        )
        nvidia_builder.assert_not_called()
        builder.build.assert_called_once_with()
        self.assertEqual("/tmp/kernel.so", result)

    def test_uses_nvidia_builder_when_is_amd_false(self) -> None:
        builder = MagicMock()
        builder.build.return_value = "/tmp/kernel.so"

        with (
            patch.object(extension_builder, "is_amd", return_value=False),
            patch.object(extension_builder, "AmdExtensionBuilder") as amd_builder,
            patch.object(
                extension_builder,
                "NvidiaExtensionBuilder",
                return_value=builder,
            ) as nvidia_builder,
        ):
            result = extension_builder.build_triton_aot_extension(
                source_dir="/tmp/source",
                kernel_name="_kernel",
                output_dir="/tmp/output",
            )

        amd_builder.assert_not_called()
        nvidia_builder.assert_called_once_with(
            "/tmp/source",
            "_kernel",
            "/tmp/output",
            build_config=None,
        )
        builder.build.assert_called_once_with()
        self.assertEqual("/tmp/kernel.so", result)
