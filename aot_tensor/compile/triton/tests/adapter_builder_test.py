# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import os
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.compile.adapter_base import CompileContext, KernelSpecs
from aot_tensor.compile.triton import adapter
from triton.backends.compiler import GPUTarget


class TritonExtensionBuildConfigTest(unittest.TestCase):
    def _compile(
        self,
        *,
        config: adapter.TritonCompileConfig,
        session_build_config: ExtensionBuildConfig | None = None,
    ) -> tuple[
        list[tuple[str, str, str, ExtensionBuildConfig | None]],
        str,
    ]:
        calls: list[tuple[str, str, str, ExtensionBuildConfig | None]] = []

        def build_extension(
            *,
            source_dir: str,
            kernel_name: str,
            output_dir: str,
            build_config: ExtensionBuildConfig | None = None,
        ) -> str:
            calls.append((source_dir, kernel_name, output_dir, build_config))
            return os.path.join(output_dir, "test_kernel.so")

        def test_kernel() -> None:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            context = CompileContext(
                tmpdir,
                lambda name: ModuleType(name),
                [config],
                extension_build_config=session_build_config,
            )
            with (
                patch.object(
                    adapter,
                    "get_kernel_specs",
                    return_value={object(): KernelSpecs()},
                ),
                patch.object(adapter, "unwrap_to_jit", return_value=test_kernel),
                patch.object(adapter, "_resolve_autotune_cache"),
                patch.object(adapter, "compile_to_cpp"),
                patch.object(adapter, "build_triton_aot_extension", build_extension),
            ):
                adapter.TritonAdapter().compile_and_build(context)

            expected_dir = os.path.join(
                tmpdir, f"{__name__.rsplit('.', 1)[-1]}_test_kernel"
            )
            return calls, expected_dir

    def test_compile_uses_session_extension_build_config(self) -> None:
        session_build_config = ExtensionBuildConfig(compiler_path="/opt/session/clang")

        calls, expected_dir = self._compile(
            config=adapter.TritonCompileConfig(gpu_target=GPUTarget("cuda", 80, 32)),
            session_build_config=session_build_config,
        )

        self.assertEqual(
            [(expected_dir, "test_kernel", expected_dir, session_build_config)],
            calls,
        )
