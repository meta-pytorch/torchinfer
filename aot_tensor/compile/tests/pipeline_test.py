# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest
from types import ModuleType
from typing import cast
from unittest.mock import MagicMock, patch

from aot_tensor.compile.triton.pipeline import spec_gen
from aot_tensor.compile.triton.spec_processing import AutotuneAttrs, KernelSpec
from parameterized import parameterized
from triton.backends.compiler import GPUTarget


class SpecGenTest(unittest.TestCase):
    """``spec_gen`` is pure compile: returns ``(code, metadata.shared)``."""

    def _patched_jit_fn(self) -> MagicMock:
        jit = MagicMock()
        jit.__module__ = "fake.module"
        jit.__name__ = "_fake_kernel"
        jit.arg_names = ["x_ptr", "BLOCK_M", "N_BLOCK"]
        jit.signature.parameters = {
            "x_ptr": MagicMock(),
            "BLOCK_M": MagicMock(),
            "N_BLOCK": MagicMock(),
        }
        return jit

    def _make_spec(self) -> KernelSpec:
        return KernelSpec(
            signature={0: "*fp32"},
            constants={1: 16, 2: 1024},
            divisible_by_16=set(),
            divisible_by_8=set(),
            autotune=AutotuneAttrs(),
        )

    def _run_spec_gen(self, metadata_shared: int) -> tuple[str, int]:
        kernel = MagicMock()
        kernel.metadata.name = "_fake_kernel"
        kernel.metadata.shared = metadata_shared

        jit_fn = self._patched_jit_fn()
        fake_pkg = MagicMock(_fake_kernel=jit_fn)

        def fake_import(_mod: str) -> ModuleType:
            return cast(ModuleType, fake_pkg)

        with (
            patch(
                "aot_tensor.compile.triton.pipeline.triton.compiler.compile",
                return_value=kernel,
            ),
            patch(
                "aot_tensor.compile.triton.pipeline.unwrap_to_jit", return_value=jit_fn
            ),
            patch(
                "aot_tensor.compile.triton.pipeline.gen_compile_arg",
                return_value=(None,),
            ),
            patch(
                "aot_tensor.compile.triton.pipeline.gen_kernel_name",
                return_value="_fake_kernel_cfg",
            ),
            patch(
                "aot_tensor.compile.triton.pipeline.gen_cubin",
                return_value="// cubin\n",
            ),
            patch(
                "aot_tensor.compile.triton.pipeline.gen_loader",
                return_value="// loader\n",
            ),
            patch(
                "aot_tensor.compile.triton.pipeline.gen_launcher",
                return_value="// launcher\n",
            ),
        ):
            return spec_gen(
                install_dir="/tmp/spec_gen_test",
                spec=self._make_spec(),
                module="fake.module",
                name="_fake_kernel",
                gpu_target=GPUTarget("cuda", 90, 32),
                import_module=fake_import,
                descriptors=[],
            )

    @parameterized.expand([("under_cap", 200000), ("oversize", 266240)])
    def test_returns_code_and_shared(self, _name: str, metadata_shared: int) -> None:
        code, shared = self._run_spec_gen(metadata_shared=metadata_shared)
        self.assertIsInstance(code, str)
        self.assertIn("cubin", code)
        self.assertIn("loader", code)
        self.assertIn("launcher", code)
        self.assertEqual(shared, metadata_shared)
