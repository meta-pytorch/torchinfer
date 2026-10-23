# Copyright (c) Meta Platforms, Inc. and affiliates.


import dataclasses
import unittest
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from aot_tensor.compile.triton.pipeline import spec_gen
from aot_tensor.compile.triton.spec_processing import AutotuneAttrs, KernelSpec
from aot_tensor.types import TritonAOT
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

    def _run_spec_gen(
        self, metadata_shared: int, compile_options: dict[str, bool] | None = None
    ) -> tuple[str, int, MagicMock]:
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
            ) as compile_mock,
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
            code, shared = spec_gen(
                install_dir="/tmp/spec_gen_test",
                spec=self._make_spec(),
                module="fake.module",
                name="_fake_kernel",
                gpu_target=GPUTarget("cuda", 90, 32),
                import_module=fake_import,
                descriptors=[],
                compile_options=compile_options,
            )
        return code, shared, compile_mock

    @parameterized.expand([("under_cap", 200000), ("oversize", 266240)])
    def test_returns_code_and_shared(self, _name: str, metadata_shared: int) -> None:
        code, shared, _ = self._run_spec_gen(metadata_shared=metadata_shared)
        self.assertIsInstance(code, str)
        self.assertIn("cubin", code)
        self.assertIn("loader", code)
        self.assertIn("launcher", code)
        self.assertEqual(shared, metadata_shared)

    def test_compile_options_reach_the_compile(self) -> None:
        options = {"enable_fp_fusion": False, "enable_reflect_ftz": False}
        *_, options_compile = self._run_spec_gen(200000, compile_options=options)
        *_, plain_compile = self._run_spec_gen(200000)

        plain = plain_compile.call_args.kwargs["options"]
        self.assertFalse(options.keys() & plain.keys())
        self.assertEqual(
            options_compile.call_args.kwargs["options"], {**plain, **options}
        )

    @parameterized.expand([(f.name,) for f in dataclasses.fields(AutotuneAttrs)])
    def test_marker_cannot_set_an_autotune_field(self, field: str) -> None:
        # The kernel name, the selector guards and the tuner meta read these
        # from the spec, so the cubin must be compiled with the spec's value.
        default = getattr(AutotuneAttrs(), field)
        with self.assertRaisesRegex(RuntimeError, rf"cannot set \['{field}'\]"):
            TritonAOT(self._patched_jit_fn(), {}, compile_options={field: default})

    @parameterized.expand(
        [
            ("cuda", GPUTarget("cuda", 90, 32), None),
            ("cuda_options", GPUTarget("cuda", 90, 32), {"enable_fp_fusion": False}),
            ("hip", GPUTarget("hip", "gfx942", 64), None),
            (
                "hip_options",
                GPUTarget("hip", "gfx942", 64),
                {"enable_fp_fusion": False},
            ),
        ]
    )
    def test_name_and_legacy_launch_follow_the_spec_num_warps(
        self,
        _name: str,
        gpu_target: GPUTarget,
        compile_options: dict[str, bool] | None,
    ) -> None:
        jit_fn = self._patched_jit_fn()
        spec = dataclasses.replace(
            self._make_spec(), autotune=AutotuneAttrs(num_warps=8)
        )

        def compile_with(
            *_args: object, options: dict[str, object], **_kwargs: object
        ) -> SimpleNamespace:
            # No launcher_src, so gen_launcher takes the legacy path.
            metadata = SimpleNamespace(
                name="_fake_kernel", shared=0, num_warps=options["num_warps"]
            )
            return SimpleNamespace(metadata=metadata, asm={})

        def fake_import(_mod: str) -> ModuleType:
            return cast(ModuleType, MagicMock(_fake_kernel=jit_fn))

        with (
            patch(
                "aot_tensor.compile.triton.pipeline.triton.compiler.compile",
                side_effect=compile_with,
            ) as compile_mock,
            patch(
                "aot_tensor.compile.triton.pipeline.unwrap_to_jit", return_value=jit_fn
            ),
            patch(
                "aot_tensor.compile.triton.pipeline.gen_compile_arg",
                return_value=(None,),
            ),
            patch(
                "aot_tensor.compile.triton.pipeline.gen_cubin",
                return_value="// cubin\n",
            ),
        ):
            code, _ = spec_gen(
                install_dir="/tmp/spec_gen_test",
                spec=spec,
                module="fake.module",
                name="_fake_kernel",
                gpu_target=gpu_target,
                import_module=fake_import,
                descriptors=[],
                compile_options=compile_options,
            )

        options = compile_mock.call_args.kwargs["options"]
        self.assertEqual(options["num_warps"], 8)
        self.assertEqual(
            options.get("enable_fp_fusion"),
            None if compile_options is None else compile_options["enable_fp_fusion"],
        )
        self.assertIn("_w8_", code)
        self.assertIn(f"grid.z, {gpu_target.warp_size} * 8, 1, 1,", code)
