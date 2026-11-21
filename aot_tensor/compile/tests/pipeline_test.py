# Copyright (c) Meta Platforms, Inc. and affiliates.


import dataclasses
import unittest
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from aot_tensor.compile.triton.pipeline import compile_to_cpp, spec_gen
from aot_tensor.compile.triton.spec_processing import AutotuneAttrs, KernelSpec, OpsUnit
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
        self,
        metadata_shared: int,
        compile_options: dict[str, bool] | None = None,
        *,
        maxnreg: int = 0,
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
                spec=dataclasses.replace(
                    self._make_spec(), autotune=AutotuneAttrs(maxnreg=maxnreg)
                ),
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

    @parameterized.expand([("unset", 0, None), ("limited", 96, 96)])
    def test_register_budget_reaches_compiler(
        self, _name: str, budget: int, expected: int | None
    ) -> None:
        *_, compiled = self._run_spec_gen(0, maxnreg=budget)
        options = compiled.call_args.kwargs["options"]
        self.assertEqual(options.get("maxnreg"), expected)
        if expected is None:
            self.assertNotIn("maxnreg", options)

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


class CompileToCppSelectorGuardAliasesTest(unittest.TestCase):
    """``selector_guard_aliases`` sees the post-filter specs and reaches the selector."""

    def _run(self, provider: MagicMock | None) -> tuple[OpsUnit, MagicMock]:
        spec = KernelSpec(
            signature={0: "*fp32"},
            constants={},
            divisible_by_16=set(),
            divisible_by_8=set(),
        )
        kept = dataclasses.replace(spec, autotune=AutotuneAttrs(num_warps=8))
        fields = AutotuneAttrs.fields_for("hip")
        unit = OpsUnit(
            cc="gfx950",
            optional=set(),
            pointer_args={0},
            scalar_dtypes={},
            constant_types={},
            specs=[spec, kept],
            autotune_fields=fields,
        )
        retained = dataclasses.replace(unit, specs=[kept])
        jit_fn = MagicMock(arg_names=["x"], __module__="m", __name__="_k")
        mod = "aot_tensor.compile.triton.pipeline"
        with (
            patch(f"{mod}.OpsUnit.from_raw_specs", return_value=unit),
            patch(f"{mod}.unwrap_to_jit", return_value=jit_fn),
            patch(f"{mod}.build_arg_descriptors", return_value=[]),
            patch(f"{mod}.generate_header_content", return_value=""),
            patch(f"{mod}.validate_unique_kernel_names"),
            patch(f"{mod}.compile_specs_parallel", return_value=[]),
            patch.object(OpsUnit, "drop_oversize_specs", return_value=(retained, [])),
            patch(f"{mod}.generate_kernel_cpp_content", return_value="") as cpp,
            patch(f"{mod}.generate_torch_op_content", return_value=""),
            patch(f"{mod}.gen_torch_op_schema", return_value={}),
            patch(f"{mod}.gen_tuner_meta_py", return_value=""),
            patch("builtins.open"),
        ):
            compile_to_cpp(
                jit_fn,
                [],
                "/nonexistent",
                "_k",
                gpu_target=GPUTarget("hip", "gfx950", 64),
                import_module=MagicMock(),
                op_namespace="ns",
                selector_guard_aliases=provider,
            )
        return retained, cpp

    def test_provider_sees_retained_specs_and_field_defaults(self) -> None:
        provider = MagicMock(return_value={"waves_per_eu": [7]})
        retained, cpp = self._run(provider)
        provider.assert_called_once()
        meta_func, specs, field_defaults, backend = provider.call_args.args
        self.assertEqual(list(specs), retained.specs)
        self.assertEqual(
            field_defaults,
            {f.name: f.default for f in AutotuneAttrs.fields_for("hip")},
        )
        self.assertEqual(backend, "hip")
        self.assertEqual(cpp.call_args.kwargs["guard_aliases"], {"waves_per_eu": [7]})

    def test_no_provider_passes_no_aliases(self) -> None:
        _, cpp = self._run(None)
        self.assertIsNone(cpp.call_args.kwargs["guard_aliases"])
