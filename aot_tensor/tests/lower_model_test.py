# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""Behavioral specification for the public ``lower_model()`` API.

The API must expose only public entry points, return the lowered module with
portable artifact paths and compile metadata, optionally validate the lowered
module, and reject unsafe work directories and weight changes.

These tests replace Triton compilation with a synthetic session so this
orchestration contract can be tested without a GPU.
"""

from __future__ import annotations

import importlib
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from aot_tensor.api.lowering import lower_model, LoweringArtifact, LoweringOptions
from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.compile.adapter_base import CompileContext, DslCompileConfig
from aot_tensor.compile.triton.adapter import TritonBuildMetadata, TritonCompileConfig
from torch.fx import GraphModule
from triton.backends.compiler import GPUTarget  # @manual=//triton:triton


class _ToyModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class _OtherDslCompileConfig(DslCompileConfig):
    pass


def _make_toy_graph_module() -> GraphModule:
    return torch.fx.symbolic_trace(_ToyModule())


class LowerModelTest(unittest.TestCase):
    def _make_session(
        self,
        package_importer: object | None = None,
        dsl_config: Sequence[DslCompileConfig] | None = None,
        extension_build_config: ExtensionBuildConfig | None = None,
        spec_collector_overrides: object | None = None,
        compile_path: str | None = None,
    ) -> MagicMock:
        del package_importer, spec_collector_overrides
        assert compile_path is not None
        assert dsl_config is not None
        work_dir = Path(compile_path)
        triton_config = next(
            config for config in dsl_config if isinstance(config, TritonCompileConfig)
        )
        kernel_dir = work_dir / "triton_test__kernel"
        shared_library = kernel_dir / "kernel.so"
        schema = kernel_dir / "aott_op_schemas.json"

        context = CompileContext(
            compile_path=compile_path,
            import_module=importlib.import_module,
            dsl_config=dsl_config,
            extension_build_config=extension_build_config,
        )
        context.record_artifact("shared_library", str(shared_library))
        context.record_artifact("operator_schema", str(schema))
        context.record_dsl_metadata(
            "triton",
            TritonBuildMetadata(
                gpu_target=GPUTarget("cuda", 90, 32),
                op_namespace=triton_config.op_namespace,
            ),
        )

        session = MagicMock()
        session.__enter__.return_value = None

        def finish_session(
            exc_type: type[BaseException] | None,
            _exc_value: BaseException | None,
            _traceback: object | None,
        ) -> None:
            if exc_type is not None:
                return
            kernel_dir.mkdir()
            shared_library.write_bytes(b"test")
            schema.write_text("[]\n")

        session.__exit__.side_effect = finish_session
        session.compile_context = context
        return session

    def test_returns_lowered_module_and_artifacts(self) -> None:
        module = _make_toy_graph_module()
        forward_hook = MagicMock(return_value=None)
        module.register_forward_hook(forward_hook)
        example = torch.tensor([2.0])
        build_config = ExtensionBuildConfig(compiler_path="/usr/bin/clang")
        dsl_configs = (TritonCompileConfig(op_namespace="test_namespace"),)
        options = LoweringOptions(
            dsl_configs=dsl_configs,
            extension_build_config=build_config,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            work_dir = Path(temporary_dir) / "work"
            with (
                patch(
                    "aot_tensor.api.lowering.AOTTCompileSession",
                    side_effect=self._make_session,
                ) as session_factory,
                patch(
                    "aot_tensor.api.lowering.transform_kernels",
                    side_effect=lambda graph_module: graph_module,
                ),
            ):
                result = lower_model(
                    module,
                    [(example,)],
                    work_dir=work_dir,
                    options=options,
                )

        self.assertIs(result.module, module)
        self.assertEqual(result.work_dir, work_dir.resolve())
        self.assertEqual(
            result.artifacts,
            (
                LoweringArtifact(
                    dsl_name="triton",
                    kind="shared_library",
                    path=Path("triton_test__kernel/kernel.so"),
                ),
                LoweringArtifact(
                    dsl_name="triton",
                    kind="operator_schema",
                    path=Path("triton_test__kernel/aott_op_schemas.json"),
                ),
            ),
        )
        self.assertEqual(result.gpu_target, "cuda:sm90")
        self.assertEqual(result.op_namespace, "test_namespace")
        self.assertEqual(
            session_factory.call_args.kwargs["dsl_config"],
            dsl_configs,
        )
        self.assertIs(
            session_factory.call_args.kwargs["extension_build_config"],
            build_config,
        )
        self.assertEqual(forward_hook.call_count, 2)

    def test_rejects_unsupported_dsl_configs(self) -> None:
        module = _make_toy_graph_module()

        with tempfile.TemporaryDirectory() as temporary_dir:
            work_dir = Path(temporary_dir) / "work"
            unsupported_dsl_configs: tuple[tuple[DslCompileConfig, ...], ...] = (
                (),
                (_OtherDslCompileConfig(),),
                (TritonCompileConfig(), _OtherDslCompileConfig()),
            )
            for dsl_configs in unsupported_dsl_configs:
                with self.subTest(dsl_configs=dsl_configs):
                    with self.assertRaisesRegex(
                        ValueError,
                        "requires exactly one TritonCompileConfig",
                    ):
                        lower_model(
                            module,
                            [(torch.tensor([2.0]),)],
                            work_dir=work_dir,
                            options=LoweringOptions(dsl_configs=dsl_configs),
                        )

            self.assertFalse(work_dir.exists())

    def test_validate_false_skips_lowered_forward(self) -> None:
        module = _make_toy_graph_module()
        forward_hook = MagicMock(return_value=None)
        module.register_forward_hook(forward_hook)

        with tempfile.TemporaryDirectory() as temporary_dir:
            with (
                patch(
                    "aot_tensor.api.lowering.AOTTCompileSession",
                    side_effect=self._make_session,
                ),
                patch(
                    "aot_tensor.api.lowering.transform_kernels",
                    side_effect=lambda graph_module: graph_module,
                ),
            ):
                lower_model(
                    module,
                    [(torch.tensor([2.0]),)],
                    work_dir=Path(temporary_dir) / "work",
                    options=LoweringOptions(validate=False),
                )

        self.assertEqual(forward_hook.call_count, 1)

    def test_rejects_nonempty_work_dir(self) -> None:
        module = _make_toy_graph_module()

        with tempfile.TemporaryDirectory() as temporary_dir:
            work_dir = Path(temporary_dir)
            (work_dir / "stale-output").write_text("stale")

            with self.assertRaisesRegex(ValueError, "work_dir must be empty"):
                lower_model(module, [(torch.tensor([2.0]),)], work_dir=work_dir)

    def test_detects_in_place_weight_signature_change(self) -> None:
        module = torch.fx.symbolic_trace(torch.nn.Linear(2, 2))

        def change_weight_shape(graph_module: GraphModule) -> GraphModule:
            graph_module.weight = torch.nn.Parameter(torch.ones(3, 2))
            return graph_module

        with tempfile.TemporaryDirectory() as temporary_dir:
            with (
                patch(
                    "aot_tensor.api.lowering.AOTTCompileSession",
                    side_effect=self._make_session,
                ),
                patch(
                    "aot_tensor.api.lowering.transform_kernels",
                    side_effect=change_weight_shape,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "changed weight 'weight'"):
                    lower_model(
                        module,
                        [(torch.ones(1, 2),)],
                        work_dir=Path(temporary_dir) / "work",
                    )
