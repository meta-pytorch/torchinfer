# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""Behavioral specification for the public ``export_model()`` API.

Export converts the lowered FX module to TorchScript, optionally validates it,
writes ``model.pt`` and a library-path manifest into ``work_dir``, and removes
partial output after a failure.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import torch
from aot_tensor.api.exporting import export_model, ExportOptions
from aot_tensor.api.lowering import LoweringArtifact, LoweringResult
from torch.fx import GraphModule


_TEST_NAMESPACE = "aot_tensor_export_test"
_KERNEL_SCHEMA = "kernel(int[] grid, Tensor(a!)? value) -> ()"


def _kernel_impl(_grid: list[int], _value: torch.Tensor | None) -> None:
    pass


class _CompiledOpModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        torch.ops.aot_tensor_export_test.kernel([1, 1, 1], value)
        return value + 1


class _FailingValidationModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        torch.ops.aot_tensor_export_test.kernel([1, 1, 1], value)
        return value[0]


def _trace_for_torchscript(module: torch.nn.Module) -> GraphModule:
    graph_module = torch.fx.symbolic_trace(module)
    for node in graph_module.graph.nodes:
        node.type = None
    graph_module.recompile()
    return graph_module


class ExportModelTest(unittest.TestCase):
    _libraries: ClassVar[list[torch.library.Library]] = []

    @classmethod
    def setUpClass(cls) -> None:
        definition = torch.library.Library(_TEST_NAMESPACE, "DEF")
        definition.define(_KERNEL_SCHEMA)
        implementation = torch.library.Library(_TEST_NAMESPACE, "IMPL", "CPU")
        implementation.impl("kernel", _kernel_impl)
        cls._libraries.extend((definition, implementation))

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._temporary_directory.name) / "work"
        self.work_dir.mkdir()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _write_lowering_fixture(self, module: GraphModule) -> LoweringResult:
        kernel_dir = self.work_dir / "kernel"
        kernel_dir.mkdir()
        library = kernel_dir / "kernel.so"
        schema = kernel_dir / "aott_op_schemas.json"
        library.write_bytes(b"test library")
        schema.write_text(json.dumps([_KERNEL_SCHEMA]) + "\n", encoding="utf-8")
        return LoweringResult(
            module=module,
            work_dir=self.work_dir.resolve(),
            artifacts=(
                LoweringArtifact(
                    dsl_name="triton",
                    kind="shared_library",
                    path=library.relative_to(self.work_dir),
                ),
                LoweringArtifact(
                    dsl_name="triton",
                    kind="operator_schema",
                    path=schema.relative_to(self.work_dir),
                ),
            ),
            gpu_target="cuda:sm90",
            torch_target_version="0x020C000000000000",
            op_namespace=_TEST_NAMESPACE,
        )

    def test_exports_torchscript_model_and_manifest(self) -> None:
        lowering = self._write_lowering_fixture(
            _trace_for_torchscript(_CompiledOpModule())
        )

        model_path = export_model(
            lowering,
            options=ExportOptions(validation_inputs=[(torch.tensor([2.0]),)]),
        )

        self.assertEqual(model_path, self.work_dir.resolve() / "model.pt")
        self.assertTrue(model_path.is_file())
        self.assertEqual(
            json.loads((self.work_dir / "manifest.json").read_text(encoding="utf-8")),
            {"shared_libraries": ["kernel/kernel.so"]},
        )
        loaded = torch.jit.load(str(model_path))
        torch.testing.assert_close(loaded(torch.tensor([4.0])), torch.tensor([5.0]))

    def test_validation_failure_writes_no_output(self) -> None:
        lowering = self._write_lowering_fixture(
            _trace_for_torchscript(_FailingValidationModule())
        )

        with self.assertRaises(RuntimeError):
            export_model(
                lowering,
                options=ExportOptions(validation_inputs=[(torch.empty(0),)]),
            )

        self.assertFalse((self.work_dir / "model.pt").exists())
        self.assertFalse((self.work_dir / "manifest.json").exists())

    def test_rejects_pt2_format_until_supported(self) -> None:
        lowering = self._write_lowering_fixture(
            _trace_for_torchscript(_CompiledOpModule())
        )

        with self.assertRaisesRegex(NotImplementedError, "PT2 export"):
            export_model(lowering, options=ExportOptions(pt2_format=True))

        self.assertFalse((self.work_dir / "model.pt").exists())
        self.assertFalse((self.work_dir / "manifest.json").exists())

    def test_save_failure_removes_partial_output(self) -> None:
        lowering = self._write_lowering_fixture(
            _trace_for_torchscript(_CompiledOpModule())
        )
        model_path = self.work_dir / "model.pt"

        def fail_save(_module: torch.jit.ScriptModule, path: str) -> None:
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

        with (
            patch(
                "aot_tensor.api.exporting.torch.jit.save",
                side_effect=fail_save,
            ),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            export_model(lowering)

        self.assertFalse(model_path.exists())
        self.assertFalse((self.work_dir / "manifest.json").exists())

    def test_rejects_existing_output(self) -> None:
        lowering = self._write_lowering_fixture(
            _trace_for_torchscript(_CompiledOpModule())
        )
        model_path = self.work_dir / "model.pt"
        model_path.write_bytes(b"existing")

        with self.assertRaises(FileExistsError):
            export_model(lowering)

        self.assertEqual(model_path.read_bytes(), b"existing")
