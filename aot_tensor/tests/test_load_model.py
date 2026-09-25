# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""Behavioral specification for the public ``load_model()`` API."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, MagicMock, patch

import torch
from aot_tensor.api.exporting import export_model
from aot_tensor.api.loading import load_model
from aot_tensor.api.lowering import LoweringResult


class _ToyModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class LoadModelTest(unittest.TestCase):
    def _write_manifest(self, root: Path, libraries: list[str]) -> None:
        (root / "manifest.json").write_text(
            json.dumps({"shared_libraries": libraries}) + "\n",
            encoding="utf-8",
        )

    def test_loads_exported_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            work_dir = Path(temporary_dir)
            module = torch.fx.symbolic_trace(_ToyModule())
            for node in module.graph.nodes:
                node.type = None
            module.recompile()
            lowering = LoweringResult(
                module=module,
                work_dir=work_dir,
                artifacts=(),
                gpu_target="cuda:sm90",
                torch_target_version="0x020C000000000000",
                op_namespace="aot_tensor",
            )

            loaded = load_model(export_model(lowering))

        torch.testing.assert_close(loaded(torch.tensor([2.0])), torch.tensor([3.0]))

    def test_loads_missing_libraries_before_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_path = root / "model.pt"
            model_path.write_bytes(b"model")
            library_paths = (root / "one.so", root / "two.so")
            for library in library_paths:
                library.write_bytes(b"library")
            self._write_manifest(root, [library.name for library in library_paths])
            loaded_module = MagicMock()

            with (
                patch("aot_tensor.api.loading.torch.ops.load_library") as load_library,
                patch(
                    "aot_tensor.api.loading.torch.jit.load",
                    return_value=loaded_module,
                ) as jit_load,
            ):
                calls = MagicMock()
                calls.attach_mock(load_library, "load_library")
                calls.attach_mock(jit_load, "jit_load")

                result = load_model(model_path)

        self.assertIs(result, loaded_module)
        self.assertEqual(
            calls.mock_calls,
            [call.load_library(str(library.resolve())) for library in library_paths]
            + [call.jit_load(str(model_path.resolve()))],
        )

    def test_skips_library_already_loaded_from_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_path = root / "model.pt"
            model_path.write_bytes(b"model")
            library = (root / "kernel.so").resolve()
            library.write_bytes(b"library")
            self._write_manifest(root, [library.name])
            torch.ops.loaded_libraries.add(str(library))
            try:
                with (
                    patch(
                        "aot_tensor.api.loading.torch.ops.load_library"
                    ) as load_library,
                    patch("aot_tensor.api.loading.torch.jit.load") as jit_load,
                ):
                    load_model(model_path)
            finally:
                torch.ops.loaded_libraries.discard(str(library))

        load_library.assert_not_called()
        jit_load.assert_called_once_with(str(model_path.resolve()))
