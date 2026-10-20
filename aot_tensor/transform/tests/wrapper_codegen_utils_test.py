# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict


"""Unit tests for the DSL-agnostic wrapper-codegen helpers in
``transform/wrapper_codegen_utils.py`` (torch.package source extraction +
``find_sole_marker_in_globals``) and ``transform/import_utils.py``
(import-header extraction).
"""

import ast
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from aot_tensor.transform.import_utils import get_original_import_header
from aot_tensor.transform.wrapper_codegen_utils import (
    _extract_function_source,
    _get_module_and_source,
    find_sole_marker_in_globals,
    generate_wrapper_files_skeleton,
)
from torch.package import PackageExporter, PackageImporter


class _IdentityTransform(ast.NodeTransformer):
    def generate_so_loading_code(self, node: ast.AST, abs_triton_aot_path: str) -> str:
        return ""


class WrapperSourceOriginTest(unittest.TestCase):
    def test_archived_and_live_wrappers_keep_distinct_helpers(self) -> None:
        helper_name = "aot_tensor.ops.codegen_origin_helper"
        target_name = "aot_tensor.ops.codegen_origin_target"
        source = f"from {helper_name} import OFFSET\n\ndef wrapper(x):\n    return x + OFFSET\n"
        archive = io.BytesIO()
        with PackageExporter(archive) as exporter:
            exporter.save_source_string("aot_tensor", "", is_package=True)
            exporter.save_source_string("aot_tensor.ops", "", is_package=True)
            exporter.save_source_string(helper_name, "OFFSET = 7\n")
            exporter.save_source_string(target_name, source)
        archive.seek(0)
        importer = PackageImporter(archive)
        archived = importer.import_module(target_name).wrapper
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
            root = Path(temporary)
            for name, code in ((helper_name, "OFFSET = 11\n"), (target_name, source)):
                path = root / (name.rsplit(".", 1)[-1] + ".py")
                path.write_text(code)
                spec = importlib.util.spec_from_file_location(name, path)
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            live = sys.modules[target_name].wrapper
            for name, target, expected in (
                ("archived", archived, 8),
                ("live", live, 12),
            ):
                with self.subTest(origin=name):
                    self.assertEqual(target(1), expected)
                    generate_wrapper_files_skeleton(
                        target, name, _IdentityTransform(), temporary, importer
                    )
                    namespace: dict[str, Any] = {"_package_importer": importer}
                    path = root / name / "wrapper_wrapper.py"
                    exec(compile(path.read_text(), str(path), "exec"), namespace)
                    self.assertEqual(namespace["wrapper"](1), expected)


class TestGetOriginalImportHeader(unittest.TestCase):
    """Unit tests for get_original_import_header function."""

    def test_extracts_import_statements(self) -> None:
        """Test that import statements are extracted correctly."""
        source_code = """
import os
import sys
from typing import List, Tuple
from torch import Tensor

def my_func():
    pass
"""
        result = get_original_import_header(source_code)

        self.assertIn("import os", result)
        self.assertIn("import sys", result)
        self.assertIn("from typing import List, Tuple", result)
        self.assertIn("from torch import Tensor", result)
        self.assertNotIn("def my_func", result)

    def test_returns_empty_header_when_no_imports(self) -> None:
        """Test that source without imports returns empty header."""
        source_code = """
def my_func():
    x = 1
    return x
"""
        result = get_original_import_header(source_code)
        self.assertEqual(result, "")


class TestExtractFunctionSource(unittest.TestCase):
    """Unit tests for _extract_function_source function."""

    def test_extracts_function_from_module_source(self) -> None:
        """Test extracting a function definition from module source."""
        module_source = """
import torch

def helper_func():
    return 1

def target_func(x, y):
    z = x + y
    return z * 2

def another_func():
    pass
"""
        result = _extract_function_source(module_source, "target_func")
        parsed = ast.parse(result)
        self.assertEqual(len(parsed.body), 1)
        self.assertIsInstance(parsed.body[0], ast.FunctionDef)
        self.assertEqual(parsed.body[0].name, "target_func")

    def test_raises_error_for_nonexistent_function(self) -> None:
        """Test that ValueError is raised for non-existent function."""
        module_source = """
def existing_func():
    pass
"""
        with self.assertRaises(ValueError) as context:
            _extract_function_source(module_source, "nonexistent_func")
        self.assertIn("nonexistent_func", str(context.exception))

    def test_extracts_decorated_function(self) -> None:
        """Test extracting a decorated function."""
        module_source = """
import torch

@torch.fx.wrap
def decorated_func(x):
    return x + 1
"""
        result = _extract_function_source(module_source, "decorated_func")
        parsed = ast.parse(result)
        self.assertIsInstance(parsed.body[0], ast.FunctionDef)


class TestGetModuleAndSource(unittest.TestCase):
    """Unit tests for _get_module_and_source function."""

    def test_handles_regular_module(self) -> None:
        """Test handling of regular (non-torch.package) module."""

        def sample_target() -> int:
            return 42

        fn_module, module_source, fn_source = _get_module_and_source(
            sample_target, None
        )

        self.assertIsNotNone(fn_module)
        self.assertIn("def sample_target", fn_source)

    @patch("aot_tensor.transform.wrapper_codegen_utils._extract_function_source")
    def test_handles_torch_package_module(
        self,
        mock_extract: MagicMock,
    ) -> None:
        """Test handling of torch.package loaded module."""
        mock_extract.return_value = "def target_fn(): pass"

        mock_target = MagicMock()
        mock_target.__module__ = "<torch_package_0>.triton_aot.ops.triton_layer_norm"
        mock_target.__name__ = "target_fn"

        mock_importer = MagicMock()
        mock_importer.get_source.return_value = "module source code"
        mock_imported_module = MagicMock()
        mock_importer.import_module.return_value = mock_imported_module

        fn_module, module_source, fn_source = _get_module_and_source(
            mock_target, mock_importer
        )

        mock_importer.get_source.assert_called_once_with(
            "triton_aot.ops.triton_layer_norm"
        )
        mock_importer.import_module.assert_called_once_with(
            "triton_aot.ops.triton_layer_norm"
        )
        mock_extract.assert_called_once_with("module source code", "target_fn")

        self.assertEqual(fn_module, mock_imported_module)
        self.assertEqual(module_source, "module source code")
        self.assertEqual(fn_source, "def target_fn(): pass")


class _StubMarker:
    """Minimal stand-in for a DSL kernel marker (TritonAOT / CuTeAOT)."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class FindSoleMarkerInGlobalsTest(unittest.TestCase):
    """Unit tests for the DSL-agnostic find_sole_marker_in_globals.

    Exercises the shared scan/validation behavior (none / one / missing-spec /
    multiple) directly; each DSL's find_kernel just injects in_specs +
    missing_spec_error and, for Triton, drops the returned global names.
    """

    def _target(self, global_vars: dict[str, object]) -> MagicMock:
        target = MagicMock()
        target.__globals__ = global_vars
        target.__name__ = "wrapper_fn"
        return target

    def test_returns_none_when_no_markers(self) -> None:
        target = self._target({"foo": 42, "bar": "hello"})
        result = find_sole_marker_in_globals(
            target,
            _StubMarker,
            in_specs=lambda var: True,
            missing_spec_error=lambda var: "unused",
        )
        self.assertIsNone(result)

    def test_returns_marker_and_bound_names(self) -> None:
        marker = _StubMarker("k")
        target = self._target({"my_kernel": marker, "noise": 1})
        result = find_sole_marker_in_globals(
            target,
            _StubMarker,
            in_specs=lambda var: True,
            missing_spec_error=lambda var: "unused",
        )
        self.assertIsNotNone(result)
        found, names = result
        self.assertIs(found, marker)
        self.assertEqual(names, {"my_kernel"})

    def test_raises_runtime_error_when_marker_not_in_specs(self) -> None:
        target = self._target({"my_kernel": _StubMarker("k")})
        with self.assertRaises(RuntimeError) as ctx:
            find_sole_marker_in_globals(
                target,
                _StubMarker,
                in_specs=lambda var: False,
                missing_spec_error=lambda var: f"no spec for {var.name}",
            )
        self.assertIn("no spec for k", str(ctx.exception))

    def test_raises_assertion_when_multiple_markers(self) -> None:
        target = self._target({"ka": _StubMarker("a"), "kb": _StubMarker("b")})
        with self.assertRaises(AssertionError) as ctx:
            find_sole_marker_in_globals(
                target,
                _StubMarker,
                in_specs=lambda var: True,
                missing_spec_error=lambda var: "unused",
            )
        self.assertIn("Expected exactly 1 kernel", str(ctx.exception))
