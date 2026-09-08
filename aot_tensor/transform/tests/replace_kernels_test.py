# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import os
import shutil
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn as nn
from aot_tensor.compile.compile_state import get_aott_compile_state
from aot_tensor.transform.replace_kernels import (
    _find_wrapper_files,
    _load_wrapper_module,
    replace_kernels,
)
from torch.fx import symbolic_trace


def _make_wrapper_content(fn_name: str) -> str:
    """Generate wrapper file content for a given function name."""
    return f"""
def {fn_name}(*args, **kwargs):
    return args[0] if args else None
"""


class ReplaceKernelsTest(unittest.TestCase):
    """Tests for the replace_kernels function.

    The replace_kernels function dynamically loads wrapper modules from
    .py files and replaces matching triton nodes in an FX GraphModule.
    """

    def setUp(self) -> None:
        """Set up a temporary directory with wrapper files."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_modules = dict(sys.modules)

    def tearDown(self) -> None:
        """Clean up temporary directory and restore sys.modules."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        for module_name in list(sys.modules.keys()):
            if module_name not in self.original_modules:
                del sys.modules[module_name]

    @contextmanager
    def _completed_session(self, compile_path: str) -> Iterator[None]:
        """Stand in for the state a finished ``AOTTCompileSession`` leaves behind,
        which ``replace_kernels`` requires via
        ``assert_aott_compile_session_completed``."""
        state = get_aott_compile_state()
        with (
            patch.object(state, "compile_path", compile_path),
            patch.object(state, "session_completed", True),
        ):
            yield

    def _create_wrapper_file(self, fn_name: str, content: str | None = None) -> str:
        """Create a wrapper file in a subdirectory of the temp directory."""
        subdir = os.path.join(self.temp_dir, f"module_{fn_name}")
        os.makedirs(subdir, exist_ok=True)
        wrapper_path = os.path.join(subdir, f"{fn_name}_wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write(content if content else _make_wrapper_content(fn_name))
        return wrapper_path

    def _create_graph_module_with_triton_op(
        self, triton_fn_name: str
    ) -> torch.fx.GraphModule:
        """Create a real GraphModule with a triton-like call_function node."""

        class TritonOp:
            def __init__(self, name: str) -> None:
                self.__name__ = name

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        triton_op = TritonOp(triton_fn_name)

        class SimpleModule(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        fx_m = symbolic_trace(SimpleModule())

        for node in fx_m.graph.nodes:
            if node.op == "call_function":
                with fx_m.graph.inserting_before(node):
                    triton_node = fx_m.graph.call_function(triton_op, (node.args[0],))
                    node.replace_input_with(node.args[0], triton_node)
                break

        fx_m.recompile()
        return fx_m

    def test_replace_kernels_loads_single_wrapper(self) -> None:
        """Test loading a single wrapper module."""
        self._create_wrapper_file("_test_kernel")
        fx_m = self._create_graph_module_with_triton_op("_test_kernel")

        with self._completed_session(self.temp_dir):
            result = replace_kernels(fx_m)

        self.assertIsInstance(result, torch.fx.GraphModule)
        # Verify the module still runs
        _ = result(torch.randn(4, 4))

    def test_replace_kernels_loads_multiple_wrappers(self) -> None:
        """Test loading multiple wrapper modules."""
        for fn_name in ["_kernel_a", "_kernel_b", "_kernel_c"]:
            self._create_wrapper_file(fn_name)

        fx_m = self._create_graph_module_with_triton_op("_kernel_a")

        with self._completed_session(self.temp_dir):
            result = replace_kernels(fx_m)

        self.assertIsInstance(result, torch.fx.GraphModule)

    def test_replace_kernels_ignores_non_wrapper_files(self) -> None:
        """Test that only *_wrapper.py files are loaded."""
        self._create_wrapper_file("_valid_kernel")

        # Create files that should be ignored
        for filename, content in [
            ("not_a_wrapper.py", "def foo(): pass"),
            ("_some_kernel.py", "def bar(): pass"),
            ("README.md", "# readme"),
        ]:
            with open(os.path.join(self.temp_dir, filename), "w") as f:
                f.write(content)

        fx_m = self._create_graph_module_with_triton_op("_valid_kernel")

        with self._completed_session(self.temp_dir):
            result = replace_kernels(fx_m)

        self.assertIsInstance(result, torch.fx.GraphModule)

    def test_replace_kernels_raises_when_dir_not_exists(self) -> None:
        """Test AssertionError when triton_aot_compile directory doesn't exist."""
        fx_m = self._create_graph_module_with_triton_op("_any_kernel")

        with self._completed_session("/path/does/not/exist/nonexistent"):
            with self.assertRaises(AssertionError) as ctx:
                replace_kernels(fx_m)

            self.assertIn("triton_aot_compile dir does not exist", str(ctx.exception))

    def test_replace_kernels_raises_when_no_replacements(self) -> None:
        """Test AssertionError when no nodes are replaced."""
        self._create_wrapper_file("_unmatched_kernel")
        fx_m = self._create_graph_module_with_triton_op("_different_kernel")

        with self._completed_session(self.temp_dir):
            with self.assertRaises(AssertionError) as ctx:
                replace_kernels(fx_m)

            self.assertIn("No ops were replaced", str(ctx.exception))


class FindWrapperFilesTest(unittest.TestCase):
    """Tests for _find_wrapper_files."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_returns_empty_for_empty_directory(self) -> None:
        result = _find_wrapper_files(self.temp_dir)
        self.assertEqual(result, [])

    def test_finds_wrapper_in_subdirectory(self) -> None:
        subdir = os.path.join(self.temp_dir, "kernel_a")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "_my_kernel_wrapper.py"), "w") as f:
            f.write("pass")

        result = _find_wrapper_files(self.temp_dir)
        self.assertEqual(len(result), 1)
        wrapper_name, fn_name, wrapper_path = result[0]
        self.assertEqual(wrapper_name, "_my_kernel_wrapper")
        self.assertEqual(fn_name, "_my_kernel")
        self.assertEqual(wrapper_path, os.path.join(subdir, "_my_kernel_wrapper.py"))

    def test_ignores_deeply_nested_wrappers(self) -> None:
        """Wrappers 2+ levels deep are not returned."""
        deep = os.path.join(self.temp_dir, "level1", "level2")
        os.makedirs(deep)
        with open(os.path.join(deep, "_deep_wrapper.py"), "w") as f:
            f.write("pass")

        result = _find_wrapper_files(self.temp_dir)
        self.assertEqual(result, [])

    def test_ignores_non_wrapper_files(self) -> None:
        subdir = os.path.join(self.temp_dir, "kernel")
        os.makedirs(subdir)
        for name in ["helper.py", "_kernel_original.py", "README.md"]:
            with open(os.path.join(subdir, name), "w") as f:
                f.write("pass")

        result = _find_wrapper_files(self.temp_dir)
        self.assertEqual(result, [])

    def test_finds_multiple_wrappers(self) -> None:
        for kernel in ["_ka", "_kb"]:
            subdir = os.path.join(self.temp_dir, f"mod{kernel}")
            os.makedirs(subdir)
            with open(os.path.join(subdir, f"{kernel}_wrapper.py"), "w") as f:
                f.write("pass")

        result = _find_wrapper_files(self.temp_dir)
        self.assertEqual(len(result), 2)
        fn_names = sorted(r[1] for r in result)
        self.assertEqual(fn_names, ["_ka", "_kb"])


class LoadWrapperModuleTest(unittest.TestCase):
    """Tests for _load_wrapper_module."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.original_modules = dict(sys.modules)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
        for module_name in list(sys.modules.keys()):
            if module_name not in self.original_modules:
                del sys.modules[module_name]

    def test_loads_and_returns_wrapper_function(self) -> None:
        wrapper_path = os.path.join(self.temp_dir, "_my_fn_wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write("def _my_fn(x):\n    return x\n")

        result = _load_wrapper_module("_my_fn_wrapper", "_my_fn", wrapper_path, None)
        self.assertIsNotNone(result)
        self.assertEqual(result.__name__, "_my_fn")

    def test_returns_none_when_fn_not_in_module(self) -> None:
        wrapper_path = os.path.join(self.temp_dir, "_missing_wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write("def other_fn(): pass\n")

        result = _load_wrapper_module(
            "_missing_wrapper", "_missing", wrapper_path, None
        )
        self.assertIsNone(result)

    def test_registers_module_in_sys_modules(self) -> None:
        wrapper_path = os.path.join(self.temp_dir, "_reg_wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write("def _reg(x): return x\n")

        _load_wrapper_module("_reg_wrapper", "_reg", wrapper_path, None)
        self.assertIn("_reg_wrapper", sys.modules)

    def test_injects_package_importer_when_provided(self) -> None:
        wrapper_path = os.path.join(self.temp_dir, "_pkg_wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write("def _pkg(): pass\n")

        mock_importer = unittest.mock.MagicMock()
        _load_wrapper_module("_pkg_wrapper", "_pkg", wrapper_path, mock_importer)

        loaded = sys.modules["_pkg_wrapper"]
        self.assertIs(loaded._package_importer, mock_importer)  # type: ignore[attr-defined]
