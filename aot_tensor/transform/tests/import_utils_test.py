# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest
from unittest.mock import MagicMock

from aot_tensor.transform.import_utils import (
    _is_extern_module,
    get_original_import_header,
    rewrite_package_imports,
)
from parameterized import parameterized


class GetOriginalImportHeaderTest(unittest.TestCase):
    """Tests for get_original_import_header."""

    def test_extracts_all_imports(self) -> None:
        source = """
import os
import sys
from typing import List, Tuple
from torch import Tensor

def my_func():
    pass
"""
        result = get_original_import_header(source)
        self.assertIn("import os", result)
        self.assertIn("import sys", result)
        self.assertIn("from typing import List, Tuple", result)
        self.assertIn("from torch import Tensor", result)
        self.assertNotIn("def my_func", result)

    def test_empty_when_no_imports(self) -> None:
        source = """
def my_func():
    return 1
"""
        result = get_original_import_header(source)
        self.assertEqual(result, "")


class IsExternModuleTest(unittest.TestCase):
    """Tests for _is_extern_module."""

    @parameterized.expand(
        [
            ("exact_match", "torch", {"torch", "typing"}, True),
            ("parent_match", "torch.fx.graph", {"torch", "typing"}, True),
            ("no_match", "hammer.ops", {"torch", "typing"}, False),
            ("partial_no_match", "torchvision", {"torch", "typing"}, False),
            ("empty_set", "torch", set(), False),
        ]
    )
    def test_is_extern(
        self,
        _name: str,
        module_name: str,
        extern_modules: set[str],
        expected: bool,
    ) -> None:
        self.assertEqual(_is_extern_module(module_name, extern_modules), expected)


class RewritePackageImportsTest(unittest.TestCase):
    """Tests for rewrite_package_imports."""

    def _make_importer(self, extern_modules: list[str]) -> MagicMock:
        importer = MagicMock()
        importer.extern_modules = extern_modules
        return importer

    def test_extern_imports_kept(self) -> None:
        header = "import torch\nfrom typing import Optional\n"
        importer = self._make_importer(["torch", "typing"])

        result = rewrite_package_imports(header, importer)

        self.assertIn("import torch", result)
        self.assertIn("from typing import Optional", result)
        self.assertNotIn("_package_importer", result)

    def test_intern_imports_rewritten(self) -> None:
        header = "from hammer.ops.triton.utils import _switch, ALLOW_TF32\n"
        importer = self._make_importer(["torch", "typing"])

        result = rewrite_package_imports(header, importer)

        self.assertNotIn("from hammer", result)
        self.assertIn(
            "_package_importer.import_module('hammer.ops.triton.utils')", result
        )
        self.assertIn("_switch", result)
        self.assertIn("ALLOW_TF32", result)

    def test_mixed_extern_and_intern(self) -> None:
        header = "import torch\nfrom hammer.utils import foo\n"
        importer = self._make_importer(["torch"])

        result = rewrite_package_imports(header, importer)

        self.assertIn("import torch", result)
        self.assertIn("_package_importer.import_module('hammer.utils')", result)
        self.assertIn("foo", result)

    def test_empty_header(self) -> None:
        result = rewrite_package_imports("", self._make_importer(["torch"]))
        self.assertEqual(result, "")
