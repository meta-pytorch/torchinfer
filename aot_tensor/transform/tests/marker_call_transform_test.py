# Copyright (c) Meta Platforms, Inc. and affiliates.

import ast
import unittest

import torch
from aot_tensor.transform.marker_call_transform import MarkerCallTransform
from parameterized import parameterized


class _DirTransform(MarkerCallTransform):
    def kernel_dir(self) -> str:
        return "kdir"


def _transform() -> MarkerCallTransform:
    return _DirTransform(op_name="_dsl_add", global_names={"_add_aot"})


def _func(source: str) -> ast.FunctionDef:
    func = ast.parse(source).body[0]
    assert isinstance(func, ast.FunctionDef)
    return func


class MarkerCallTransformTest(unittest.TestCase):
    def test_rewrites_marker_call_to_torch_op(self) -> None:
        tree = ast.parse("def f(x, n):\n    _add_aot(x, n=n)\n    other(x)\n")
        code = ast.unparse(_transform().visit(tree))
        self.assertIn("torch.ops.aot_tensor._dsl_add(x, n=n)", code)
        self.assertIn("other(x)", code)

    @parameterized.expand(
        [
            ("calls_marker", "    _add_aot(x)\n", 0),
            ("no_call", "    return x\n", 1),
            ("call_only_in_nested_def", "    def g(y):\n        _add_aot(y)\n", 1),
        ]
    )
    def test_strips_jit_unused_only_on_the_calling_scope(
        self, _name: str, body: str, expected_decorators: int
    ) -> None:
        func = _func("@torch.jit.unused\ndef f(x):\n" + body)
        _transform().visit_FunctionDef(func)
        self.assertEqual(len(func.decorator_list), expected_decorators)

    def test_so_loading_points_at_kernel_dir(self) -> None:
        with_call = ast.parse("def f(x):\n    _add_aot(x)\n")
        without_call = ast.parse("def f(x):\n    return x\n")
        self.assertIn(
            'torch.ops.load_library("/out/kdir/dsl_add.so")',
            _transform().generate_so_loading_code(with_call, "/out"),
        )
        self.assertEqual(
            _transform().generate_so_loading_code(without_call, "/out"), ""
        )

    def test_python_meta_kernel_only_when_requested(self) -> None:
        with_call = ast.parse("def f(x):\n    _add_aot(x)\n")
        meta_impl = 'torch.library.impl("aot_tensor::_dsl_add", "Meta"'
        self.assertNotIn(
            meta_impl, _transform().generate_so_loading_code(with_call, "/out")
        )
        transform = _transform()
        transform.python_meta_kernel = True
        self.assertIn(meta_impl, transform.generate_so_loading_code(with_call, "/out"))

    def test_python_meta_kernel_registers_once_per_process(self) -> None:
        torch.library.define("aot_tensor::_dsl_add", "(Tensor x) -> ()")
        transform = _transform()
        transform.python_meta_kernel = True
        code = transform.generate_so_loading_code(
            ast.parse("def f(x):\n    _add_aot(x)\n"), "/out"
        )
        meta_only = code.replace('torch.ops.load_library("/out/kdir/dsl_add.so")', "")
        # Two wrapper modules of the same op, e.g. two PackageImporters.
        exec(meta_only, {})
        exec(meta_only, {})
        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "aot_tensor::_dsl_add", "Meta"
            )
        )
