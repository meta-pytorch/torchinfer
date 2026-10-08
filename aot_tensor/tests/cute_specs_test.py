# Copyright (c) Meta Platforms, Inc. and affiliates.


import unittest

import torch
from aot_tensor.cute_specs import (
    _scalar_hash_key,
    _tensor_hash_key,
    CuTeArgSpec,
    CuTeConstexprArg,
    CuTeRawSpec,
    CuTeScalarArg,
    CuTeTensorArg,
    module_basename_for_callable,
    resolve_runtime_args,
)
from parameterized import parameterized


class ResolveRuntimeArgsTest(unittest.TestCase):
    SPECS: list[CuTeArgSpec] = [CuTeTensorArg("x"), CuTeScalarArg("n", "i32")]

    @parameterized.expand(
        [
            ("positional_and_kwargs", ("X",), {"n": 4}),
            ("all_positional", ("X", 4), {}),
        ]
    )
    def test_binds_args(self, _name: str, args: tuple, kwargs: dict) -> None:
        self.assertEqual(
            resolve_runtime_args(self.SPECS, "op", *args, **kwargs),
            {"x": "X", "n": 4},
        )

    @parameterized.expand(
        [
            ("too_many_positional", ("X", 4, 5), {}, "expected at most 2"),
            ("missing_arg", ("X",), {}, "missing arg n"),
            (
                "unexpected_kwarg",
                ("X",),
                {"n": 4, "z": 9},
                r"unexpected kwargs \['z'\]",
            ),
            (
                "duplicate_positional_and_kwarg",
                ("X", 4),
                {"x": "X2"},
                r"multiple values for arg\(s\) \['x'\]",
            ),
        ]
    )
    def test_rejects(self, _name: str, args: tuple, kwargs: dict, regex: str) -> None:
        with self.assertRaisesRegex(RuntimeError, regex):
            resolve_runtime_args(self.SPECS, "op", *args, **kwargs)


class ScalarHashKeyTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("i32_int", 3, "i32"),
            ("i64_int", 3, "i64"),
            ("fp32_float", 1.5, "fp32"),
            ("fp32_int", 2, "fp32"),  # int is an acceptable fp32 sample
            ("bool_bool", True, "bool"),
        ]
    )
    def test_accepts(self, _name: str, value: object, dtype: str) -> None:
        self.assertEqual(_scalar_hash_key(value, dtype), {"dtype": dtype})

    @parameterized.expand(
        [
            ("i32_float", 1.5, "i32", "expected int scalar"),
            ("fp32_str", "x", "fp32", "expected float scalar"),
            ("bool_int", 1, "bool", "expected bool scalar"),
        ]
    )
    def test_rejects(self, _name: str, value: object, dtype: str, regex: str) -> None:
        with self.assertRaisesRegex(RuntimeError, regex):
            _scalar_hash_key(value, dtype)


class TensorHashKeyTest(unittest.TestCase):
    def test_contiguous_tensor(self) -> None:
        key = _tensor_hash_key(torch.empty(4, 8))
        self.assertEqual(
            key,
            {
                "dtype": "torch.float32",
                "dim": 2,
                "layout_order": (0, 1),
                "aligned16": True,
            },
        )

    def test_shape_independent(self) -> None:
        # cutedsl marks layouts dynamic: one .so serves any shape of a given
        # dtype/rank/alignment/layout, so two different contiguous shapes must
        # collapse to the SAME key (otherwise compile_cutedsl_to_cpp rejects
        # len(specs) > 1 for an op invoked with multiple runtime shapes).
        self.assertEqual(
            _tensor_hash_key(torch.empty(4, 8)),
            _tensor_hash_key(torch.empty(16, 32)),
        )

    def test_layout_order_distinguishes_transpose(self) -> None:
        # Same dtype/rank/alignment but a different memory layout (dim order by
        # descending stride) must NOT alias — a transposed view keys differently
        # from its contiguous source.
        contig = torch.empty(4, 8)
        transposed = contig.t()
        self.assertEqual(_tensor_hash_key(transposed)["layout_order"], (1, 0))
        self.assertNotEqual(_tensor_hash_key(contig), _tensor_hash_key(transposed))

    def test_dtype_distinguishes(self) -> None:
        self.assertNotEqual(
            _tensor_hash_key(torch.empty(4, 8, dtype=torch.float32)),
            _tensor_hash_key(torch.empty(4, 8, dtype=torch.float16)),
        )

    def test_unaligned_pointer(self) -> None:
        # Drop one float32 (4 bytes) off a 64-byte-aligned base so the data_ptr
        # no longer sits on a 16-byte boundary.
        t = torch.empty(17)[1:]
        self.assertNotEqual(t.data_ptr() % 16, 0)
        self.assertFalse(_tensor_hash_key(t)["aligned16"])

    def test_non_tensor_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected a torch.Tensor-like"):
            _tensor_hash_key(object())


def _vector_add_kernel() -> None:
    pass


class CuTeRawSpecFromCallTest(unittest.TestCase):
    SPECS: list[CuTeArgSpec] = [
        CuTeTensorArg("x"),
        CuTeScalarArg("n", "i32"),
        CuTeConstexprArg("flag", bool),
    ]

    def test_hash_key_and_runtime_args(self) -> None:
        x = torch.randn(8)
        spec = CuTeRawSpec.from_call(
            self.SPECS, "_cutedsl_vector_add", "toy_cutedsl", x, 4, flag=True
        )
        # tensor is held by identity (== on tensors isn't a plain bool)
        self.assertIs(spec.runtime_args[0], x)
        self.assertEqual(spec.runtime_args[1:], (4,))
        self.assertEqual(spec.runtime_kwargs, {"flag": True})
        self.assertEqual(spec.hash_key["name"], "_cutedsl_vector_add")
        self.assertEqual(spec.hash_key["jit_module"], "toy_cutedsl")
        self.assertEqual(
            spec.hash_key["args"],
            [
                {"name": "x", "tensor": _tensor_hash_key(x)},
                {"name": "n", "scalar": {"dtype": "i32"}},
                {"name": "flag", "constexpr": True},
            ],
        )

    def test_constexpr_type_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "constexpr arg flag"):
            CuTeRawSpec.from_call(
                self.SPECS, "op", "toy_cutedsl", torch.randn(8), 4, flag="nope"
            )


class ModuleBasenameForCallableTest(unittest.TestCase):
    # __module__ is "aot_tensor.tests.cute_specs_test"; we want the last segment.
    def test_function_uses_module(self) -> None:
        self.assertEqual(
            module_basename_for_callable(_vector_add_kernel), "cute_specs_test"
        )

    def test_instance_uses_class_module(self) -> None:
        # A callable instance (like a cute kernel object) resolves to the module
        # its class is defined in.
        class _Kernel:
            def __call__(self) -> None:
                pass

        self.assertEqual(module_basename_for_callable(_Kernel()), "cute_specs_test")
