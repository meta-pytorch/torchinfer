# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest
from collections.abc import Callable
from dataclasses import dataclass

import torch
from aot_tensor.arg_specs.common import (
    ConstexprArg,
    module_basename_for_callable,
    NamedArgSpec,
    resolve_runtime_args,
    sample_spec_and_hash,
    scalar_hash_key,
    ScalarArg,
    tensor_hash_key,
)
from parameterized import parameterized


@dataclass(frozen=True)
class _Arg:
    name: str


class ResolveRuntimeArgsTest(unittest.TestCase):
    SPECS: list[_Arg] = [_Arg("x"), _Arg("n")]

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


class SampleSpecAndHashTest(unittest.TestCase):
    # _Arg is neither scalar nor constexpr, so it is keyed as a tensor.
    SPECS: list[NamedArgSpec] = [
        _Arg("x"),
        ScalarArg("n", "i32"),
        ConstexprArg("flag", bool),
    ]

    def test_hash_key_and_runtime_args(self) -> None:
        x = torch.randn(8)
        spec, _ = sample_spec_and_hash(self.SPECS, "op", "toy", x, 4, flag=True)
        self.assertIs(spec.runtime_args[0], x)
        self.assertEqual(spec.runtime_args[1:], (4,))
        self.assertEqual(spec.runtime_kwargs, {"flag": True})
        self.assertEqual(
            spec.hash_key,
            {
                "name": "op",
                "jit_module": "toy",
                "args": [
                    {"name": "x", "tensor": tensor_hash_key(x)},
                    {"name": "n", "scalar": {"dtype": "i32"}},
                    {"name": "flag", "constexpr": True},
                ],
            },
        )

    def test_constexpr_type_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "constexpr arg flag"):
            sample_spec_and_hash(self.SPECS, "op", "toy", torch.randn(8), 4, flag="no")


class ScalarHashKeyTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("i32_int", 3, "i32"),
            ("fp32_float", 1.5, "fp32"),
            ("fp32_int", 2, "fp32"),  # int is an acceptable fp32 sample
            ("bool_bool", True, "bool"),
        ]
    )
    def test_accepts(self, _name: str, value: object, dtype: str) -> None:
        self.assertEqual(scalar_hash_key(value, dtype), {"dtype": dtype})

    @parameterized.expand(
        [
            ("i32_float", 1.5, "i32", "expected int scalar"),
            ("fp32_str", "x", "fp32", "expected float scalar"),
            ("bool_int", 1, "bool", "expected bool scalar"),
        ]
    )
    def test_rejects(self, _name: str, value: object, dtype: str, regex: str) -> None:
        with self.assertRaisesRegex(RuntimeError, regex):
            scalar_hash_key(value, dtype)


class TensorHashKeyTest(unittest.TestCase):
    def test_contiguous_tensor(self) -> None:
        self.assertEqual(
            tensor_hash_key(torch.empty(4, 8)),
            {
                "dtype": "torch.float32",
                "dim": 2,
                "layout_order": (0, 1),
                "aligned16": True,
            },
        )

    def test_shape_independent(self) -> None:
        self.assertEqual(
            tensor_hash_key(torch.empty(4, 8)),
            tensor_hash_key(torch.empty(16, 32)),
        )

    @parameterized.expand(
        [
            ("transposed", lambda: torch.empty(8, 4).t(), "layout_order", (1, 0)),
            (
                "fp16",
                lambda: torch.empty(4, 8, dtype=torch.half),
                "dtype",
                "torch.float16",
            ),
            # 4 bytes past an aligned base.
            ("unaligned", lambda: torch.empty(17)[1:], "aligned16", False),
        ]
    )
    def test_field(
        self, _name: str, make: Callable[[], torch.Tensor], field: str, expected: object
    ) -> None:
        self.assertEqual(tensor_hash_key(make())[field], expected)

    def test_non_tensor_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected a torch.Tensor-like"):
            tensor_hash_key(object())


def _kernel_fn() -> None:
    pass


class _KernelObj:
    def __call__(self) -> None:
        pass


class ModuleBasenameForCallableTest(unittest.TestCase):
    def test_function_and_callable_instance(self) -> None:
        self.assertEqual(module_basename_for_callable(_kernel_fn), "common_test")
        self.assertEqual(module_basename_for_callable(_KernelObj()), "common_test")
