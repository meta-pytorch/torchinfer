# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest
from collections.abc import Callable
from dataclasses import dataclass

import torch
from aot_tensor.arg_specs.common import (
    ArgSpec,
    ConstexprArg,
    leading_dim,
    module_basename_for_callable,
    resolve_runtime_args,
    sample_spec_and_hash,
    scalar_hash_key,
    ScalarArg,
    tensor_hash_key,
    TensorArg,
)
from aot_tensor.arg_specs.cute import CuTeTensorArg
from aot_tensor.arg_specs.fly import FlyTensorArg
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
    SPECS: list[ArgSpec] = [
        TensorArg("x"),
        ScalarArg("n", "i32"),
        ConstexprArg("flag", bool),
    ]

    @parameterized.expand(
        [
            ("base", TensorArg("x", leading_dim=1)),
            ("cute", CuTeTensorArg("x", leading_dim=1)),
            ("fly", FlyTensorArg("x", leading_dim=1)),
        ]
    )
    def test_tensor_key_uses_the_spec_leading_dim(
        self, _name: str, arg: TensorArg
    ) -> None:
        # (8, 1) has stride 1 in both dims; auto-detection would pick dim 0.
        spec, _ = sample_spec_and_hash([arg], "op", "toy", torch.empty(8, 1))
        self.assertEqual(spec.hash_key["args"][0]["tensor"]["leading_dim"], 1)

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

    def test_unknown_spec_kind_rejected(self) -> None:
        # _Arg is not an ArgSpec.
        with self.assertRaisesRegex(RuntimeError, "unsupported arg spec"):
            # pyrefly: ignore[bad-argument-type]
            sample_spec_and_hash([_Arg("x")], "op", "toy", torch.randn(8))


_TENSOR_ARG_CLASSES: list[tuple[str, type[TensorArg]]] = [
    ("base", TensorArg),
    ("cute", CuTeTensorArg),
    ("fly", FlyTensorArg),
]


class TensorArgTest(unittest.TestCase):
    @parameterized.expand(_TENSOR_ARG_CLASSES)
    def test_only_name_is_positional(self, _name: str, cls: type[TensorArg]) -> None:
        with self.assertRaises(TypeError):
            # pyrefly: ignore[bad-argument-count]
            cls("x", 1)

    @parameterized.expand(_TENSOR_ARG_CLASSES)
    def test_negative_leading_dim_rejected(
        self, _name: str, cls: type[TensorArg]
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "leading_dim must be >= 0"):
            cls("x", leading_dim=-1)

    @parameterized.expand(_TENSOR_ARG_CLASSES)
    def test_alignment_must_be_a_power_of_2(
        self, _name: str, cls: type[TensorArg]
    ) -> None:
        for alignment in (1, 4, 16):
            self.assertEqual(cls("x", alignment=alignment).alignment, alignment)
        for alignment in (0, -16, 3, 24):
            with self.subTest(alignment=alignment):
                with self.assertRaisesRegex(RuntimeError, "must be a power of 2"):
                    cls("x", alignment=alignment)


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


def _jagged_view(m: int) -> torch.Tensor:
    # How the jagged_dense_bmm CuTeDSL launchers pass (M, K): [M, K, 1] with
    # strides (K, 1, M * K).
    return torch.empty(m, 128).unsqueeze(0).permute(1, 2, 0)


class LeadingDimTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("contiguous", lambda: torch.empty(4, 8), 1),
            ("transposed", lambda: torch.empty(8, 4).t(), 0),
            # Both dims have stride 1; only dim 0 has size > 1.
            ("unit_stride_tie", lambda: torch.empty(8, 1), 0),
            ("no_unit_stride", lambda: torch.empty(8, 4)[:, ::2], None),
            ("ambiguous", lambda: torch.empty(1, 1), None),
            # M <= 1 reorders the other strides; the unit-stride dim stays.
            *[
                (f"jagged_view_m{m}", lambda m=m: _jagged_view(m), 1)
                for m in (512, 1, 0)
            ],
        ]
    )
    def test_auto_detect(
        self, _name: str, make: Callable[[], torch.Tensor], expected: int | None
    ) -> None:
        self.assertEqual(leading_dim(make()), expected)

    def test_spec_leading_dim_wins(self) -> None:
        self.assertEqual(leading_dim(torch.empty(8, 1), 1), 1)

    def test_spec_leading_dim_out_of_range_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "out of range for a rank-2"):
            leading_dim(torch.empty(8, 1), 2)


class TensorHashKeyTest(unittest.TestCase):
    def test_contiguous_tensor(self) -> None:
        self.assertEqual(
            tensor_hash_key(torch.empty(4, 8)),
            {
                "dtype": "torch.float32",
                "dim": 2,
                "leading_dim": 1,
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
            ("transposed", lambda: torch.empty(8, 4).t(), "leading_dim", 0),
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
