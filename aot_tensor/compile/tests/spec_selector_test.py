# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest
from typing import Any

import torch
from aot_tensor.arg_specs.common import ConstexprArg, sample_spec_and_hash, ScalarArg
from aot_tensor.arg_specs.fly import FlyTensorArg
from aot_tensor.compile.adapter_base import hash_spec
from aot_tensor.compile.spec_selector import (
    cpp_literal,
    distinct_variants,
    match_condition,
    render_selector,
    Variant,
)
from parameterized import parameterized

_FLOAT = "torch::headeronly::ScalarType::Float"


def _key(x: torch.Tensor, fast: bool) -> dict[str, Any]:
    spec, _ = sample_spec_and_hash(
        [FlyTensorArg("x"), ScalarArg("n", "i32"), ConstexprArg("fast", bool)],
        "op",
        "mod",
        x,
        3,
        fast,
    )
    return spec.hash_key


class MatchConditionTest(unittest.TestCase):
    def test_matches_dtype_rank_layout_and_constexpr_but_not_scalars(self) -> None:
        x = torch.empty(4, 8).t()  # strides (1, 8): dim 1 is outermost
        self.assertEqual(
            match_condition(_key(x, True)["args"]),
            f"x.scalar_type() == {_FLOAT} && x.dim() == 2 && "
            "x.stride(1) > x.stride(0) && fast == true",
        )

    def test_tied_strides_match_in_index_order(self) -> None:
        # strides (1, 1) sort to layout_order (0, 1), so the tie must pass.
        self.assertIn(
            "x.stride(0) >= x.stride(1)",
            match_condition(_key(torch.empty(1, 1), True)["args"]),
        )

    @parameterized.expand(
        [("non_finite", float("inf"), "non-finite"), ("beyond_int64", 2**63, "int64")]
    )
    def test_rejects_constexprs_without_an_exact_literal(
        self, _name: str, value: object, error: str
    ) -> None:
        with self.assertRaisesRegex(NotImplementedError, error):
            cpp_literal(value)

    def test_rejects_dtypes_the_stable_abi_cannot_name(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "complex64"):
            match_condition(_key(torch.empty(2, dtype=torch.complex64), True)["args"])


class DistinctVariantsTest(unittest.TestCase):
    def _variants(self, keys: list[dict[str, Any]]) -> list[Variant]:
        keyed = [(i, hash_spec(k), k) for i, k in enumerate(keys)]
        return [v for _, v in distinct_variants("op", keyed)]

    def test_one_variant_per_distinct_condition_named_by_spec_hash(self) -> None:
        on, off = _key(torch.empty(8), True), _key(torch.empty(8), False)
        variants = self._variants([on, off])
        self.assertEqual(
            [v.symbol for v in variants],
            [f"op__{hash_spec(on)[:8]}", f"op__{hash_spec(off)[:8]}"],
        )
        self.assertEqual(variants[1].description, "x: torch.float32 rank 1, fast=False")

    def test_specs_differing_only_in_alignment_share_a_variant(self) -> None:
        aligned = _key(torch.empty(8), True)
        unaligned = _key(torch.empty(9)[1:], True)
        self.assertNotEqual(hash_spec(aligned), hash_spec(unaligned))
        self.assertEqual(len(self._variants([aligned, unaligned])), 1)

    def test_symbol_collision_rejected(self) -> None:
        on, off = _key(torch.empty(8), True), _key(torch.empty(8), False)
        with self.assertRaisesRegex(RuntimeError, "share symbol op__deadbeef"):
            distinct_variants("op", [(0, "deadbeef01", on), (1, "deadbeef02", off)])


class RenderSelectorTest(unittest.TestCase):
    def test_dispatches_in_order_then_fails(self) -> None:
        variants = [
            Variant("op__a", "fast == true", "fast=True"),
            Variant("op__b", "fast == false", "fast=False"),
        ]
        lines = render_selector(
            variants, {"op__a": "op__a_run(x)", "op__b": "op__b_run(x)"}, "fail"
        )
        self.assertEqual(
            lines,
            [
                "if (fast == true) {",
                "  op__a_run(x);",
                "  return;",
                "}",
                "if (fast == false) {",
                "  op__b_run(x);",
                "  return;",
                "}",
                'fail("no compiled specialization matches the arguments; '
                'compiled: fast=True; fast=False");',
            ],
        )
