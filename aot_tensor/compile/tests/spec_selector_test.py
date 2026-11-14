# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest
from typing import Any

import torch
from aot_tensor.arg_specs.common import (
    ArgSpec,
    ConstexprArg,
    RawSpec,
    sample_spec_and_hash,
    ScalarArg,
)
from aot_tensor.arg_specs.fly import FlyTensorArg
from aot_tensor.autotune.autotune import Autotune
from aot_tensor.autotune.routing import RoutingTable
from aot_tensor.compile.adapter_base import hash_spec
from aot_tensor.compile.spec_selector import (
    cpp_literal,
    describe,
    distinct_variants,
    match_condition,
    render_selector,
    Route,
    Variant,
    variant_routes,
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
    def test_matches_dtype_rank_unit_stride_and_constexpr_but_not_scalars(
        self,
    ) -> None:
        x = torch.empty(4, 8).t()  # strides (1, 8): dim 0 is the leading dim
        self.assertEqual(
            match_condition(_key(x, True)["args"]),
            f"x.scalar_type() == {_FLOAT} && x.dim() == 2 && "
            "x.stride(0) == 1 && fast == true",
        )

    def test_ignores_the_order_of_the_other_strides(self) -> None:
        # Strides (K, 1, M * K): M <= 1 reorders them, the condition stays.
        def view(m: int) -> torch.Tensor:
            return torch.empty(m, 128).unsqueeze(0).permute(1, 2, 0)

        conditions = {match_condition(_key(view(m), True)["args"]) for m in (512, 1, 0)}
        self.assertEqual(
            conditions,
            {
                f"x.scalar_type() == {_FLOAT} && x.dim() == 3 && "
                "x.stride(1) == 1 && fast == true"
            },
        )

    def test_no_unit_stride_dim_matches_any_strides(self) -> None:
        args = _key(torch.empty(8, 4)[:, ::2], True)["args"]  # strides (4, 2)
        self.assertNotIn("stride", match_condition(args))
        self.assertEqual(describe(args), "x: torch.float32 rank 2, fast=True")

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
        self.assertEqual(
            variants[1].description,
            "x: torch.float32 rank 1 unit-stride dim 0, fast=False",
        )

    def test_specs_differing_only_in_alignment_share_a_variant(self) -> None:
        aligned = _key(torch.empty(8), True)
        unaligned = _key(torch.empty(9)[1:], True)
        self.assertNotEqual(hash_spec(aligned), hash_spec(unaligned))
        self.assertEqual(len(self._variants([aligned, unaligned])), 1)

    def test_symbol_collision_rejected(self) -> None:
        on, off = _key(torch.empty(8), True), _key(torch.empty(8), False)
        with self.assertRaisesRegex(RuntimeError, "share symbol op__deadbeef"):
            distinct_variants("op", [(0, "deadbeef01", on), (1, "deadbeef02", off)])


_TUNED_SPECS: list[ArgSpec] = [
    FlyTensorArg("x"),
    ScalarArg("n", "i32"),
    ConstexprArg("NEG", bool),
    ConstexprArg("BM", int),
]
# Buckets 16 (log2 4) and 1024 (log2 10); BM=2 is not valid with NEG.
_AUTOTUNE = Autotune(
    key=[("x", 0)],
    tuned={16: {"BM": 1}, 1024: {"BM": 2}},
    valid=lambda config, constexprs: not (constexprs["NEG"] and config["BM"] == 2),
)


def _collected(m: int, neg: bool) -> RawSpec:
    """The spec a marker collects for this call: BM filled from ``tuned``."""
    values = _AUTOTUNE.resolve(_TUNED_SPECS, "op", torch.empty(m), m, NEG=neg)
    spec, _ = sample_spec_and_hash(_TUNED_SPECS, "op", "mod", **values)
    return spec


class VariantRoutesTest(unittest.TestCase):
    def test_one_route_per_variant_without_autotune(self) -> None:
        on, off = _key(torch.empty(8), True), _key(torch.empty(8), False)
        keyed = [(i, hash_spec(k), k) for i, k in enumerate([on, off])]
        variants = [v for _, v in distinct_variants("op", keyed)]
        routes = variant_routes("op", [(on, variants[0]), (off, variants[1])], None)
        self.assertEqual(
            routes,
            [
                Route(v.condition, v.description, RoutingTable({(): v.symbol}))
                for v in variants
            ],
        )

    def test_autotuned_variants_share_a_route_per_caller_match(self) -> None:
        # One collected M per NEG; expand compiles every config valid for each.
        specs = _AUTOTUNE.expand(
            [_collected(4096, False), _collected(8, True)], _TUNED_SPECS, "op"
        )
        keyed = [(s, hash_spec(s.hash_key), s.hash_key) for s in specs]
        compiled = distinct_variants("op", keyed)
        x = "x: torch.float32 rank 1 unit-stride dim 0"
        self.assertEqual(
            [v.description for _, v in compiled],
            [f"{x}, NEG=False, BM=1", f"{x}, NEG=False, BM=2", f"{x}, NEG=True, BM=1"],
        )
        a, b, c = (v.symbol for _, v in compiled)
        routes = variant_routes("op", [(s.hash_key, v) for s, v in compiled], _AUTOTUNE)
        caller = f"x.scalar_type() == {_FLOAT} && x.dim() == 1 && x.stride(0) == 1"
        self.assertEqual(
            routes,
            [
                Route(
                    f"{caller} && NEG == false",
                    f"{x}, NEG=False",
                    RoutingTable({(4,): a, (10,): b}),
                ),
                Route(
                    f"{caller} && NEG == true",
                    f"{x}, NEG=True",
                    RoutingTable({(4,): c}),
                ),
            ],
        )


class RenderSelectorTest(unittest.TestCase):
    CALL: dict[str, str] = {"op__a": "op__a_run(x)", "op__b": "op__b_run(x)"}

    def test_dispatches_in_order_then_fails(self) -> None:
        routes = [
            Route("fast == true", "fast=True", RoutingTable({(): "op__a"})),
            Route("fast == false", "fast=False", RoutingTable({(): "op__b"})),
        ]
        self.assertEqual(
            render_selector(routes, (), self.CALL, "fail"),
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

    def test_route_with_variants_takes_the_nearest_entry(self) -> None:
        table = RoutingTable({(4,): "op__a", (10,): "op__b", (13,): "op__a"})
        lines = render_selector(
            [Route("NEG == false", "NEG=False", table)],
            ("x.size(0)",),
            self.CALL,
            "fail",
        )
        self.assertEqual(
            lines,
            [
                "if (NEG == false) {",
                "  static constexpr int64_t aott_entries[3][1] = {{4}, {10}, {13}};",
                "  const int64_t aott_keys[] = {x.size(0)};",
                "  switch (aott_nearest_entry(aott_entries, aott_keys)) {",
                "    case 1:",
                "      op__b_run(x);",
                "      return;",
                "    default:",
                "      op__a_run(x);",
                "      return;",
                "  }",
                "}",
                'fail("no compiled specialization matches the arguments; '
                'compiled: NEG=False");',
            ],
        )
