# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# Multi-specialization dispatch for the generated torch op, shared by CuTeDSL
# and FlyDSL. Reads only a spec's ``hash_key["args"]`` (built by
# ``arg_specs.common``), so it imports no DSL:
#
#   {"name": n, "tensor": {"dtype", "dim", "leading_dim", "aligned16"}}
#   {"name": n, "scalar": {"dtype"}}
#   {"name": n, "constexpr": value}
#
# A tensor matches on what the compiled code assumes: dtype, rank, and the
# static unit stride of its leading dim (``arg_specs.common.leading_dim``).
#
# Not matched:
# - scalar: its dtype comes from the arg spec, so it is the same in every spec.
# - aligned16: both DSLs compile with the arg spec's alignment, not the
#   sample's, so specs differing only in it are the same code (see the TODO in
#   ``arg_specs.common.tensor_hash_key``).

import math
from dataclasses import dataclass
from typing import Any, TypeVar

from aot_tensor.autotune.routing import RoutingTable
from aot_tensor.compile.stable_types import TORCH_DTYPE_TO_STABLE

_T = TypeVar("_T")


@dataclass(frozen=True)
class Variant:
    """One compiled specialization of an op."""

    # C symbol prefix for everything this variant defines.
    symbol: str
    # C++ boolean expression over the op's parameters selecting this variant.
    condition: str
    # Human-readable signature for the no-match error.
    description: str


def cpp_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        raise NotImplementedError(f"non-finite float constexpr {value!r}")
    if isinstance(value, int) and not -(2**63) <= value < 2**63:
        raise NotImplementedError(f"constexpr {value} does not fit int64")
    if isinstance(value, (int, float)):
        # Exact for int64 (C++ gives an unsuffixed literal the first of
        # int/long/long long that fits) and double (repr round-trips).
        return repr(value)
    raise NotImplementedError(f"unsupported constexpr value {value!r}")


def _tensor_terms(name: str, key: dict[str, Any]) -> list[str]:
    stable_dtype = TORCH_DTYPE_TO_STABLE.get(key["dtype"])
    if stable_dtype is None:
        raise NotImplementedError(f"unsupported tensor dtype {key['dtype']} for {name}")
    terms = [f"{name}.scalar_type() == {stable_dtype}", f"{name}.dim() == {key['dim']}"]
    # After the rank term, so the index is in range.
    if key["leading_dim"] is not None:
        terms.append(f"{name}.stride({key['leading_dim']}) == 1")
    return terms


def match_condition(args: list[dict[str, Any]]) -> str:
    terms: list[str] = []
    for arg in args:
        if "tensor" in arg:
            terms += _tensor_terms(arg["name"], arg["tensor"])
        elif "constexpr" in arg:
            terms.append(f"{arg['name']} == {cpp_literal(arg['constexpr'])}")
    return " && ".join(terms) or "true"


def describe(args: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for arg in args:
        if "tensor" in arg:
            t = arg["tensor"]
            part = f"{arg['name']}: {t['dtype']} rank {t['dim']}"
            if t["leading_dim"] is not None:
                part += f" unit-stride dim {t['leading_dim']}"
            parts.append(part)
        elif "constexpr" in arg:
            parts.append(f"{arg['name']}={arg['constexpr']!r}")
    return ", ".join(parts)


def distinct_variants(
    op_name: str, specs: list[tuple[_T, str, dict[str, Any]]]
) -> list[tuple[_T, Variant]]:
    """``specs`` holds ``(spec, spec_hash, hash_key)``. Keeps the first spec per
    distinct match condition, in collection order."""
    seen: set[str] = set()
    symbols: set[str] = set()
    variants: list[tuple[_T, Variant]] = []
    for spec, spec_hash, hash_key in specs:
        condition = match_condition(hash_key["args"])
        if condition in seen:
            continue
        seen.add(condition)
        symbol = f"{op_name}__{spec_hash[:8]}"
        if symbol in symbols:
            raise RuntimeError(f"{op_name}: two variants share symbol {symbol}")
        symbols.add(symbol)
        variants.append(
            (
                spec,
                Variant(
                    symbol=symbol,
                    condition=condition,
                    description=describe(hash_key["args"]),
                ),
            )
        )
    return variants


@dataclass(frozen=True)
class Route:
    """One branch of the generated selector: a call matching ``condition`` runs
    the variant its key picks from ``table``."""

    condition: str
    description: str
    # log2 bucket per key -> variant symbol; one entry needs no key.
    table: RoutingTable[str]


def variant_routes(variants: list[Variant]) -> list[Route]:
    """One route per variant."""
    return [
        Route(v.condition, v.description, RoutingTable({(): v.symbol}))
        for v in variants
    ]


def render_selector(routes: list[Route], call: dict[str, str], fail: str) -> list[str]:
    """Statements dispatching to ``call[symbol]`` for the first matching route;
    ``fail`` is a function taking the no-match message."""
    lines: list[str] = []
    for route in routes:
        (symbol,) = route.table.entries.values()
        lines += [f"if ({route.condition}) {{", f"  {call[symbol]};", "  return;", "}"]
    compiled = "; ".join(route.description for route in routes)
    message = f"no compiled specialization matches the arguments; compiled: {compiled}"
    lines.append(f"{fail}({_quote(message)});")
    return lines


def _quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
