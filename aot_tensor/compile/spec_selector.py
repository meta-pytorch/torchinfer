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
#
# An autotuned op's autotuned constexprs are not op params: ``variant_routes``
# groups its variants on the other args, and each group picks a variant by the
# call's key.

import math
from dataclasses import dataclass
from typing import Any, TypeVar

from aot_tensor.autotune.autotune import Autotune
from aot_tensor.autotune.routing import RoutingTable
from aot_tensor.compile.stable_types import TORCH_DTYPE_TO_STABLE

_T = TypeVar("_T")
# An autotune config as a dict key: its sorted items.
FrozenConfig = tuple[tuple[str, Any], ...]


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


def variant_routes(
    op_name: str,
    variants: list[tuple[dict[str, Any], Variant]],
    autotune: Autotune | None,
) -> list[Route]:
    """``variants`` holds ``(hash_key, variant)``. One route per distinct match
    on the args the caller passes: without ``autotune`` that is one per variant;
    with it, the variants differing only in the autotuned constexprs share a
    route whose table is the op's tuned entries valid for its other args."""
    autotuned = frozenset() if autotune is None else autotune.constexpr_names
    # Match condition on the caller's args -> (those args, {config: symbol}).
    groups: dict[str, tuple[list[dict[str, Any]], dict[FrozenConfig, str]]] = {}
    for hash_key, variant in variants:
        args = hash_key["args"]
        caller = [a for a in args if a["name"] not in autotuned]
        config = {a["name"]: a["constexpr"] for a in args if a["name"] in autotuned}
        _, symbols = groups.setdefault(match_condition(caller), (caller, {}))
        symbols[frozen_config(config)] = variant.symbol
    routes: list[Route] = []
    for condition, (caller, symbols) in groups.items():
        if autotune is None:
            table = RoutingTable({(): symbols[()]})
        else:
            constexprs = {a["name"]: a["constexpr"] for a in caller if "constexpr" in a}
            routing = autotune.routing_for(constexprs, op_name)
            table = RoutingTable(
                {
                    point: symbols[frozen_config(config)]
                    for point, config in routing.entries.items()
                }
            )
        routes.append(Route(condition, describe(caller), table))
    return routes


def frozen_config(config: dict[str, Any]) -> FrozenConfig:
    return tuple(sorted(config.items()))


def key_exprs(autotune: Autotune | None) -> tuple[str, ...]:
    """C++ value of each autotune key in the op body, where tensors are bound."""
    if autotune is None:
        return ()
    return tuple(
        name if dim is None else f"{name}.size({dim})" for name, dim in autotune.key
    )


def render_selector(
    routes: list[Route], keys: tuple[str, ...], call: dict[str, str], fail: str
) -> list[str]:
    """Statements dispatching to ``call[symbol]`` for the first matching route;
    a route with several variants picks one on the C++ ``keys`` with the
    template's ``aott_nearest_entry``. ``fail`` is a function taking the
    no-match message."""
    lines: list[str] = []
    for route in routes:
        lines.append(f"if ({route.condition}) {{")
        lines += [f"  {line}" for line in route_body(route.table, keys, call)]
        lines.append("}")
    compiled = "; ".join(route.description for route in routes)
    message = f"no compiled specialization matches the arguments; compiled: {compiled}"
    lines.append(f"{fail}({_quote(message)});")
    return lines


def route_body(
    table: RoutingTable[str], keys: tuple[str, ...], call: dict[str, str]
) -> list[str]:
    symbols = list(table.entries.values())
    last = symbols[-1]
    if len(set(symbols)) == 1:
        return [f"{call[last]};", "return;"]
    entries = ", ".join(
        "{" + ", ".join(str(b) for b in point) + "}" for point in table.entries
    )
    lines = [
        f"static constexpr int64_t aott_entries[{len(symbols)}][{len(keys)}] = "
        f"{{{entries}}};",
        f"const int64_t aott_keys[] = {{{', '.join(keys)}}};",
        "switch (aott_nearest_entry(aott_entries, aott_keys)) {",
    ]
    # One case group per variant; the last entry's variant takes the rest.
    indices: dict[str, list[int]] = {}
    for i, symbol in enumerate(symbols):
        indices.setdefault(symbol, []).append(i)
    for symbol, hits in indices.items():
        if symbol != last:
            lines += [f"  case {i}:" for i in hits]
            lines += [f"    {call[symbol]};", "    return;"]
    return [*lines, "  default:", f"    {call[last]};", "    return;", "}"]


def _quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
