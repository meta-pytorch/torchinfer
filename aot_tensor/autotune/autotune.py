# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# The ``autotune=`` of ``cutedsl_aot`` and ``flydsl_aot``: the constexprs an op
# picks itself, and the call values it picks them by.

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from aot_tensor.arg_specs.common import (
    ArgSpec,
    ConstexprArg,
    resolve_runtime_args,
    ScalarArg,
    TensorArg,
)
from aot_tensor.autotune.routing import RoutingTable


# What picks an autotune config: a tensor dim ``("mA", 0)`` or an int arg's name.
AutotuneKey = str | tuple[str, int]


class Autotune:
    """Constexprs an op picks itself. The caller does not pass them: each call
    takes the ``tuned`` entry nearest to its ``key`` values (rule in
    ``aot_tensor.autotune.routing``) and uses that entry's config.

    ``tuned``: bucket per key (powers of 2; a bare int with one key) -> config,
    each config setting the same constexprs; earlier entries win ties.
    ``valid(config, constexprs)``: whether a config works with the call's other
    constexprs; entries that fail it are skipped for that call.

    E.g. ``key=["M"]``, ``tuned={1024: A, 65536: B}``: the buckets are log2 10
    and 16, so M <= 8192 (bucket <= 13; 13 is a tie, won by A) takes A and
    larger M take B."""

    def __init__(
        self,
        key: Sequence[AutotuneKey],
        tuned: Mapping[int | tuple[int, ...], Mapping[str, Any]],
        valid: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    ) -> None:
        if not key or not tuned:
            raise ValueError("Autotune: key and tuned must not be empty.")
        # (arg name, tensor dim); the dim is None for an int arg.
        self.key: tuple[tuple[str, int | None], ...] = tuple(
            key_arg_and_dim(k) for k in key
        )
        self.valid: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = (
            valid
        )
        self.routing: RoutingTable[dict[str, Any]] = RoutingTable.from_buckets(
            bucket_table(tuned, key)
        )
        # The constexprs the op fills itself.
        self.constexpr_names: frozenset[str] = config_names(
            list(self.routing.entries.values())
        )

    def check(self, arg_specs: Sequence[ArgSpec], op_name: str) -> None:
        """Reject constexprs and keys that the op's arg specs cannot back."""
        by_name = {spec.name: spec for spec in arg_specs}
        for name in sorted(self.constexpr_names):
            spec = by_name.get(name)
            if not isinstance(spec, ConstexprArg):
                raise RuntimeError(
                    f"AOT-T {op_name}: autotuned {name} must be a ConstexprArg."
                )
            for config in self.routing.entries.values():
                if not isinstance(config[name], spec.python_type):
                    raise RuntimeError(
                        f"AOT-T {op_name}: autotune config {config} sets {name} to "
                        f"a {type(config[name])}, expected {spec.python_type}."
                    )
        for name, dim in self.key:
            if not can_be_key(
                by_name.get(name), dim, autotuned=name in self.constexpr_names
            ):
                key = name if dim is None else (name, dim)
                raise RuntimeError(
                    f"AOT-T {op_name}: autotune key {key!r} must name a tensor dim "
                    f"or an int arg that is not autotuned."
                )

    def routing_for(
        self, constexprs: Mapping[str, Any], op_name: str
    ) -> RoutingTable[dict[str, Any]]:
        """The entries whose config is valid with these other constexprs."""
        valid = self.valid
        if valid is None:
            return self.routing
        routing = self.routing.filter(lambda config: valid(config, constexprs))
        if not routing.entries:
            raise RuntimeError(
                f"AOT-T {op_name}: no tuned autotune config is valid for "
                f"{dict(constexprs)}."
            )
        return routing

    def resolve(
        self, arg_specs: Sequence[ArgSpec], op_name: str, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Bind a call that leaves out the autotuned constexprs, and fill them in."""
        passed = self.constexpr_names & set(kwargs)
        if passed:
            raise RuntimeError(
                f"AOT-T {op_name}: {sorted(passed)} are autotuned; do not pass them."
            )
        caller_specs = [
            spec for spec in arg_specs if spec.name not in self.constexpr_names
        ]
        values = resolve_runtime_args(caller_specs, op_name, *args, **kwargs)
        routing = self.routing_for(self.other_constexprs(arg_specs, values), op_name)
        return values | routing.select(self.key_values(values, op_name))

    def key_values(self, values: Mapping[str, Any], op_name: str) -> list[int]:
        """Each key's value in a call bound to ``values``."""
        key_values: list[int] = []
        for name, dim in self.key:
            if dim is None:
                key_values.append(values[name])
                continue
            tensor = values[name]
            # The arg spec doesn't know the rank, so this waits for the call.
            if dim >= tensor.dim():
                raise RuntimeError(
                    f"AOT-T {op_name}: autotune key {(name, dim)!r} needs {name} of "
                    f"rank > {dim}, got {tensor.dim()}."
                )
            key_values.append(tensor.size(dim))
        return key_values

    def other_constexprs(
        self, arg_specs: Sequence[ArgSpec], values: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The constexprs of a call bound to ``values`` that are not autotuned."""
        return {
            spec.name: values[spec.name]
            for spec in arg_specs
            if isinstance(spec, ConstexprArg) and spec.name not in self.constexpr_names
        }


def key_arg_and_dim(key: AutotuneKey) -> tuple[str, int | None]:
    if isinstance(key, str):
        return key, None
    name, dim = key
    if dim < 0:
        raise ValueError(f"Autotune: key {key} needs a dim >= 0.")
    return name, dim


def can_be_key(spec: ArgSpec | None, dim: int | None, autotuned: bool) -> bool:
    """Whether ``spec`` can give a key's value: a tensor's ``dim``, or with no
    dim an int that the caller passes."""
    if dim is not None:
        return isinstance(spec, TensorArg)
    if isinstance(spec, ScalarArg):
        return spec.dtype in {"i32", "i64"}
    if isinstance(spec, ConstexprArg):
        return spec.python_type is int and not autotuned
    return False


def bucket_table(
    tuned: Mapping[int | tuple[int, ...], Mapping[str, Any]],
    key: Sequence[AutotuneKey],
) -> dict[tuple[int, ...], dict[str, Any]]:
    """``tuned`` with each entry's buckets as a tuple of one bucket per key."""
    table: dict[tuple[int, ...], dict[str, Any]] = {}
    for raw, config in tuned.items():
        buckets = (raw,) if isinstance(raw, int) else tuple(raw)
        if len(buckets) != len(key):
            raise ValueError(
                f"Autotune: tuned entry {buckets} needs one bucket per key {list(key)}."
            )
        if buckets in table:
            raise ValueError(f"Autotune: tuned lists {buckets} twice.")
        table[buckets] = dict(config)
    return table


def config_names(configs: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """The constexprs the configs set; every config must set the same ones."""
    names = frozenset(configs[0])
    for config in configs:
        if set(config) != names:
            raise ValueError(
                f"Autotune: config {dict(config)} sets {sorted(config)}, expected "
                f"{sorted(names)}."
            )
    return names
