# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# Shared by the arg-spec DSLs: scalar/constexpr specs, arg binding and the
# specialization hash key. Each DSL only defines its own tensor spec.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from aot_tensor.compile.adapter_base import hash_spec


class NamedArgSpec(Protocol):
    @property
    def name(self) -> str: ...


@dataclass(frozen=True)
class ScalarArg:
    """Runtime scalar argument, e.g. ``dtype="i32"``."""

    name: str
    dtype: str


@dataclass(frozen=True)
class ConstexprArg:
    """Compile-time argument, baked into the .so."""

    name: str
    python_type: type[Any]


@dataclass
class RawSpec:
    """A sampled specialization: the call's args and its hash key."""

    runtime_args: tuple[Any, ...]
    runtime_kwargs: dict[str, Any]
    hash_key: dict[str, Any]


def resolve_runtime_args(
    arg_specs: Sequence[NamedArgSpec], op_name: str, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Bind call args to ``{spec.name: value}``."""
    values: dict[str, Any] = {}
    if len(args) > len(arg_specs):
        raise RuntimeError(
            f"AOT-T {op_name}: expected at most {len(arg_specs)} "
            f"positional args, got {len(args)}."
        )

    for spec, value in zip(arg_specs, args):
        values[spec.name] = value

    # values holds only positional args here, so an overlap means passed twice.
    duplicate = set(values) & set(kwargs)
    if duplicate:
        raise RuntimeError(
            f"AOT-T {op_name}: got multiple values for arg(s) {sorted(duplicate)}."
        )

    for spec in arg_specs[len(args) :]:
        if spec.name not in kwargs:
            raise RuntimeError(f"AOT-T {op_name}: missing arg {spec.name}.")
        values[spec.name] = kwargs[spec.name]

    unexpected = set(kwargs.keys()) - {spec.name for spec in arg_specs}
    if unexpected:
        raise RuntimeError(f"AOT-T {op_name}: unexpected kwargs {sorted(unexpected)}.")
    return values


def module_basename_for_callable(fn: Any) -> str:
    module = getattr(fn, "__module__", None)
    if module is None:
        module = fn.__class__.__module__
    return module.rsplit(".", 1)[-1]


def layout_order(arg: Any) -> tuple[int, ...]:
    """Dim indices sorted by descending stride: contiguous vs transposed,
    regardless of sizes. Both the AOT spec key and the eager cache key use it."""
    strides = [int(s) for s in arg.stride()]
    return tuple(sorted(range(int(arg.dim())), key=lambda i: strides[i], reverse=True))


def tensor_hash_key(arg: Any) -> dict[str, Any]:
    if not hasattr(arg, "dtype"):
        raise RuntimeError(
            f"AOT-T: expected a torch.Tensor-like argument, got {type(arg)!r}."
        )
    # Layouts are marked dynamic, so one .so serves any shape. Key on the layout
    # pattern, not sizes, or N shapes would become N specializations.
    return {
        "dtype": str(arg.dtype),
        "dim": int(arg.dim()),
        "layout_order": layout_order(arg),
        "aligned16": bool(arg.data_ptr() % 16 == 0),
    }


def scalar_hash_key(arg: Any, dtype: str) -> dict[str, Any]:
    if dtype in {"i32", "i64"} and not isinstance(arg, int):
        raise RuntimeError(
            f"AOT-T: expected int scalar for dtype {dtype}, got {type(arg)!r}."
        )
    if dtype == "fp32" and not isinstance(arg, (float, int)):
        raise RuntimeError(
            f"AOT-T: expected float scalar for dtype {dtype}, got {type(arg)!r}."
        )
    if dtype == "bool" and not isinstance(arg, bool):
        raise RuntimeError(
            f"AOT-T: expected bool scalar for dtype {dtype}, got {type(arg)!r}."
        )
    return {"dtype": dtype}


def spec_hash_key(
    arg_specs: Sequence[NamedArgSpec],
    op_name: str,
    jit_module: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Specialization key of a call. Specs other than scalar/constexpr are tensors."""
    args: list[dict[str, Any]] = []
    for spec in arg_specs:
        value = values[spec.name]
        if isinstance(spec, ScalarArg):
            args.append(
                {"name": spec.name, "scalar": scalar_hash_key(value, spec.dtype)}
            )
        elif isinstance(spec, ConstexprArg):
            if not isinstance(value, spec.python_type):
                raise RuntimeError(
                    f"AOT-T {op_name}: constexpr arg {spec.name} "
                    f"expected {spec.python_type}, got {type(value)}."
                )
            args.append({"name": spec.name, "constexpr": value})
        else:
            args.append({"name": spec.name, "tensor": tensor_hash_key(value)})
    return {"name": op_name, "jit_module": jit_module, "args": args}


def sample_spec_and_hash(
    arg_specs: Sequence[NamedArgSpec],
    op_name: str,
    jit_module: str,
    *args: Any,
    **kwargs: Any,
) -> tuple[RawSpec, str]:
    values = resolve_runtime_args(arg_specs, op_name, *args, **kwargs)
    spec = RawSpec(
        runtime_args=tuple(args),
        runtime_kwargs=dict(kwargs),
        hash_key=spec_hash_key(arg_specs, op_name, jit_module, values),
    )
    return spec, hash_spec(spec.hash_key)
