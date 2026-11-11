# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# Shared by the arg-spec DSLs: scalar/constexpr specs, the TensorArg base, arg
# binding and the specialization hash key.

from collections.abc import Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class TensorArg:
    """Runtime tensor argument. Holds only what DSL-agnostic code reads; each DSL
    subclasses it with its own fields. ``leading_dim``: the dim kept at stride 1
    (>= 0), ``None`` to auto-detect. Fields after ``name`` are keyword-only, so a
    positional value can't land in the wrong field of a subclass."""

    name: str
    leading_dim: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        # The spec doesn't know the rank, so -1 can't be resolved here; FlyDSL
        # would also read it as auto-detect.
        if self.leading_dim is not None and self.leading_dim < 0:
            raise RuntimeError(
                f"{type(self).__name__} {self.name}: leading_dim must be >= 0, "
                f"or None to auto-detect; got {self.leading_dim}."
            )


# All arg spec kinds.
ArgSpec = ScalarArg | ConstexprArg | TensorArg


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


def leading_dim(arg: Any, spec_leading_dim: int | None = None) -> int | None:
    """The dim whose stride the DSL's ``mark_layout_dynamic`` fixes at 1; all
    other shapes and strides stay dynamic. ``spec_leading_dim`` (the arg
    spec's ``leading_dim``) wins. Otherwise: the only stride-1 dim; if several,
    the only one with size > 1; else ``None``.

    This fixed stride is what makes the kernel fast: the compiler knows the
    elements are contiguous and uses vector loads, ``cp.async`` and TMA (as
    Triton does for an int arg equal to 1). So a call whose stride there isn't 1
    must not use that variant."""
    rank = int(arg.dim())
    if spec_leading_dim is not None:
        if not 0 <= spec_leading_dim < rank:
            raise RuntimeError(
                f"AOT-T: leading_dim {spec_leading_dim} is out of range for a "
                f"rank-{rank} tensor."
            )
        return spec_leading_dim
    unit = [d for d in range(rank) if int(arg.stride(d)) == 1]
    if len(unit) > 1:
        unit = [d for d in unit if int(arg.size(d)) > 1]
    return unit[0] if len(unit) == 1 else None


def tensor_hash_key(arg: Any, spec_leading_dim: int | None = None) -> dict[str, Any]:
    if not hasattr(arg, "dtype"):
        raise RuntimeError(
            f"AOT-T: expected a torch.Tensor-like argument, got {type(arg)!r}."
        )
    # Key on the leading dim, not on sizes or the other strides: those vary per
    # call.
    # TODO: specialize on alignment and divisibility, like Triton's
    # tt.divisibility: record the best level each sample meets (capped by the
    # arg spec) instead of aligned16, compile each variant with it, and match it
    # in spec_selector.match_condition, stricter variants first. CuTeDSL first
    # needs the TODO in compile/cutedsl/pipeline.py.
    # Unlike Triton, a lower level is not always correct: CuTe-style kernels
    # hard-code copy widths (e.g. a 128-bit BufferCopy), so the fallback must
    # be opt-in. FlyDSL's JIT cache already keys on alignment and divisibility.
    return {
        "dtype": str(arg.dtype),
        "dim": int(arg.dim()),
        "leading_dim": leading_dim(arg, spec_leading_dim),
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
    arg_specs: Sequence[ArgSpec],
    op_name: str,
    jit_module: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Specialization key of a call."""
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
        elif isinstance(spec, TensorArg):
            key = tensor_hash_key(value, spec.leading_dim)
            args.append({"name": spec.name, "tensor": key})
        else:
            raise RuntimeError(f"AOT-T {op_name}: unsupported arg spec {spec!r}.")
    return {"name": op_name, "jit_module": jit_module, "args": args}


def sample_spec_and_hash(
    arg_specs: Sequence[ArgSpec],
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
