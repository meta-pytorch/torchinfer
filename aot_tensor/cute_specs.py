# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from aot_tensor.compile.adapter_base import hash_spec


@dataclass(frozen=True)
class CuTeTensorArg:
    """Runtime tensor argument for CuTeDSL AOT.

    ``leading_dim`` / ``compact_mode`` / ``stride_order`` / ``divisibility`` are
    optional layout hints threaded to the ``cute.Tensor`` conversion. They mirror
    the eager
    ``from_dlpack(...).mark_layout_dynamic(leading_dim=...).mark_compact_shape_dynamic(...)``
    chain a hand-written host wrapper builds. Defaults reproduce the prior
    behavior: auto-detected unit-stride leading dim (plain ``mark_layout_dynamic()``)
    and no compact/divisibility hint.

    Set ``compact_mode`` + ``divisibility`` (and typically ``leading_dim`` +
    ``stride_order``) for vectorized kernels whose 128-bit ``cp.async`` atoms need
    a statically provable stride alignment -- without the divisibility hint the
    compiler cannot prove the runtime stride is 16B aligned and IR verification
    rejects the copy ("src ptr alignment (16 bits) does not meet requirement
    (128 bits)").
    """

    name: str
    alignment: int = 16
    leading_dim: Optional[int] = None
    compact_mode: Optional[int] = None
    stride_order: Optional[Tuple[int, ...]] = None
    divisibility: int = 1


@dataclass(frozen=True)
class CuTeScalarArg:
    """Runtime scalar argument for CuTeDSL AOT."""

    name: str
    dtype: str


@dataclass(frozen=True)
class CuTeConstexprArg:
    """Compile-time specialization argument for CuTeDSL AOT."""

    name: str
    python_type: type[Any]


CuTeArgSpec = Union[CuTeTensorArg, CuTeScalarArg, CuTeConstexprArg]


def resolve_runtime_args(
    arg_specs: list[CuTeArgSpec], op_name: str, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Bind positional/keyword call args to a {spec.name: value} dict per arg_specs.

    Pure arg-binding (no cute/cuda), shared by spec collection, the eager runner,
    and the C++ compile path.
    """
    values: dict[str, Any] = {}
    if len(args) > len(arg_specs):
        raise RuntimeError(
            f"CuTeAOT {op_name}: expected at most {len(arg_specs)} "
            f"positional args, got {len(args)}."
        )

    for spec, value in zip(arg_specs, args):
        values[spec.name] = value

    # values currently holds only the positionally-bound names (kwargs not added
    # yet), so an overlap with kwargs means the same arg was passed twice.
    duplicate = set(values) & set(kwargs)
    if duplicate:
        raise RuntimeError(
            f"CuTeAOT {op_name}: got multiple values for arg(s) {sorted(duplicate)}."
        )

    for spec in arg_specs[len(args) :]:
        if spec.name not in kwargs:
            raise RuntimeError(f"CuTeAOT {op_name}: missing arg {spec.name}.")
        values[spec.name] = kwargs[spec.name]

    unexpected = set(kwargs.keys()) - {spec.name for spec in arg_specs}
    if unexpected:
        raise RuntimeError(
            f"CuTeAOT {op_name}: unexpected kwargs {sorted(unexpected)}."
        )
    return values


def module_basename_for_callable(fn: Any) -> str:
    module = getattr(fn, "__module__", None)
    if module is None:
        module = fn.__class__.__module__
    return module.rsplit(".", 1)[-1]


def layout_order(arg: Any) -> tuple[int, ...]:
    """Memory-layout pattern: dim indices sorted by descending stride. Captures
    contiguous-vs-transposed (and higher-rank permutations) independent of the
    concrete sizes, so it can key a shape-independent specialization. Shared by
    the AOT spec key (``_tensor_hash_key``) and the eager compile-cache key so the
    two never diverge on what makes two tensors interchangeable.
    """
    strides = [int(s) for s in arg.stride()]
    return tuple(sorted(range(int(arg.dim())), key=lambda i: strides[i], reverse=True))


def _tensor_hash_key(arg: Any) -> dict[str, Any]:
    if not hasattr(arg, "dtype"):
        raise RuntimeError(
            f"CuTeAOT: expected a torch.Tensor-like argument, got {type(arg)!r}."
        )
    # cutedsl AOT marks tensor layouts dynamic (cute_arg -> mark_layout_dynamic):
    # one compiled .so serves any shape of a given dtype/rank/alignment/layout.
    # The specialization key must therefore NOT include concrete shape/stride
    # values -- otherwise an op invoked with N runtime shapes yields N distinct
    # "specializations" and compile_cutedsl_to_cpp rejects len(specs) > 1. Key on
    # the layout *pattern* (dim order by descending stride) instead of raw sizes.
    return {
        "dtype": str(arg.dtype),
        "dim": int(arg.dim()),
        "layout_order": layout_order(arg),
        "aligned16": bool(arg.data_ptr() % 16 == 0),
    }


def _scalar_hash_key(arg: Any, dtype: str) -> dict[str, Any]:
    if dtype in {"i32", "i64"} and not isinstance(arg, int):
        raise RuntimeError(
            f"CuTeAOT: expected int scalar for dtype {dtype}, got {type(arg)!r}."
        )
    if dtype == "fp32" and not isinstance(arg, (float, int)):
        raise RuntimeError(
            f"CuTeAOT: expected float scalar for dtype {dtype}, got {type(arg)!r}."
        )
    if dtype == "bool" and not isinstance(arg, bool):
        raise RuntimeError(
            f"CuTeAOT: expected bool scalar for dtype {dtype}, got {type(arg)!r}."
        )
    return {"dtype": dtype}


def _arg_hashes(
    arg_specs: list[CuTeArgSpec], op_name: str, values: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-arg specialization hash entries from already-resolved ``values``."""
    arg_hashes: list[dict[str, Any]] = []
    for spec in arg_specs:
        value = values[spec.name]
        if isinstance(spec, CuTeTensorArg):
            arg_hashes.append({"name": spec.name, "tensor": _tensor_hash_key(value)})
        elif isinstance(spec, CuTeScalarArg):
            arg_hashes.append(
                {"name": spec.name, "scalar": _scalar_hash_key(value, spec.dtype)}
            )
        elif isinstance(spec, CuTeConstexprArg):
            if not isinstance(value, spec.python_type):
                raise RuntimeError(
                    f"CuTeAOT {op_name}: constexpr arg {spec.name} "
                    f"expected {spec.python_type}, got {type(value)}."
                )
            arg_hashes.append({"name": spec.name, "constexpr": value})
    return arg_hashes


@dataclass
class CuTeRawSpec:
    """A sampled CuTeDSL AOT specialization."""

    runtime_args: tuple[Any, ...]
    runtime_kwargs: dict[str, Any]
    hash_key: dict[str, Any]

    @classmethod
    def from_call(
        cls,
        arg_specs: list[CuTeArgSpec],
        op_name: str,
        jit_module: str,
        *args: Any,
        **kwargs: Any,
    ) -> CuTeRawSpec:
        """Sample a specialization from a concrete call (args validated per spec)."""
        values = resolve_runtime_args(arg_specs, op_name, *args, **kwargs)
        return cls(
            runtime_args=tuple(args),
            runtime_kwargs=dict(kwargs),
            hash_key={
                "name": op_name,
                "jit_module": jit_module,
                "args": _arg_hashes(arg_specs, op_name, values),
            },
        )


def sample_spec_and_hash(
    arg_specs: list[CuTeArgSpec],
    op_name: str,
    jit_module: str,
    *args: Any,
    **kwargs: Any,
) -> tuple[CuTeRawSpec, str]:
    """Sample a specialization and its stable hash — the single specialization
    key shared by the AOT spec collector and the eager compile cache, so the two
    can't diverge on what makes two calls interchangeable.
    """
    spec = CuTeRawSpec.from_call(arg_specs, op_name, jit_module, *args, **kwargs)
    return spec, hash_spec(spec.hash_key)


def spec_hash_from_values(
    arg_specs: list[CuTeArgSpec],
    op_name: str,
    jit_module: str,
    values: dict[str, Any],
) -> str:
    """Specialization hash from already-resolved ``values`` -- same key as
    ``sample_spec_and_hash`` but without re-binding args, for the eager hot path
    (which already resolved them to build the call)."""
    return hash_spec(
        {
            "name": op_name,
            "jit_module": jit_module,
            "args": _arg_hashes(arg_specs, op_name, values),
        }
    )
