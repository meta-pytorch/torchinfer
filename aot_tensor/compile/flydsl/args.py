# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# torch values -> FlyDSL launcher kwargs. Eager and AOT both use this, so they
# pass the same JIT args and get the same ABI.

import functools
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aot_tensor.arg_specs.fly import FlyArgSpec, FlyTensorArg
from aot_tensor.compile.flydsl.compat import private_attr
from aot_tensor.types import FlyAOT


@functools.cache
def stream_param_name(op: FlyAOT) -> str:
    """Name of the launcher's ``fx.Stream`` param; also checks arg_specs name the
    other params in order. The stream param is required: without it FlyDSL uses
    the default stream, which races the caller's torch work."""
    launcher = op.jit_fn
    private_attr(launcher, "_ensure_sig")()
    sig = private_attr(launcher, "_sig")
    streams = [
        name
        for name, param in sig.parameters.items()
        if getattr(param.annotation, "_is_stream_param", False)
    ]
    if len(streams) != 1:
        raise ValueError(
            f"FlyAOT {op.name}: launcher must declare exactly one `fx.Stream` "
            f"parameter, found {streams}"
        )
    expected = [name for name in sig.parameters if name != streams[0]]
    declared = [spec.name for spec in op.arg_specs]
    if declared != expected:
        raise ValueError(
            f"FlyAOT {op.name}: arg_specs {declared} must name the launcher's "
            f"non-stream parameters in order {expected}"
        )
    return streams[0]


def _jit_value(spec: FlyArgSpec, value: Any) -> Any:
    if not isinstance(spec, FlyTensorArg):
        # Scalars and constexprs: the launcher annotation drives conversion.
        return value
    return flyc.from_torch_tensor(
        value, assumed_align=spec.alignment
    ).mark_layout_dynamic(leading_dim=spec.leading_dim, divisibility=spec.divisibility)


def launcher_kwargs(op: FlyAOT, values: dict[str, Any]) -> dict[str, Any]:
    kwargs = {spec.name: _jit_value(spec, values[spec.name]) for spec in op.arg_specs}
    kwargs[stream_param_name(op)] = fx.Stream(torch.cuda.current_stream())
    return kwargs
