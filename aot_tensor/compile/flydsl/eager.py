# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# FlyDSL eager path, imported lazily by ``FlyAOT.__call__``. ``@flyc.jit`` keeps
# its own compile cache, so there is no AOT-T eager cache for this DSL.

from typing import Any

import torch
from aot_tensor.arg_specs.common import resolve_runtime_args
from aot_tensor.arg_specs.fly import FlyTensorArg
from aot_tensor.compile.flydsl.args import launcher_kwargs
from aot_tensor.types import FlyAOT


def _same(before: torch.Tensor, after: torch.Tensor) -> bool:
    # Compare bytes, so an unchanged NaN counts as same and fp8 works too.
    return torch.equal(
        before.flatten().view(torch.uint8), after.flatten().view(torch.uint8)
    )


def run_flydsl_jit(op: FlyAOT, *args: Any, **kwargs: Any) -> Any:
    values = resolve_runtime_args(op.arg_specs, op.name, *args, **kwargs)
    tensors: dict[str, torch.Tensor] = {
        s.name: values[s.name]
        for s in op.arg_specs
        if isinstance(s, FlyTensorArg) and values[s.name] is not None
    }
    # The launcher takes raw pointers, so a CPU tensor would be written silently.
    off_device = sorted(name for name, t in tensors.items() if not t.is_cuda)
    if off_device:
        raise RuntimeError(f"FlyAOT {op.name}: tensors {off_device} are not on GPU.")
    if type(op).spec_collector is None:
        return op.jit_fn(**launcher_kwargs(op, values))

    # While collecting, check that read-only tensors are really not written.
    read_only = {n: t.clone() for n, t in tensors.items() if n not in op.mutated}
    result = op.jit_fn(**launcher_kwargs(op, values))
    written = sorted(n for n, t in read_only.items() if not _same(t, tensors[n]))
    if written:
        raise RuntimeError(
            f"FlyAOT {op.name}: wrote {written}, which are not in mutates_args."
        )
    return result
