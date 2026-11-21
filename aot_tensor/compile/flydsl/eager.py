# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# FlyDSL eager path, imported lazily by ``FlyAOT.__call__``. ``@flyc.jit`` keeps
# its own compile cache, so there is no AOT-T eager cache for this DSL.

import functools
from typing import Any, Callable

import torch
from aot_tensor.arg_specs.fly import FlyTensorArg
from aot_tensor.compile.eager_call import bind_call
from aot_tensor.compile.flydsl.args import launcher_kwargs
from aot_tensor.types import FlyAOT


def _same(before: torch.Tensor, after: torch.Tensor) -> bool:
    # Compare bytes, so an unchanged NaN counts as same and fp8 works too.
    return torch.equal(
        before.flatten().view(torch.uint8), after.flatten().view(torch.uint8)
    )


def flydsl_launcher(op: FlyAOT, values: dict[str, Any]) -> Callable[[], Any]:
    """``op``'s call bound to ``values``, ready to launch; ``@flyc.jit``
    compiles it on the first launch."""
    kwargs = launcher_kwargs(op, values)
    return lambda: op.jit_fn(**kwargs)


def run_flydsl_jit(op: FlyAOT, *args: Any, **kwargs: Any) -> Any:
    """Bind ``op``'s call (``bind_call``: autotune, collect), then run it."""
    launcher = functools.partial(flydsl_launcher, op)
    values = bind_call(op, launcher, *args, **kwargs)
    tensors: dict[str, torch.Tensor] = {
        s.name: values[s.name]
        for s in op.arg_specs
        if isinstance(s, FlyTensorArg) and values[s.name] is not None
    }
    # The launcher takes raw pointers, so a CPU tensor would be written silently.
    off_device = sorted(name for name, t in tensors.items() if not t.is_cuda)
    if off_device:
        raise RuntimeError(f"FlyAOT {op.name}: tensors {off_device} are not on GPU.")
    launch = launcher(values)
    if type(op).spec_collector is None:
        return launch()

    # While collecting, check that read-only tensors are really not written.
    read_only = {n: t.clone() for n, t in tensors.items() if n not in op.mutated}
    result = launch()
    written = sorted(n for n, t in read_only.items() if not _same(t, tensors[n]))
    if written:
        raise RuntimeError(
            f"FlyAOT {op.name}: wrote {written}, which are not in mutates_args."
        )
    return result
