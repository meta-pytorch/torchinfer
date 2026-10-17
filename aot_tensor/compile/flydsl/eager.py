# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# FlyDSL eager path, imported lazily by ``FlyAOT.__call__``. ``@flyc.jit`` keeps
# its own compile cache, so there is no AOT-T eager cache for this DSL.

from typing import Any

from aot_tensor.arg_specs.common import resolve_runtime_args
from aot_tensor.compile.flydsl.args import launcher_kwargs
from aot_tensor.types import FlyAOT


def run_flydsl_jit(op: FlyAOT, *args: Any, **kwargs: Any) -> Any:
    values = resolve_runtime_args(op.arg_specs, op.name, *args, **kwargs)
    return op.jit_fn(**launcher_kwargs(op, values))
