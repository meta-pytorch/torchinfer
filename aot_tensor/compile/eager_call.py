# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

# The eager call path every DSL shares: bind a marker's call, autotune it, and
# record its spec while collecting. Each DSL's eager runner starts with
# ``bind_call``.

import functools
from typing import Any

from aot_tensor.arg_specs.common import resolve_runtime_args
from aot_tensor.autotune.autotune import Tuner
from aot_tensor.autotune.timing import Launcher, time_configs
from aot_tensor.compile.spec_selector import route_group
from aot_tensor.types import CuTeAOT, FlyAOT


def bind_call(
    op: CuTeAOT | FlyAOT, launcher: Launcher, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Bind one call of ``op``, autotuned if ``op`` has a tuner, and record its
    spec while collecting. Returns the call's values."""
    if op.tuner is None:
        values = resolve_runtime_args(op.arg_specs, op.name, *args, **kwargs)
    else:
        values = autotune_call(op, op.tuner, launcher, *args, **kwargs)
    # Read via the class (see ``TritonAOT.run``) to avoid double-binding.
    collector = type(op).spec_collector
    if collector is not None:
        collector(op, **values)
    return values


def autotune_call(
    op: CuTeAOT | FlyAOT,
    tuner: Tuner,
    launcher: Launcher,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Autotune one call of ``op``: bind its args and pick the config for its
    route group and key buckets, timing the candidates with ``launcher`` while
    collecting. Returns the args with the config filled in."""
    # 1. Bind the args the caller passed.
    values = tuner.bind(op.arg_specs, op.name, *args, **kwargs)
    # 2. The C++ route this call lands on.
    group = route_group(op.arg_specs, op.name, values, tuner.constexpr_names)
    collecting = type(op).spec_collector is not None
    time = functools.partial(time_configs, launcher) if collecting else None
    # 3. The config from that route's routing table; timed while collecting.
    return values | tuner.best_config(group, op.arg_specs, values, op.name, time)
