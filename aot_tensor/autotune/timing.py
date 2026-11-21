# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

# Times a GPU call for autotune on the current stream. Uses torch.cuda only, so
# it serves every DSL, on CUDA and ROCm alike.

import statistics
from collections.abc import Callable, Mapping
from typing import Any

import torch

# Zeroed before each timed run so the call starts on a cold L2, as Triton's
# do_bench does; larger than the L2 of current GPUs.
FLUSH_BYTES: int = 256 * 1024 * 1024

# A DSL's call bound to the given values, ready to launch; compiled by its
# first launch at the latest.
Launcher = Callable[[dict[str, Any]], Callable[[], Any]]


def time_ms(fn: Callable[[], Any], warmup: int = 10, reps: int = 50) -> float:
    """Median time of ``fn`` over ``reps`` runs, in ms, after ``warmup`` runs."""
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for start, end in zip(starts, ends):
        flush.zero_()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )


def time_configs(
    launcher: Launcher, values: Mapping[str, Any], configs: list[dict[str, Any]]
) -> list[float]:
    """Each autotune config's time on the call bound to ``values``, in ms. The
    kernel may write any of its tensors, so the timed runs use clones and the
    call's own run sees them untouched. All configs share the clones: each
    config's repeated runs already see their own writes, so timing assumes
    speed does not depend on the values, as Triton's autotuner does.

    Unlike Triton's, a config that fails to compile or launch is not skipped:
    the error fails the call, so list only configs that run on every serving
    GPU."""
    cloned = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in values.items()
    }
    return [time_ms(launcher(cloned | config)) for config in configs]
