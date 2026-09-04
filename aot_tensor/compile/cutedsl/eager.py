# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# CuTeDSL AOT eager path: JIT-compile (via a caller-injected compile fn) and run
# the kernel. Imported lazily by CuTeAOT.__call__. Has no GPU imports of its own,
# so it stays importable without a GPU (CPU tests import the cache-key helper).

from typing import Any, Callable

from aot_tensor.compile.cutedsl.adapter import get_cutedsl_eager_cache
from aot_tensor.compile.cutedsl.codegen import build_cute_call_args
from aot_tensor.cute_specs import (
    CuTeArgSpec,
    module_basename_for_callable,
    resolve_runtime_args,
    spec_hash_from_values,
)


def _compile_cache_key(
    jit_fn: Any,
    arg_specs: list[CuTeArgSpec],
    op_name: str,
    values: dict[str, Any],
) -> str:
    """Eager cache key == the AOT specialization hash for this call, from
    already-resolved ``values`` so a cached eager kernel matches what the lowered
    ``.so`` specializes to."""
    return spec_hash_from_values(
        arg_specs, op_name, module_basename_for_callable(jit_fn), values
    )


def run_cutedsl_jit(
    compile_fn: Callable[..., Any],
    jit_fn: Any,
    arg_specs: list[CuTeArgSpec],
    op_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """JIT-compile (via ``compile_fn``, e.g. ``cutlass.cute.compile``) and run the
    kernel. ``compile_fn`` is injected by the caller so this module needs no
    GPU-only cutlass import of its own."""
    values = resolve_runtime_args(arg_specs, op_name, *args, **kwargs)
    cute_args = build_cute_call_args(arg_specs, values)
    key = _compile_cache_key(jit_fn, arg_specs, op_name, values)
    cache = get_cutedsl_eager_cache()
    compiled = cache.get(key)
    if compiled is None:
        compiled = compile_fn(jit_fn, *cute_args)
        cache[key] = compiled
    return compiled(*cute_args)
