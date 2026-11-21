# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# CuTeDSL AOT eager path: JIT-compile (via a caller-injected compile fn) and run
# the kernel. Imported lazily by CuTeAOT.__call__. Has no GPU imports of its own,
# so it stays importable without a GPU (CPU tests import the cache-key helper).

import functools
from typing import Any, Callable

from aot_tensor.arg_specs.common import module_basename_for_callable, spec_hash_key
from aot_tensor.arg_specs.cute import CuTeArgSpec
from aot_tensor.compile.adapter_base import hash_spec
from aot_tensor.compile.cutedsl.adapter import get_cutedsl_eager_cache
from aot_tensor.compile.cutedsl.codegen import build_cute_call_args, make_codegens
from aot_tensor.compile.eager_call import bind_call
from aot_tensor.types import CuTeAOT


def _compile_cache_key(
    jit_fn: Any,
    arg_specs: list[CuTeArgSpec],
    op_name: str,
    values: dict[str, Any],
) -> str:
    """Eager cache key == the AOT specialization hash for this call, from
    already-resolved ``values`` so a cached eager kernel matches what the lowered
    ``.so`` specializes to."""
    return hash_spec(
        spec_hash_key(arg_specs, op_name, module_basename_for_callable(jit_fn), values)
    )


def cutedsl_launcher(
    compile_fn: Callable[..., Any], op: CuTeAOT, values: dict[str, Any]
) -> Callable[[], Any]:
    """``op``'s call bound to ``values``, JIT-compiled (via ``compile_fn``, e.g.
    ``cutlass.cute.compile``) and ready to launch."""
    cute_args = build_cute_call_args(op.arg_specs, values)
    key = _compile_cache_key(op.jit_fn, op.arg_specs, op.name, values)
    cache = get_cutedsl_eager_cache()
    compiled = cache.get(key)
    if compiled is None:
        compiled = compile_fn(op.jit_fn, *cute_args)
        cache[key] = compiled
    # The compiled function takes only the runtime args (constexprs are baked in);
    # the stream is the last of cute_args.
    runtime_args = [
        arg for cg, arg in zip(make_codegens(op.arg_specs), cute_args) if cg.is_runtime
    ]
    return lambda: compiled(*runtime_args, cute_args[-1])


def run_cutedsl_jit(
    compile_fn: Callable[..., Any], op: CuTeAOT, *args: Any, **kwargs: Any
) -> Any:
    """Bind ``op``'s call (``bind_call``: autotune, collect), then JIT-compile and
    run the kernel. ``compile_fn`` is injected by the caller so this module needs
    no GPU-only cutlass import of its own."""
    launcher = functools.partial(cutedsl_launcher, compile_fn, op)
    return launcher(bind_call(op, launcher, *args, **kwargs))()
