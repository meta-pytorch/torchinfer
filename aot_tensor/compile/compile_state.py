# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional

from aot_tensor.compile.adapter_base import (
    AOTTAdapter,
    DslSpecStore,
    KernelId,
    KernelSpecs,
)
from aot_tensor.types import ALL_MARKERS

logger: logging.Logger = logging.getLogger(__name__)


class AOTTCompileState:
    """Process-wide singleton holding AOT-T compile state.

    DSL-agnostic: per-DSL state lives in ``dsl_state`` (one ``DslSpecStore`` per
    DSL, keyed by ``AOTTAdapter.name``) -- its collected kernels plus that DSL's
    eager compile cache (``DslSpecStore.eager_compiled``); the singleton itself
    holds only the shared compile path. Whether collection is active is tracked by
    marker collector registration (``is_aott_compile_enabled``).

    One instance process-wide: the compile machinery is never interned into a
    torch.package, so every call site (packaged kernels included) shares it via
    ``get_instance()``.
    """

    _instance: Optional["AOTTCompileState"] = None

    # Annotations only (no mutable class defaults). dsl_state (DSL name -> store)
    # is created in __new__ and kept process-wide; reset() clears only each
    # store's per-session kernels, keeping its adapter + eager cache.
    dsl_state: Dict[str, DslSpecStore]
    compile_base_dir: str
    # None until get_aott_compile_path() lazily mkdtemps it on first use.
    compile_path: Optional[str]

    def __new__(cls) -> "AOTTCompileState":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance.dsl_state = {}
            instance.reset()
            cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "AOTTCompileState":
        """Get the singleton instance, creating it once via ``__new__``."""
        return cls()

    def reset(self) -> None:
        """Reset per-session state: clear each store's collected kernels but keep
        the stores (their adapter + eager cache are process-lifetime). The compile
        dir is created lazily by ``get_aott_compile_path``."""
        for store in self.dsl_state.values():
            store.kernels.clear()
        # Also unregister the marker spec collectors (full reset to the
        # not-compiling state).
        for marker in ALL_MARKERS:
            marker.set_spec_collector(None)
        self.compile_base_dir = os.getenv("TRITON_AOT_PATH_PREFIX", "/var/tmp")
        self.compile_path = None


def is_aott_compile_enabled() -> bool:
    """True while AOT spec collection is active: any marker has a collector
    registered (the single source of truth, not a separate flag).
    """
    return any(m.spec_collector is not None for m in ALL_MARKERS)


def get_aott_compile_state() -> AOTTCompileState:
    """Get the process-wide AOTTCompileState singleton."""
    return AOTTCompileState.get_instance()


def get_aott_compile_path() -> str:
    """Return the compile output dir, creating it on first use. Deferred (not in
    ``reset``) so a JIT-only process never creates a temp dir.
    """
    state = get_aott_compile_state()
    if state.compile_path is None:
        state.compile_path = tempfile.mkdtemp(
            dir=state.compile_base_dir, prefix="triton_aot_compile_"
        )
    return state.compile_path


def get_kernel_specs(name: str) -> Dict[KernelId, KernelSpecs]:
    """The named DSL's collected kernels (its ``dsl_state`` slice), keyed by the
    DSL's kernel key; written by ``add_spec``. A free function (not an
    ``AOTTAdapter`` method) so the ABC never imports this state module.
    """
    return get_aott_compile_state().dsl_state[name].kernels


def add_spec(name: str, key: KernelId, spec: Any, hashed_spec: str) -> None:
    """Append ``spec`` under ``key`` in the named DSL's store, deduped by
    ``hashed_spec`` (identical signatures collapse, distinct ones accumulate).
    """
    get_aott_compile_state().dsl_state[name].add(key, spec, hashed_spec)


def register_active(dsl_cls: type[AOTTAdapter[Any]]) -> None:
    """Record ``dsl_cls`` as having collected this session (replaces an upfront
    registry). Creates the stateless DSL instance once and stores it in its
    ``DslSpecStore``; ``reset()`` clears it, so no module-level singleton.
    """
    dsl_state = get_aott_compile_state().dsl_state
    if dsl_cls.name not in dsl_state:
        dsl_state[dsl_cls.name] = DslSpecStore(dsl=dsl_cls())
