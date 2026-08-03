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
    """Process-wide state for AOT-T compilation, holding one session at a time.

    Two lifetimes share this object, which is what ``reset`` is discriminating
    between:

    - **Process-lifetime** -- ``dsl_state`` maps DSL name to its
      ``DslSpecStore``, whose reused ``AOTTAdapter`` and ``eager_compiled``
      cache outlive any one session.
    - **Session-scoped** -- each store's collected ``kernels``, plus
      ``compile_path`` and ``session_completed``. A session runs from
      ``AOTTCompileSession.__enter__`` (which calls ``reset``) to a successful
      ``__exit__`` (which sets ``session_completed``). Whether collection is
      *currently* active is a different question, answered by
      ``is_aott_compile_enabled`` from marker registration rather than by a
      field here.

    One instance process-wide, reached through ``get_instance``: the compile
    machinery is never interned into a torch.package, so packaged kernels
    cannot be handed a reference and have to share it through the module.

    Read a finished session's output via ``assert_aott_compile_session_completed``
    rather than ``get_aott_compile_state`` -- every field of an unset state is
    indistinguishable from an empty one.
    """

    _instance: Optional["AOTTCompileState"] = None

    # Annotations only, no mutable class defaults. Note there is deliberately no
    # __init__: __new__ caches, so an __init__ would re-run on every
    # AOTTCompileState() call and wipe live state.
    dsl_state: Dict[str, DslSpecStore]
    compile_base_dir: str
    # None until get_aott_compile_path() lazily mkdtemps it on first use.
    compile_path: Optional[str]
    # True only between a successful AOTTCompileSession.__exit__ and the next
    # reset(). Consumers of the session's output gate on it via
    # assert_aott_compile_session_completed().
    session_completed: bool

    def __new__(cls) -> "AOTTCompileState":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance.dsl_state = {}
            instance.reset()
            cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "AOTTCompileState":
        """The one process-wide instance, created on first call via ``__new__``."""
        return cls()

    def reset(self) -> None:
        """Start a new session: drop everything session-scoped, keep everything
        process-lifetime.

        Clears each store's collected kernels but keeps the stores themselves,
        since their adapter and eager cache are process-lifetime. The compile
        dir is not recreated here -- ``get_aott_compile_path`` mkdtemps it
        lazily, so a JIT-only process never makes one."""
        for store in self.dsl_state.values():
            store.kernels.clear()
        # Also unregister the marker spec collectors (full reset to the
        # not-compiling state).
        for marker in ALL_MARKERS:
            marker.set_spec_collector(None)
        self.compile_base_dir = os.getenv("TRITON_AOT_PATH_PREFIX", "/var/tmp")
        self.compile_path = None
        self.session_completed = False


def is_aott_compile_enabled() -> bool:
    """True while AOT spec collection is active: any marker has a collector
    registered (the single source of truth, not a separate flag).
    """
    return any(m.spec_collector is not None for m in ALL_MARKERS)


def get_aott_compile_state() -> AOTTCompileState:
    """The process-wide ``AOTTCompileState``, in whatever phase it is in.

    For writers -- session setup, spec collection, compile. Readers of a
    finished session's output want ``assert_aott_compile_session_completed``.
    """
    return AOTTCompileState.get_instance()


def assert_aott_compile_session_completed() -> None:
    """The precondition for reading a finished session's output.

    ``get_aott_compile_state`` and ``get_aott_compile_path`` serve three roles
    -- session setup, spec collection, and reading the result -- and only the
    third has a precondition. Without this check an unset state reads as an
    empty one: ``get_aott_compile_path`` mkdtemps a fresh dir and ``dsl_state``
    is ``{}``, so wrapper codegen becomes a silent no-op and the failure only
    surfaces later as an empty ``wrapper_dict``. Fail where the precondition
    is, not three layers downstream.

    Writers (session ``__enter__``/``__exit__``, ``add_spec``,
    ``register_active``, ``compile_and_build``) run before the flag is set and
    must not call this.

    ``session_completed`` alone is sufficient -- ``__exit__`` builds its
    ``CompileContext`` from ``get_aott_compile_path()`` before setting the
    flag, so ``compile_path`` is never ``None`` once the flag is set.

    ``AssertionError`` because this is a caller wiring mistake meant to be
    caught in development, not something production reaches: it always goes
    through ``aott_lower_full``, which compiles and transforms together.
    Raised rather than ``assert``-ed so ``python -O`` cannot strip it.
    """
    if not get_aott_compile_state().session_completed:
        raise AssertionError(
            "AOTTCompileSession has not completed in this process. "
            "Possible reasons:\n"
            "  - the caller never opened a `with AOTTCompileSession():` block\n"
            "  - the compile ran in a separate process: specs live in a "
            "per-process singleton and the compile dir is a fresh mkdtemp per "
            "run, so the transform must run in the same process as the compile\n"
            "  - the session raised, so it never compiled\n"
            "  - the state was reset after the session completed"
        )


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
