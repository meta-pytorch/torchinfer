# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import importlib
import logging
from collections.abc import Sequence
from types import ModuleType, TracebackType
from typing import Any, Callable, Optional, Type

from aot_tensor.compile.adapter_base import CompileContext, DslCompileConfig
from aot_tensor.compile.compile_state import (
    get_aott_compile_path,
    get_aott_compile_state,
)
from aot_tensor.types import ALL_MARKERS, AOTTMarker, CuTeAOT, SpecCollector, TritonAOT
from torch import package

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec-collection wiring (the composition root). Kept here -- not in
# ``compile_state`` -- because mapping each marker to its DSL's ``collect`` means
# naming the plugin modules, and the plugins depend on ``compile_state`` (not the
# reverse). Wiring above them keeps the graph acyclic.
# ---------------------------------------------------------------------------
def _triton_spec_collector(marker: TritonAOT, *args: Any, **kwargs: Any) -> None:
    """Forward a Triton marker call to the Triton plugin's ``collect``. Lazy
    import so this module never statically depends on the DSL (no cycle)."""
    from aot_tensor.compile.triton.adapter import collect  # @manual  # pyre-ignore[21]

    collect(marker, *args, **kwargs)


def _cutedsl_spec_collector(marker: CuTeAOT, *args: Any, **kwargs: Any) -> None:
    """Forward a CuTeDSL marker call to the CuTeDSL plugin's ``collect`` (lazy,
    see ``_triton_spec_collector``; also keeps cutlass off the import path)."""
    from aot_tensor.compile.cutedsl.adapter import collect  # @manual  # pyre-ignore[21]

    collect(marker, *args, **kwargs)


# Marker type -> its spec collector. The one place to edit when adding a DSL.
_SPEC_COLLECTORS: dict[type[AOTTMarker], SpecCollector] = {
    TritonAOT: _triton_spec_collector,
    CuTeAOT: _cutedsl_spec_collector,
}

# Keep in lockstep with ALL_MARKERS: a marker missing here would silently get no
# collector (its DSL collects nothing). Fail fast at import.
assert set(_SPEC_COLLECTORS) == set(ALL_MARKERS), (
    "_SPEC_COLLECTORS out of sync with ALL_MARKERS: "
    f"{set(_SPEC_COLLECTORS) ^ set(ALL_MARKERS)}"
)


def enable_spec_collection() -> None:
    """Start AOT spec collection: register each marker's collector so marker calls
    forward to the owning DSL's ``collect``. No upfront registry needed.
    """
    for marker, collector in _SPEC_COLLECTORS.items():
        marker.set_spec_collector(collector)


def disable_spec_collection() -> None:
    """Stop AOT spec collection: clear every marker's collector (marker calls
    fall back to the plain JIT path)."""
    for marker in ALL_MARKERS:
        marker.set_spec_collector(None)


class AOTTCompileSession:
    """Context manager that compiles AOT-T kernels to C++ and builds the shared
    libraries (cached in a temp dir).

    DSL-agnostic: ``__enter__`` resets + enables collection; ``__exit__`` hands
    each DSL that collected a ``CompileContext`` to compile + build its own
    kernels. The DSL set comes from collection (``dsl_state``), not a registry.

    - package_importer: torch.package importer for kernel source (else importlib).
    - dsl_config: flat list of per-DSL ``DslCompileConfig``, opaque to the session
      -- each DSL finds its own by type via ``ctx.find_config`` (e.g. Triton's
      autotune-cache override path).
    """

    def __init__(
        self,
        package_importer: Optional[package.PackageImporter] = None,
        dsl_config: Optional[Sequence[DslCompileConfig]] = None,
    ) -> None:
        self._import_module: Callable[[str], ModuleType] = (
            package_importer.import_module
            if package_importer is not None
            else importlib.import_module
        )
        self._dsl_config: Sequence[DslCompileConfig] = dsl_config or ()

    def __enter__(self) -> None:
        state = get_aott_compile_state()
        state.reset()
        enable_spec_collection()
        logger.info(f"Start AOTT compile, output dir: {get_aott_compile_path()}")

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        try:
            # Skip compile if the body raised, so a build error can't mask the
            # original exception.
            if exc_type is None:
                ctx = CompileContext(
                    compile_path=get_aott_compile_path(),
                    import_module=self._import_module,
                    dsl_config=self._dsl_config,
                )
                for store in get_aott_compile_state().dsl_state.values():
                    store.dsl.compile_and_build(ctx)
                # Set last: only a session that built everything counts as
                # completed, so assert_aott_compile_session_completed() cannot pass
                # on a half-built compile dir.
                get_aott_compile_state().session_completed = True
        finally:
            # Always clear the marker collectors, even if a build raises, so
            # later JIT-only calls don't keep collecting into ``dsl_state``.
            disable_spec_collection()
