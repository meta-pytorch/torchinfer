# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, cast, Generic, TypeVar

from torch import package


# A DSL's kernel key. ``Any`` (not ``Hashable``) so each DSL reads its keys back
# as its own concrete type without pyre casts. The key is DSL-defined: Triton
# uses the JIT ``fn``, CuTeDSL the ``CuTeAOT`` marker object.
KernelId = Any


class CustomEncoder(json.JSONEncoder):
    # pyre-ignore[14]: Inconsistent override
    def default(self, obj: object) -> Any:
        if isinstance(obj, set):
            return {"__set__": True, "items": sorted(obj)}
        return super().default(obj)


def hash_spec(spec: dict[str, Any]) -> str:
    serialized_dict = json.dumps(spec, cls=CustomEncoder, sort_keys=True)
    return hashlib.sha256(serialized_dict.encode("utf-8")).hexdigest()


@dataclass
class KernelSpecs:
    """Specs collected for one kernel, deduped by hash. ``add`` appends a spec
    only if its hash is new, so re-collecting the same signature is a no-op."""

    specs: list[Any] = field(default_factory=list)
    hashes: set[str] = field(default_factory=set)

    def add(self, spec: Any, hashed_spec: str) -> None:
        if hashed_spec not in self.hashes:
            self.hashes.add(hashed_spec)
            self.specs.append(spec)


@dataclass
class DslSpecStore:
    """One DSL's slice of ``dsl_state``: its reused ``AOTTAdapter`` instance, the
    kernels it collected (keyed by ``KernelId``), and its eager compile cache."""

    dsl: AOTTAdapter[Any]
    kernels: dict[KernelId, KernelSpecs] = field(default_factory=dict)
    # Compiled eager kernels for this DSL (empty for self-caching DSLs like
    # Triton). Process-lifetime: reset() clears ``kernels``, not this.
    eager_compiled: dict[Any, Any] = field(default_factory=dict)

    def add(self, key: KernelId, spec: Any, hashed_spec: str) -> None:
        self.kernels.setdefault(key, KernelSpecs()).add(spec, hashed_spec)


class DslCompileConfig(ABC):
    """Base for a DSL's compile-time options. Each DSL subclasses it and is the
    only code that builds/reads its own config; the shared layer sees only this
    base, so adding a DSL never touches this module.
    """


TDslConfig = TypeVar("TDslConfig", bound=DslCompileConfig)


class CompileContext:
    """Config handed to every DSL's ``compile_and_build``. Indexes a flat list of
    ``DslCompileConfig`` by type; a DSL gets its own via ``find_config``.
    """

    def __init__(
        self,
        compile_path: str,
        import_module: Callable[[str], ModuleType],
        dsl_config: Sequence[DslCompileConfig],
    ) -> None:
        self.compile_path = compile_path
        self.import_module = import_module
        # Index by concrete type: at most one config per type.
        self.type_to_config: dict[type[DslCompileConfig], DslCompileConfig] = {}
        for cfg in dsl_config:
            t = type(cfg)
            if t in self.type_to_config:
                raise ValueError(f"Duplicate DslCompileConfig of type {t.__name__}")
            self.type_to_config[t] = cfg

    def find_config(self, cls: type[TDslConfig]) -> TDslConfig | None:
        """Return the ``cls`` config, or ``None``."""
        return cast("TDslConfig | None", self.type_to_config.get(cls))


# ``TMatch`` is a DSL's match type (Triton: ``TritonAOT``;
# CuTeDSL: a ``(CuTeAOT, global_names)`` tuple)
TMatch = TypeVar("TMatch")


class AOTTAdapter(ABC, Generic[TMatch]):
    """AOT-T integration for one kernel source (Triton, CuTeDSL, ...).

    Compile and transform iterate the DSLs that collected (``dsl_state``) and
    call these methods -- no special-casing, no registry. ``TMatch`` is a DSL's
    ``find_kernel`` result, handed back to ``generate_wrapper_files``.

    Stateless: all per-compile state lives in the ``AOTTCompileState`` singleton,
    so the instance can be rebuilt on demand. Spec collection isn't here -- each
    DSL has a module-level ``collect`` the session wires to the marker.
    """

    name: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise TypeError(f"{cls.__name__} must define a class-level ``name``")

    # --- compile + build (whole per-DSL loop) ---
    @abstractmethod
    def compile_and_build(self, ctx: CompileContext) -> None:
        """Compile every collected spec to C++ and build the extension(s)."""
        ...

    # --- transform ---
    @abstractmethod
    def find_kernel(self, node_target: Any) -> TMatch | None:
        """Return this DSL's match for the kernel behind an FX ``call_function``
        target, or ``None`` if it owns none. Passed to ``generate_wrapper_files``."""
        ...

    @abstractmethod
    def generate_wrapper_files(
        self,
        node_target: Any,
        match: TMatch,
        compile_path: str,
        package_importer: package.PackageImporter | None,
    ) -> None:
        """Emit ``{fn}_original.py`` and ``{fn}_wrapper.py`` for ``match``."""
        ...
