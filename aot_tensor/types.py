# Copyright (c) Meta Platforms, Inc. and affiliates.


from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    cast,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
)

import torch

# @manual=//triton:triton
import triton
from aot_tensor.arg_specs.common import module_basename_for_callable
from aot_tensor.arg_specs.cute import CuTeArgSpec
from aot_tensor.arg_specs.fly import FlyArgSpec, FlyTensorArg
from aot_tensor.compile.triton.utils import is_autotuner, is_heuristics
from triton.compiler import make_backend
from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.driver import driver
from triton.runtime.jit import KernelInterface


_VALID_HINTS: frozenset[int] = frozenset({1, 8, 16})
_VALID_POINTER_HINTS: frozenset[int] = frozenset({16})

# The Triton backend options a TritonAOT kernel may set: numerics options that
# the JIT launch and the AOT compile apply the same way. AOT-T sets the launch
# geometry per spec and the target per compile, and the JIT derives other
# options from some of the rest (``debug`` turns on ``sanitize_overflow``),
# which the AOT compile does not.
_SUPPORTED_COMPILE_OPTIONS: frozenset[str] = frozenset(
    {"enable_fp_fusion", "enable_reflect_ftz"}
)


def _keywords_the_kernel_binds(fn: Any) -> set[str]:
    """Launch keywords the kernel binds itself: its parameters, and the keys
    its autotune Configs and heuristics pass to the JIT."""
    bound: set[str] = set()
    while isinstance(fn, KernelInterface):
        bound.update(getattr(fn, "arg_names", ()))
        if is_autotuner(fn):
            bound.update(
                k for cfg in cast(Autotuner, fn).configs for k in cfg.all_kwargs()
            )
        elif is_heuristics(fn):
            bound.update(cast(Heuristics, fn).values)
        fn = getattr(fn, "fn", None)
    return bound


@dataclass(frozen=True)
class AnnotationHint:
    """Annotation with a value hint (dtype + divisibility/alignment).

    Valid hints: 16 (divisible_by_16), 8 (divisible_by_8), 1 (equal_to_1).
    For pointers (dtype starts with ``*``), only 16 is valid — other values
    would cause incorrect codegen (e.g. alignment=1 folds the pointer as a
    constexpr constant, causing a segfault at launch).
    """

    dtype: str
    hint: int

    def __post_init__(self) -> None:
        if self.hint not in _VALID_HINTS:
            raise RuntimeError(
                f"TritonAOT: invalid annotation hint {self.hint!r} for "
                f"dtype {self.dtype!r}. Valid hints: {sorted(_VALID_HINTS)}."
            )
        if self.dtype.startswith("*") and self.hint not in _VALID_POINTER_HINTS:
            raise RuntimeError(
                f"TritonAOT: invalid pointer alignment {self.hint!r} for "
                f"dtype {self.dtype!r}. Pointer annotations only support "
                f"alignment={sorted(_VALID_POINTER_HINTS)}."
            )

    def to_tuple(self) -> tuple[str, int]:
        """Convert to plain tuple for raw spec format."""
        return (self.dtype, self.hint)


# Internal annotation type (after normalization).
Annotation = Union[str, AnnotationHint]

# User-facing input type (also accepts raw tuples).
AnnotationInput = Union[str, tuple[str, int], AnnotationHint]


def _normalize_annotation(ann: AnnotationInput) -> Annotation:
    """Convert a raw tuple to AnnotationHint (triggers validation)."""
    if isinstance(ann, AnnotationHint):
        return ann
    if isinstance(ann, tuple):
        return AnnotationHint(ann[0], ann[1])
    return ann


def _default_cutedsl_op_name(jit_fn: Any) -> str:
    # Derive the op name from the kernel itself (like TritonAOT uses the jit fn name).
    name = getattr(jit_fn, "__name__", None) or type(jit_fn).__name__
    return f"_cutedsl_{name}"


logger: logging.Logger = logging.getLogger(__name__)


# Spec-collection hook: called with the marker instance + the call's runtime
# args. Registered on the marker classes by the compile session.
SpecCollector = Callable[..., None]


class AOTTMarkerMeta(type):
    """Kernel-marker metaclass: gives each concrete marker its own ``_instances``
    list (so ``TritonAOT`` / ``CuTeAOT`` stay independent) and records every
    instance created.
    """

    def __init__(
        cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> None:
        super().__init__(name, bases, attrs)
        cls._instances: list[Any] = []

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        cls._instances.append(instance)
        return instance

    def get_instances(cls) -> list[Any]:
        return cls._instances


class AOTTMarker:
    """DSL-agnostic kernel marker base.

    Wraps a kernel; while a compile is active it forwards each call to the
    registered ``spec_collector`` to record a spec, else it's a transparent
    pass-through (normal JIT). The collector is injected by the compile session,
    inverting the dependency so ``types`` never imports the DSL adapters.
    """

    spec_collector: ClassVar[Optional[SpecCollector]] = None

    @classmethod
    def set_spec_collector(cls, collector: Optional[SpecCollector]) -> None:
        """Register (``None`` clears) this marker class's collector. Set on the
        concrete ``cls`` so ``TritonAOT`` / ``CuTeAOT`` stay independent."""
        cls.spec_collector = collector


class TritonAOT(
    KernelInterface[Callable[..., Any]], AOTTMarker, metaclass=AOTTMarkerMeta
):
    """Wraps a Triton kernel for ahead-of-time compilation.

    Annotations specify dtype and optional value hints for kernel parameters:

    - Scalar:  ``"i32"``, ``"fp32"``, or ``AnnotationHint("i32", 16)``
      where 16 means the runtime value is divisible by 16.
    - Pointer: ``AnnotationHint("*fp32", 16)`` for 16-byte aligned tensors.
      Only alignment=16 is valid for pointers.
    - Tensor:  typically inferred from runtime ``torch.Tensor.dtype``.
    - Optional tensor:  auto-detected when the same kernel is called
      with a tensor at one site and ``None`` at another.

    ``compile_options`` are Triton backend options for this kernel:
    ``enable_fp_fusion`` and ``enable_reflect_ftz``. The JIT launch and the AOT
    compile both get them, so the cubin computes what the JIT run did. Each
    backend gets only the options it defines, so one kernel can carry options
    for several. A launch may repeat a declared option but not change it (the
    wrapper rewrite accepts a repeat only as a literal), and the kernel may not
    bind the same name itself, as a parameter or through a ``triton.Config`` or
    ``triton.heuristics``. Every marker that launches one kernel must declare
    the same options; spec collection, ``compile_and_build`` and wrapper
    generation check this. The first launch with options under a Triton driver
    class looks up that backend's options, which dynamo cannot trace.
    """

    def __init__(
        self,
        fn: KernelInterface[Callable[..., Any]],
        annotations: Dict[str, AnnotationInput],
        compile_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.fn: KernelInterface[Callable[..., Any]] = fn
        self.annotations: Dict[str, Annotation] = {
            k: _normalize_annotation(v) for k, v in annotations.items()
        }
        # A dict: ``run`` branches on it, and dynamo takes an empty
        # MappingProxyType for true.
        self._compile_options: Dict[str, Any] = dict(compile_options or {})
        if self._compile_options:
            unsupported = sorted(
                self._compile_options.keys() - _SUPPORTED_COMPILE_OPTIONS
            )
            if unsupported:
                raise RuntimeError(
                    f"TritonAOT: compile_options cannot set {unsupported}; only "
                    f"{sorted(_SUPPORTED_COMPILE_OPTIONS)} reach the JIT launch "
                    f"and the AOT compile the same way."
                )
            bound = sorted(
                self._compile_options.keys() & _keywords_the_kernel_binds(fn)
            )
            if bound:
                raise RuntimeError(
                    f"TritonAOT: compile_options cannot set {bound}, which the "
                    f"kernel binds itself as a parameter, a triton.Config key or "
                    f"a triton.heuristics key."
                )
        # Keyed by the active Triton driver's class, which fixes the backend.
        self._launch_options: Dict[type, Dict[str, Any]] = {}

    @property
    def compile_options(self) -> Mapping[str, Any]:
        return MappingProxyType(self._compile_options)

    def _active_backend_options(self) -> Dict[str, Any]:
        # The JIT raises on an option its backend does not define, while
        # ``triton.compiler.compile`` drops it, so only launches filter.
        active = driver.active
        options = self._launch_options.get(type(active))
        if options is None:
            backend = make_backend(active.get_current_target())
            known = backend.parse_options({}).__dict__
            options = {k: v for k, v in self._compile_options.items() if k in known}
            self._launch_options[type(active)] = options
        return options

    def _without_declared_options(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # The AOT compile sees only the declaration, so a launch may repeat a
        # declared option but not change it. The repeat is dropped: the
        # collector binds every launch keyword to a kernel parameter.
        repeated = self._compile_options.keys() & kwargs.keys()
        if not repeated:
            return kwargs
        changed = sorted(k for k in repeated if kwargs[k] != self._compile_options[k])
        if changed:
            raise RuntimeError(
                "TritonAOT: the launch sets "
                + ", ".join(f"{k}={kwargs[k]!r}" for k in changed)
                + " but compile_options declare "
                + ", ".join(f"{k}={self._compile_options[k]!r}" for k in changed)
                + ", which the AOT compile uses."
            )
        return {k: v for k, v in kwargs.items() if k not in repeated}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if self._compile_options:
            kwargs = self._without_declared_options(kwargs)
        # Read via the class: a function stored as a class attr would otherwise
        # bind ``self`` twice on instance access.
        collector = type(self).spec_collector
        if collector is not None:
            collector(self, *args, **kwargs)
        if self._compile_options:
            # After the collector, which binds every launch keyword to a kernel
            # parameter.
            kwargs = {**self._active_backend_options(), **kwargs}
        return self.fn.run(*args, **kwargs)


class CuTeAOT(AOTTMarker, metaclass=AOTTMarkerMeta):
    """Wraps a CuTeDSL callable for AOT collection and wrapper rewriting."""

    def __init__(
        self,
        jit_fn: Callable[..., Any],
        arg_specs: list[CuTeArgSpec],
    ) -> None:
        self.name: str = _default_cutedsl_op_name(jit_fn)
        self.jit_fn: Callable[..., Any] = jit_fn
        self.arg_specs: list[CuTeArgSpec] = arg_specs
        self.module_basename: str = module_basename_for_callable(jit_fn)

        if not self.name or not self.name.replace("_", "").isalnum():
            raise RuntimeError(
                f"CuTeAOT: derived invalid op name {self.name!r} from "
                f"kernel {type(jit_fn).__name__!r}."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Read via the class (see ``TritonAOT.run``) to avoid double-binding.
        collector = type(self).spec_collector
        if collector is not None:
            collector(self, *args, **kwargs)

        # Lazy imports: `eager` breaks a types<->cutedsl.eager cycle; cutlass is a
        # stub-less GPU-only dep, injected here so eager.py needs no GPU import.
        import cutlass.cute as cute  # @manual  # pyre-ignore[21]: no stubs
        from aot_tensor.compile.cutedsl import (  # @manual  # pyre-ignore[21]
            eager as cutedsl_eager,
        )

        return cutedsl_eager.run_cutedsl_jit(
            cute.compile, self.jit_fn, self.arg_specs, self.name, *args, **kwargs
        )


class FlyAOT(AOTTMarker, metaclass=AOTTMarkerMeta):
    """Wraps a ``@flyc.jit`` launcher for AOT collection and wrapper rewriting.

    ``arg_specs`` lists every launcher parameter except the ``fx.Stream`` one,
    which is always filled with the current stream (eager and AOT alike).

    ``mutates_args``: the tensors the launcher writes (default: all). The rest
    are read-only in the op schema, so torch.export does not turn weights
    passed to them into extra outputs.
    """

    def __init__(
        self,
        jit_fn: Any,
        arg_specs: list[FlyArgSpec],
        mutates_args: Iterable[str] | Literal["unknown"] = "unknown",
    ) -> None:
        # A @flyc.jit JitFunction keeps the user function on ``.func``.
        fn: Any = jit_fn.func
        self.name: str = f"_flydsl_{fn.__name__}"
        self.jit_fn: Any = jit_fn
        self.arg_specs: list[FlyArgSpec] = arg_specs
        self.module_basename: str = module_basename_for_callable(fn)

        if not self.name.replace("_", "").isalnum():
            raise RuntimeError(
                f"FlyAOT: derived invalid op name {self.name!r} from launcher."
            )
        tensors = {s.name for s in arg_specs if isinstance(s, FlyTensorArg)}
        self.mutated: frozenset[str] = frozenset(
            tensors if mutates_args == "unknown" else mutates_args
        )
        if not self.mutated <= tensors:
            raise RuntimeError(
                f"FlyAOT {self.name}: mutates_args {sorted(self.mutated - tensors)} "
                f"are not tensor args {sorted(tensors)}."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        collector = type(self).spec_collector
        if collector is not None:
            collector(self, *args, **kwargs)

        # Lazy: flydsl is AMD-only and must stay off the Triton import path.
        from aot_tensor.compile.flydsl import (  # @manual  # pyrefly: ignore[missing-import]
            eager as flydsl_eager,
        )

        return flydsl_eager.run_flydsl_jit(self, *args, **kwargs)


# All concrete kernel marker classes -- the single source of truth for the set
# of markers the compile layer registers collectors on / clears (see
# ``compile.compile_state`` and the compile session ``aott_compile``).
ALL_MARKERS: tuple[type[AOTTMarker], ...] = (TritonAOT, CuTeAOT, FlyAOT)


def triton_aot(
    annotations: Dict[str, AnnotationInput],
    compile_options: Optional[Mapping[str, Any]] = None,
) -> Callable[[KernelInterface[Callable[..., Any]]], TritonAOT]:
    def decorator(fn: KernelInterface[Callable[..., Any]]) -> TritonAOT:
        return TritonAOT(fn, annotations, compile_options)

    return decorator


def cutedsl_aot(
    jit_fn: Callable[..., Any],
    arg_specs: list[CuTeArgSpec],
) -> CuTeAOT:
    return CuTeAOT(jit_fn=jit_fn, arg_specs=arg_specs)


def flydsl_aot(
    jit_fn: Any,
    arg_specs: list[FlyArgSpec],
    mutates_args: Iterable[str] | Literal["unknown"] = "unknown",
) -> FlyAOT:
    return FlyAOT(jit_fn=jit_fn, arg_specs=arg_specs, mutates_args=mutates_args)


def get_all_triton_aot_instances() -> List[TritonAOT]:
    """Return all triton aot function instances (e.g. decorated with @triton_aot)."""
    return TritonAOT.get_instances()


def get_all_cutedsl_aot_instances() -> List[CuTeAOT]:
    """Return all cutedsl aot function instances."""
    return CuTeAOT.get_instances()


def get_cutedsl_aot_dir_name(op: CuTeAOT) -> str:
    return f"cutedsl_{op.module_basename}_{op.name}"


def get_flydsl_aot_dir_name(op: FlyAOT) -> str:
    return f"flydsl_{op.module_basename}_{op.name}"


def reset_all_triton_aot_autotune_cache() -> bool:
    """Reset triton autotune cache for all triton aot kernels.

    If triton aot compile is not enabled, this function is no op. Return True if any
    kernel's autotune cache is reset. Else return False.

    """
    if TritonAOT.spec_collector is None:
        return False

    reset = False
    for triton_aot_kernel in get_all_triton_aot_instances():
        if is_autotuner(triton_aot_kernel.fn):
            autotune_fn = triton_aot_kernel.fn
            autotune_fn.cache.clear()  # pyre-ignore [16]
            logger.info(
                f"Reset autotune cache for triton kernel {autotune_fn.fn.__name__}"  # pyre-ignore [16]
            )
            reset = True

    return reset


# Default Triton allocator for the JIT pre-AOT path (spec collection,
# @triton.autotune benchmark, assert_compile_publish reference).
# Mirrors Inductor's pattern in `caffe2/torch/_inductor/runtime/triton_heuristics.py`
# The AOT-T `.so` handles TMA scratch from cpp side
if hasattr(triton, "set_allocator"):

    def _triton_aot_default_allocator(
        size: int, alignment: int, stream: int | None
    ) -> Any:
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(_triton_aot_default_allocator)
