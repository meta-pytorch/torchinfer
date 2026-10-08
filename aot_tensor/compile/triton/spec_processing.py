# Copyright (c) Meta Platforms, Inc. and affiliates.


"""Kernel spec processing for AOT-T compilation.
Kernel arg taxonomy
===================
Every name AOT-T propagates falls into one of 4 disjoint buckets:

  Layer 1 -- Kernel function args (appear in ``fn.arg_names``):
    A1. Pointer args    -- tensors, ``*fp32`` etc.  e.g. ``x_ptr``
    A2. Scalar args     -- runtime scalars, ``i32``/``fp32`` etc.  e.g. ``M``
    A3. Constexpr args  -- ``tl.constexpr``-annotated.  e.g. ``BLOCK_M``.
                           Two sources, same bucket:
                             - bare literal from infer_spec (``ALLOW_TF32=0``)
                             - autotuner-promoted (``BLOCK_M=64`` from cfg)

  Layer 2 -- Triton backend options (NOT in ``fn.arg_names``; passed to
             ``triton.compiler.compile(..., options=...)``):
    B.  ``num_warps``, ``num_stages``, ``matrix_instr_nonkdim``,
        ``waves_per_eu``, ``kpack`` -- AOT-T's curated subset is
        ``AutotuneAttrs.fields_for(backend)``.

``triton.Config.kwargs`` is a *hybrid* dict spanning A3 + B; the split
is performed by ``cfg_constexpr_keys(fn, cfg)`` (name in ``fn.arg_names``
=> A3, else => B).  The two subsets never collide -- Triton does not
name kernel args after backend opts.

Mapping into AOT-T types
-----------------------
  bucket | KernelSpec field          | OpsUnit field
  -------+---------------------------+----------------------------------------
  A1     | signature[i] starts "*"   | pointer_args
  A2     | signature[i] non-"*"      | scalar_dtypes
  A3     | constants[i]              | constant_types (all A3, idx -> py_type),
         |                           | constexpr_keys (autotuner subset of A3,
         |                           |   ordered cfg.kwargs names)
  B      | autotune: AutotuneAttrs   | autotune_fields (schema)

The 4-face codegen contract (cpp signature / cpp guards / meta tuple /
wrapper unpack) is anchored on ``constexpr_keys + autotune_fields`` --
the A3-autotuner-subset followed by B.

The Python wrapper transform (``TritonAOTOperatorTransform``,
``transform/kernel_wrapper_codegen.py``) re-derives the same
A3-autotuner-subset + B name list directly from the autotuner via
``compute_autotune_param_names``; it does not read the ``OpsUnit``.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import logging
import sys
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, cast, ClassVar

import torch

# @manual=//triton:triton
import triton
from aot_tensor.compile.adapter_base import hash_spec
from aot_tensor.compile.spec_conversion import (
    collect_constraints,
    extract_constants,
    get_fp8_replacement_signature_for_amd,
    get_fp8_replacement_signature_for_sm80,
    signature_list_to_dict,
    SignatureElement,
)
from aot_tensor.compile.triton.utils import cfg_constexpr_keys, unwrap_to_jit
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.compiler.compiler import ASTSource, max_shared_mem
from triton.runtime.jit import JITFunction

logger: logging.Logger = logging.getLogger(__name__)

# A raw kernel spec produced by infer_spec.  The only key is "signature".
RawKernelSpec = dict[str, list[SignatureElement]]


def _max_dynamic_shared_per_block_optin(gpu_target: GPUTarget) -> int:
    """Per-block opt-in dynamic SMEM cap, in bytes.

    Returns ``sys.maxsize`` (filter disabled) on non-CUDA / CPU-only
    envs; runtime ``enable_large_smem_or_throw`` is the backstop.
    Trusts ``AOTTCompileSession`` to warn on host/target arch mismatch.
    """
    # TODO: investigate AMD/HIP -- does ``hipDeviceAttributeMaxSharedMemoryPerBlock``
    # need an analogous compile-time filter, or is LDS large enough that
    # the JIT path never trips OutOfResources in practice?
    if gpu_target.backend != "cuda" or not torch.cuda.is_available():
        return sys.maxsize
    return max_shared_mem(0)


@dataclass
class AutotuneAttrs:
    """Tunable backend opts the autotuner picks per kernel variant.

    Single source of truth for "which Triton autotune backend opts AOT-T
    propagates into compile artifacts" — add a new opt = add a field +
    add to the appropriate ``*_FIELDS`` ClassVar. Per-platform partition
    is a manual mirror of upstream ``CUDAOptions`` / ``HIPOptions``;

    TODO: consider deriving the partition dynamically from upstream
    once AOT-T tracks more fields
    """

    # --- Per-platform field partition ---------------------------------------
    COMMON_FIELDS: ClassVar[frozenset[str]] = frozenset({"num_warps", "num_stages"})
    AMD_ONLY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"matrix_instr_nonkdim", "waves_per_eu", "kpack"}
    )
    # Add ``maxnreg`` etc. when a production kernel autotunes over them.
    NVIDIA_ONLY_FIELDS: ClassVar[frozenset[str]] = frozenset({"num_ctas", "auto_tma"})

    # --- Actual autotune fields ---------------------------------------------
    # ``cubin_short`` metadata = short prefix used in cubin file names
    num_warps: int = field(default=4, metadata={"cubin_short": "w"})
    num_stages: int = field(default=3, metadata={"cubin_short": "s"})
    matrix_instr_nonkdim: int = field(default=0, metadata={"cubin_short": "matrix"})
    waves_per_eu: int = field(default=1, metadata={"cubin_short": "wave"})
    kpack: int = field(default=1, metadata={"cubin_short": "kpack"})
    # Hopper+ Thread Block Cluster size. Upstream maps it to
    # ``kernel.metadata.cluster_dims`` at compile time.
    num_ctas: int = field(default=1, metadata={"cubin_short": "cta"})
    # Auto-TMA pass toggle (NVIDIA, Triton 3.8 and beta). Per-variant:
    # True/False produce different cubins. Older 3.5 ignores this option.
    # Omitted from the cubin name when False (see gen_kernel_name) so default
    # kernels are not renamed.
    auto_tma: bool = field(default=False, metadata={"cubin_short": "atma"})

    @classmethod
    @functools.cache
    def field_python_types(cls) -> dict[str, type[Any]]:
        """``field name -> resolved Python type``. Prefer over
        ``type(field.default)`` — defaults drift, annotations don't."""
        hints = typing.get_type_hints(cls)
        return {f.name: hints[f.name] for f in dataclasses.fields(cls)}

    @classmethod
    def fields_for(cls, backend: str) -> tuple[dataclasses.Field[Any], ...]:
        """Autotune fields relevant to *backend* ('cuda' or 'hip').
        Order matters for later codegen
        """
        all_fields = dataclasses.fields(cls)
        if backend == "cuda":
            relevant = cls.COMMON_FIELDS | cls.NVIDIA_ONLY_FIELDS
        elif backend == "hip":
            relevant = cls.COMMON_FIELDS | cls.AMD_ONLY_FIELDS
        else:
            raise ValueError(f"Unknown backend {backend!r}; expected 'cuda' or 'hip'.")
        return tuple(f for f in all_fields if f.name in relevant)

    @classmethod
    def from_cfg(
        cls,
        cfg: "triton.Config",
        autotune_fields: tuple[dataclasses.Field[Any], ...],
    ) -> "AutotuneAttrs":
        """Materialize an ``AutotuneAttrs`` populating only *autotune_fields*.
        Read order: ``cfg.kwargs`` → ``cfg`` attrs → field default."""

        def _read(name: str, default: Any) -> Any:
            if name in cfg.kwargs:
                v = cfg.kwargs[name]
            else:
                v = getattr(cfg, name, default)
            # A Config may carry an unset tri-state opt as ``None`` (e.g.
            # ``auto_tma``); fall back to the field default so typed fields
            # (bool/int) never receive ``None``.
            return default if v is None else v

        return cls(**{f.name: _read(f.name, f.default) for f in autotune_fields})


def _compute_constexpr_keys(
    autotuner: triton.runtime.autotuner.Autotuner,
) -> tuple[str, ...]:
    """Constexpr keys autotuned by *autotuner*, validated uniform across cfgs.

    Order follows the first cfg's ``kwargs`` insertion order; cfgs must all
    agree (see ``ValueError`` below).
    """
    cfgs = list(autotuner.cache.values()) or list(autotuner.configs)
    if not cfgs:
        return ()
    keys = tuple(cfg_constexpr_keys(autotuner, cfgs[0]))
    for cfg in cfgs[1:]:
        other = tuple(cfg_constexpr_keys(autotuner, cfg))
        if other != keys:
            raise ValueError(
                f"Autotuner cfgs disagree on constexpr keys: "
                f"first={keys!r}, other={other!r}"
            )
    return keys


def compute_autotune_param_names(
    autotuner: triton.runtime.autotuner.Autotuner | None,
    backend: str,
) -> list[str]:
    """Flat ``constexpr_keys + autotune_field_names`` for *backend*."""
    constexpr_keys = _compute_constexpr_keys(autotuner) if autotuner is not None else ()
    autotune_fields = AutotuneAttrs.fields_for(backend)
    return list(constexpr_keys) + [f.name for f in autotune_fields]


@dataclass
class KernelSpec:
    """A single compilation variant for a kernel.

    Each variant is one combination of dtypes, constant values, alignment
    constraints, and autotune configuration.  Variants are grouped into
    an ``OpsUnit``.  See module docstring §Kernel arg taxonomy for the
    A1/A2/A3/B bucket model these fields project onto.

    Attributes:
        signature: A1 + A2 indexed by arg idx → dtype string
            (``*fp32`` for pointers, ``i32``/``fp32`` for scalars).
        constants: A3 indexed by arg idx → compile-time value.  Mixes
            bare literals (128, ``"leaky_relu"``), absent optional tensors
            (``None``), equal-to-1 specializations, and autotuner-promoted
            constexprs (``BLOCK_M=64``).
        divisible_by_16: Indices whose values are 16-byte aligned (pointers)
            or multiples of 16 (scalars).
        divisible_by_8: Indices whose scalar values are multiples of 8
            (pointer alignment is always ≥16, so meaningless for pointers).
        autotune: B values picked by the autotuner; see ``AutotuneAttrs``.
    """

    signature: dict[int, str]
    constants: dict[int, Any]
    divisible_by_16: set[int]
    divisible_by_8: set[int]
    autotune: AutotuneAttrs = dataclasses.field(default_factory=AutotuneAttrs)


@dataclass
class OpsUnit:
    """All compilation variants for a single kernel op.

    Groups per-kernel invariants with the list of ``KernelSpec`` variants.
    Use ``OpsUnit.from_raw_specs()`` to build — it performs the complete
    spec processing pipeline (convert → detect optional → validate →
    autotune → dedup → compute invariants).  See module docstring
    §Kernel arg taxonomy for the A1/A2/A3/B bucket model these fields
    project onto.

    Attributes:
        cc: Compute capability (int for NVIDIA, str for AMD).
        optional: Indices of optional tensor args (unified across all
            call sites).
        pointer_args: A1 — all tensor pointer indices (required + optional).
            Invariant across specs.
        scalar_dtypes: A2 — non-pointer signature arg idx → widest dtype
            across specs (per ``_wider_type``); individual specs may use
            narrower types.
        constant_types: A3 *full set* — every constant arg idx → Python
            type.  Includes bare literals AND autotuner-promoted entries.
            Used by cpp type codegen (``PY_TYPES_TO_CPP_TYPES``).
            Invariant across specs.
        specs: Per-variant ``KernelSpec`` list.
        constexpr_keys: A3 *autotuner subset only* — ordered constexpr arg
            names in ``cfg.kwargs`` insertion order.  ``()`` for plain
            ``@triton.jit``.  Used by meta-function tuple ordering and by
            ``_autotune_specs`` to promote ``cfg.kwargs[name]`` into
            ``KernelSpec.constants``.  Each name resolves (via
            ``fn.arg_names.index``) to an idx already in ``constant_types``.
        autotune_fields: B — ``AutotuneAttrs`` fields relevant to ``cc``'s
            backend.  Schema only; per-variant values live in
            ``KernelSpec.autotune``.

    ``constexpr_keys + [f.name for f in autotune_fields]`` is the 4-face
    contract anchor (cpp signature / cpp guards / meta tuple / wrapper
    unpack codegen).
    """

    cc: int | str
    optional: set[int]
    pointer_args: set[int]
    scalar_dtypes: dict[int, str]
    constant_types: dict[int, type[Any]]
    specs: list[KernelSpec]
    constexpr_keys: tuple[str, ...] = ()
    autotune_fields: tuple[dataclasses.Field[Any], ...] = ()
    # Per-block opt-in dynamic SMEM cap (bytes). Set by ``from_raw_specs``;
    # defaults to ``sys.maxsize`` (no filter) for bare-ctor test use.
    smem_cap: int = sys.maxsize

    def drop_oversize_specs(
        self,
        generated_specs: list[tuple[str, int]],
    ) -> tuple[OpsUnit, list[str]]:
        """Drop specs whose compiled SMEM exceeds ``self.smem_cap``.

        Mirrors Triton JIT's launch-time ``OutOfResources`` check
        (``compiler.py::_init_handles``) at AOT-T compile time.

        Returns ``(narrowed_unit, filtered_codes)``. Raises if every
        spec was dropped.
        """
        kept_specs: list[KernelSpec] = []
        filtered_specs: list[str] = []
        for spec, (code, shared) in zip(self.specs, generated_specs):
            if shared <= self.smem_cap:
                kept_specs.append(spec)
                filtered_specs.append(code)
            else:
                logger.warning(
                    f"[AOT-T] Dropping over-SMEM spec: {shared} > cap "
                    f"{self.smem_cap} bytes (autotune={spec.autotune}, "
                    f"constants={spec.constants})"
                )
        if not kept_specs:
            raise RuntimeError(
                f"All {len(self.specs)} specs exceeded SMEM cap "
                f"{self.smem_cap} bytes; nothing left to compile."
            )
        return dataclasses.replace(self, specs=kept_specs), filtered_specs

    @classmethod
    def from_raw_specs(
        cls,
        base_specs: list[RawKernelSpec],
        gpu_target: GPUTarget,
        tuned_func: triton.runtime.autotuner.Autotuner | None = None,
    ) -> OpsUnit:
        """Build an OpsUnit from raw kernel specs.

        Performs the complete spec processing pipeline:
        1. Convert raw specs to KernelSpecs
        2. Detect optional tensor args (cross-spec + 3-tuple)
        3. Validate consistency across converted specs
        4. Apply autotuning (if tuned_func provided)
        5. Deduplicate specs
        6. Compute shared invariants (pointer_args, scalar_dtypes, constant_types)
        """
        # Validate raw specs upfront, before any rewriting.
        num_params = _check_uniform_signature_length(base_specs)
        specs, three_tuple_optional = _convert_raw_specs(base_specs, gpu_target)
        optional = _detect_optional_args(specs) | three_tuple_optional

        _validate_converted_specs(specs, optional, num_params)

        autotune_fields = AutotuneAttrs.fields_for(gpu_target.backend)
        constexpr_keys = (
            _compute_constexpr_keys(tuned_func) if tuned_func is not None else ()
        )

        # Plain @triton.jit kernels (no @triton.autotune) skip config expansion.
        if tuned_func is not None:
            specs = _autotune_specs(tuned_func, constexpr_keys, autotune_fields, specs)

        specs = _dedup_specs(specs)

        pointer_args, scalar_dtypes, constant_types = _compute_invariants(
            specs, optional
        )

        return cls(
            cc=gpu_target.arch,
            optional=optional,
            pointer_args=pointer_args,
            scalar_dtypes=scalar_dtypes,
            constant_types=constant_types,
            specs=specs,
            constexpr_keys=constexpr_keys,
            autotune_fields=autotune_fields,
            smem_cap=_max_dynamic_shared_per_block_optin(gpu_target),
        )


# ---------------------------------------------------------------------------
# Public helpers (used outside spec processing)
# ---------------------------------------------------------------------------


def gen_compile_arg(
    spec: KernelSpec,
    func: JITFunction[Callable[..., Any]],
) -> tuple[ASTSource]:
    # ASTSource expects tuple-keyed dicts: {(idx,): value} for constants,
    # {(idx,): [[attr_name, attr_val], ...]} for attrs.  Tuple keys support
    # nested paths into structured types (asserted by ASTSource.__init__).
    new_signature = {}
    new_constants = {}
    param_names = list(func.signature.parameters.keys())
    for idx, param in enumerate(param_names):
        if idx in spec.signature:
            new_signature[param] = spec.signature[idx]
        if idx in spec.constants:
            new_constants[(idx,)] = spec.constants[idx]
            new_signature[param] = "constexpr"

    # Constexprs get no attrs: they are not runtime args, and upstream Triton
    # applies each attr at the current runtime-arg position, so an attr on a
    # constexpr lands on the next runtime arg or past the last one.
    # parse_attr("D") returns a fresh [["tt.divisibility", 16]] each call.
    new_attrs = {
        (idx,): BaseBackend.parse_attr("D")
        for idx in spec.divisible_by_16
        if idx not in spec.constants
    }

    return (
        ASTSource(
            func,
            new_signature,
            constexprs=new_constants,
            attrs=new_attrs,
        ),
    )


# ---------------------------------------------------------------------------
# Int width helpers
# ---------------------------------------------------------------------------

_INT_WIDTH_RANK: dict[str, int] = {"i32": 0, "i64": 1}


def _wider_type(t1: str, t2: str) -> str:
    """Return the wider of two scalar dtypes.

    Only i32/i64 widening is supported.  All other types must match exactly.
    """
    if t1 == t2:
        return t1
    r1 = _INT_WIDTH_RANK.get(t1)
    r2 = _INT_WIDTH_RANK.get(t2)
    if r1 is not None and r2 is not None:
        return t1 if r1 >= r2 else t2
    raise ValueError(f"Cannot widen incompatible types: {t1!r} vs {t2!r}")


# ---------------------------------------------------------------------------
# Private helpers — called by OpsUnit.from_raw_specs
# ---------------------------------------------------------------------------


def _detect_optional_args(specs: list[KernelSpec]) -> set[int]:
    """Detect optional tensor args by cross-spec comparison.

    An arg at index ``i`` is optional if:
    - Some specs have ``i`` in ``signature`` as a pointer type (``*...``)
    - Other specs have ``constants[i] = None``

    Single-spec None args (always-absent tensors) are NOT detected here
    but are handled by ``_compute_invariants`` which adds any
    ``constants[i] = None`` to ``pointer_args``.
    """
    if len(specs) <= 1:
        return set()
    optional: set[int] = set()
    all_indices: set[int] = set()
    for spec in specs:
        all_indices |= spec.signature.keys()
        all_indices |= spec.constants.keys()
    for i in all_indices:
        has_pointer = any(
            i in s.signature and s.signature[i].startswith("*") for s in specs
        )
        has_none_const = any(i in s.constants and s.constants[i] is None for s in specs)
        if has_pointer and has_none_const:
            optional.add(i)
    return optional


def _check_uniform_signature_length(base_specs: list[RawKernelSpec]) -> int:
    """All raw specs must declare the same param count; return that count.

    Each raw spec is one ``infer_spec`` call site for the same kernel,
    so all should have ``len(fn.signature.parameters)`` entries.  Differing
    lengths means upstream bug (mixed kernels, truncated spec, etc.) and
    would surface later as silent IndexError or wrong bound checks.
    """
    if not base_specs:
        return 0
    sig_lens = {len(spec["signature"]) for spec in base_specs}
    if len(sig_lens) != 1:
        raise ValueError(
            f"Raw specs declare inconsistent signature lengths: "
            f"{sorted(sig_lens)}.  All specs for the same kernel must have "
            f"one entry per declared param."
        )
    return sig_lens.pop()


def _check_arg_indices_in_range(
    specs: list[KernelSpec],
    num_params: int,
) -> None:
    """Every spec arg index must be in ``[0, num_params)``.

    Out-of-range indices would silently drop in ``gen_compile_arg``'s
    ``enumerate(param_names)`` loop.  ``num_params <= 0`` disables the check.
    """
    if num_params <= 0:
        return
    for idx, spec in enumerate(specs):
        all_indices = (
            spec.signature.keys()
            | spec.constants.keys()
            | spec.divisible_by_16
            | spec.divisible_by_8
        )
        for i in all_indices:
            if not 0 <= i < num_params:
                raise ValueError(
                    f"Spec {idx}: arg index {i} out of range "
                    f"[0, {num_params}) — kernel has {num_params} declared params"
                )


def _collect_pointer_args(
    specs: list[KernelSpec],
    optional: set[int],
) -> set[int]:
    """Collect all tensor pointer indices across all specs.

    Includes optional args (from _detect_optional_args) AND any arg
    whose constant value is None (single-spec optional tensor case
    where _detect_optional_args didn't fire).
    """
    pointer_args: set[int] = set(optional)
    for spec in specs:
        for i, dtype in spec.signature.items():
            if dtype.startswith("*"):
                pointer_args.add(i)
        for i, val in spec.constants.items():
            if val is None:
                pointer_args.add(i)
    return pointer_args


def _collect_scalar_dtypes(
    specs: list[KernelSpec],
    pointer_args: set[int],
) -> dict[int, str]:
    """Collect non-pointer signature arg dtypes, widening compatible int types.

    Invariant across specs (validated by _validate_converted_specs).
    """
    scalar_dtypes: dict[int, str] = {}
    for spec in specs:
        for i, dtype in spec.signature.items():
            if i not in pointer_args:
                if i in scalar_dtypes:
                    scalar_dtypes[i] = _wider_type(scalar_dtypes[i], dtype)
                else:
                    scalar_dtypes[i] = dtype
    return scalar_dtypes


def _collect_constant_types(
    specs: list[KernelSpec],
) -> dict[int, type[Any]]:
    """Collect Python type per constant position.

    Excludes None constants (optional tensor args — already in pointer_args).
    """
    constant_types: dict[int, type[Any]] = {}
    for spec in specs:
        for i, val in spec.constants.items():
            if val is not None and i not in constant_types:
                constant_types[i] = type(val)
    return constant_types


def _compute_invariants(
    specs: list[KernelSpec],
    optional: set[int],
) -> tuple[set[int], dict[int, str], dict[int, type[Any]]]:
    """Compute shared invariants from processed specs.

    Returns (pointer_args, scalar_dtypes, constant_types).

    When annotation-as-variant produces mixed partitions (arg in
    ``signature`` in some specs, ``constants`` in others), the arg
    appears in both ``scalar_dtypes`` and ``constant_types``.  The
    selector must receive it as a runtime parameter for dispatch,
    so ``scalar_dtypes`` wins and the arg is removed from
    ``constant_types``.
    """
    pointer_args = _collect_pointer_args(specs, optional)
    scalar_dtypes = _collect_scalar_dtypes(specs, pointer_args)
    constant_types = _collect_constant_types(specs)

    # Resolve overlap: if any spec has the arg in signature (scalar),
    # the selector needs it as a runtime parameter → not a constant.
    for i in scalar_dtypes:
        constant_types.pop(i, None)

    return pointer_args, scalar_dtypes, constant_types


def _validate_converted_specs(
    specs: list[KernelSpec],
    optional: set[int],
    num_params: int = 0,
) -> None:
    """Validate that converted specs are consistent before further processing.

    Checks that all specs produce identical C++ function signatures:
    - All arg indices are in ``[0, num_params)`` (when ``num_params > 0``)
    - Optional args: each spec has either a pointer in signature or None in constants
    - Non-optional scalar args: same dtype (or compatible int widths)
    - Non-optional constant args: same Python type

    Called after _convert_raw_specs + _detect_optional_args, before autotuning.
    """
    _check_arg_indices_in_range(specs, num_params)
    if len(specs) <= 1:
        return
    ref = specs[0]
    for idx, spec in enumerate(specs[1:], 1):
        _check_optional_consistency(ref, spec, idx, optional)
        _check_signature_consistency(ref, spec, idx, optional)
        _check_constants_consistency(ref, spec, idx, optional)


def _check_optional_consistency(
    ref: KernelSpec,
    spec: KernelSpec,
    idx: int,
    optional: set[int],
) -> None:
    """Optional positions must be pointer-in-signature or None-in-constants.

    Validates that optional tensor args are not misclassified as scalars
    or non-None constants, which would produce incompatible C++ types.
    """
    for i in optional:
        for label, s in [("spec 0", ref), (f"spec {idx}", spec)]:
            if i in s.signature:
                if not s.signature[i].startswith("*"):
                    raise ValueError(
                        f"Arg {i}: optional position has non-pointer type "
                        f"'{s.signature[i]}' in {label}"
                    )
            elif i in s.constants:
                if s.constants[i] is not None:
                    raise ValueError(
                        f"Arg {i}: optional position has non-None constant "
                        f"{s.constants[i]!r} in {label}"
                    )


def _check_signature_consistency(
    ref: KernelSpec,
    spec: KernelSpec,
    idx: int,
    optional: set[int],
) -> None:
    """Non-optional, non-pointer scalar args must have compatible dtypes.

    Pointer args are skipped (different tensor dtypes are dispatched by
    the dtype guard in ``gen_guarded_calls``).  Compatible int widths
    (i32/i64) are allowed — handled by ``_wider_type`` and int range guards.
    Optional positions are validated by ``_check_optional_consistency``.

    Partition differences are allowed: an arg may be in ``signature`` in
    one spec and in ``constants`` in another (e.g., annotation-as-variant
    where stride=1 is constexpr in one spec but a runtime parameter in
    another).  The per-spec codegen handles this correctly.
    """
    for i in ref.signature.keys() | spec.signature.keys():
        if i in optional:
            continue
        if (i in ref.signature and ref.signature[i].startswith("*")) or (
            i in spec.signature and spec.signature[i].startswith("*")
        ):
            continue
        # Allow partition differences: arg in signature in one spec,
        # in constants in another (annotation-as-variant pattern).
        if i not in ref.signature or i not in spec.signature:
            continue
        if ref.signature[i] != spec.signature[i]:
            r1 = _INT_WIDTH_RANK.get(ref.signature[i])
            r2 = _INT_WIDTH_RANK.get(spec.signature[i])
            if r1 is not None and r2 is not None:
                continue
            raise ValueError(
                f"Arg {i}: dtype mismatch '{ref.signature[i]}' vs "
                f"'{spec.signature[i]}' (spec 0 vs spec {idx})"
            )


def _check_constants_consistency(
    ref: KernelSpec,
    spec: KernelSpec,
    idx: int,
    optional: set[int],
) -> None:
    """Non-optional constant args must have the same Python type across specs.

    C++ codegen uses one type per constant arg position (``PY_TYPES_TO_CPP_TYPES``),
    so ``BLOCK_M=64`` (int) and ``BLOCK_M=64.0`` (float) would produce
    incompatible launchers.  Optional positions are validated separately
    by ``_check_optional_consistency``.
    """
    for i in ref.constants.keys() | spec.constants.keys():
        if i in optional:
            continue
        if ref.constants.get(i) is None or spec.constants.get(i) is None:
            continue
        if type(ref.constants[i]) is not type(spec.constants[i]):
            raise ValueError(
                f"Arg {i}: constant type mismatch "
                f"{type(ref.constants[i]).__name__} vs "
                f"{type(spec.constants[i]).__name__} (spec 0 vs spec {idx})"
            )


def _convert_raw_specs(
    base_specs: list[RawKernelSpec],
    gpu_target: GPUTarget,
) -> tuple[list[KernelSpec], set[int]]:
    """Convert raw specs to KernelSpecs.

    Returns (specs, three_tuple_optional) where three_tuple_optional is the
    union of optional_args detected from 3-tuple signature elements across
    all specs (backward compat with ``collect_constraints``).
    """
    raw_specs = cast(list[dict[str, Any]], copy.deepcopy(base_specs))
    is_amd = gpu_target.backend == "hip"

    result: list[KernelSpec] = []
    three_tuple_optional: set[int] = set()
    for raw_spec in raw_specs:
        constraints = collect_constraints(raw_spec["signature"])
        constants = extract_constants(raw_spec["signature"], constraints)
        signature: dict[int, str] = signature_list_to_dict(
            raw_spec["signature"], constants
        )
        three_tuple_optional |= constraints.optional_args

        spec = KernelSpec(
            signature=signature,
            constants=constants,
            divisible_by_16=constraints.divisible_by_16,
            divisible_by_8=constraints.divisible_by_8,
        )

        if constraints.has_fp8:
            if is_amd:
                spec.signature = get_fp8_replacement_signature_for_amd(
                    {"signature": spec.signature}, {str(gpu_target.arch)}
                )
            elif gpu_target.arch == 80:
                spec.signature = get_fp8_replacement_signature_for_sm80(
                    {"signature": spec.signature}
                )

        result.append(spec)

    return result, three_tuple_optional


def _autotune_specs(
    autotuner: triton.runtime.autotuner.Autotuner,
    constexpr_keys: tuple[str, ...],
    autotune_fields: tuple[dataclasses.Field[Any], ...],
    specs: list[KernelSpec],
) -> list[KernelSpec]:
    """Expand each base spec into one variant per autotuner cache entry.

    Splits ``cfg.kwargs`` into two flows:
      *constexpr_keys* -> ``KernelSpec.constants`` promotion
      backend opt fields -> ``KernelSpec.autotune`` via ``AutotuneAttrs.from_cfg``

    Iterates ``autotuner.cache`` (autotune *decisions*), not ``configs``
    (the candidate pool), unlike ``_compute_constexpr_keys`` which can
    fall back to ``configs`` for the key-name schema.

    TODO: full cartesian over-emits cells for per-call constexpr (some
    never co-bench'd, may exceed SMEM -- ``drop_oversize_specs`` is the
    backstop). Investigate diagonal-on-constants pairing for better
    coverage/perf balance; needs cfg ↔ constants info from frontend.
    """
    jit_fn = unwrap_to_jit(autotuner)
    tuned_specs: list[KernelSpec] = []
    for spec in specs:
        for cfg in autotuner.cache.values():
            constants = spec.constants.copy()
            for arg_name in constexpr_keys:
                arg_idx = jit_fn.arg_names.index(arg_name)
                if constants.get(arg_idx, -1) == -1:
                    constants[arg_idx] = cfg.kwargs[arg_name]
            tuned_specs.append(
                dataclasses.replace(
                    spec,
                    constants=constants,
                    autotune=AutotuneAttrs.from_cfg(cfg, autotune_fields),
                )
            )
    return tuned_specs


def _dedup_specs(specs: list[KernelSpec]) -> list[KernelSpec]:
    deduped_specs: list[KernelSpec] = []
    duplicated_specs: list[KernelSpec] = []
    hash_spec_ids: set[str] = set()
    for spec in specs:
        id = hash_spec(dataclasses.asdict(spec))
        if id in hash_spec_ids:
            duplicated_specs.append(spec)
        else:
            hash_spec_ids.add(id)
            deduped_specs.append(spec)

    logger.debug(
        f"[TritonAOT Dedup] {len(specs)=} {len(deduped_specs)=} {len(duplicated_specs)=}"
    )
    return deduped_specs
