# Copyright (c) Meta Platforms, Inc. and affiliates.


from __future__ import annotations

import ast
import copy
import inspect
import logging
import os
import pickle
from dataclasses import dataclass
from inspect import getcallargs, Parameter, signature
from typing import Any, Callable, Dict, List, Optional

import torch

# @manual=//triton:triton
import triton.language as tl
from aot_tensor.build.triton.extension_builder import build_triton_aot_extension
from aot_tensor.compile.adapter_base import (
    AOTTAdapter,
    CompileContext,
    DslCompileConfig,
    hash_spec,
)
from aot_tensor.compile.compile_state import add_spec, get_kernel_specs, register_active
from aot_tensor.compile.stable_types import SCALAR_TYPES
from aot_tensor.compile.triton.codegen import is_non_empty_mapping_of_type
from aot_tensor.compile.triton.compat import get_kernel_name
from aot_tensor.compile.triton.pipeline import compile_to_cpp
from aot_tensor.compile.triton.spec_processing import compute_autotune_param_names
from aot_tensor.compile.triton.utils import (
    AutotuneCache,
    is_autotuner,
    kernel_param_names,
    try_get_autotuner,
    unwrap_to_jit,
)
from aot_tensor.constants import TRITON
from aot_tensor.transform.wrapper_codegen_utils import (
    _get_clean_module_basename,
    find_sole_marker_in_globals,
    generate_wrapper_files_skeleton,
    strip_jit_unused_decorator,
)
from aot_tensor.types import Annotation, AnnotationHint, TritonAOT
from pyre_extensions import none_throws
from torch import package

# @manual=//triton:triton
from triton.backends.compiler import GPUTarget

# @manual=//triton:triton
from triton.runtime import driver

# @manual=//triton:triton
from triton.runtime.autotuner import Autotuner, Config

# @manual=//triton:triton
from triton.runtime.jit import JITFunction, KernelInterface, mangle_type

logger: logging.Logger = logging.getLogger(__name__)


# =====================================================================
# spec collection (records into the singleton's per-DSL store)
# =====================================================================


def _collect_triton_spec(
    fn: KernelInterface[List[Any]],
    annotations: Dict[str, Annotation],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Record the specs for one Triton kernel call (invoked by ``collect``).

    Always collects the annotated spec (which equals the inferred spec
    when no annotations are present).  Also collects the inferred spec
    when it differs and either:
    - annotations conflict with sample (fallback for safety), or
    - inferred has perf hints the annotation lacks (perf variant).
    """
    _ensure_multi_config_autotuner(fn)
    spec = infer_spec(fn, annotations, *args, **kwargs)
    annotated_hash = hash_spec(spec)
    add_spec(TRITON, fn, spec, annotated_hash)

    if annotations:
        inferred = infer_spec(fn, {}, *args, **kwargs)
        inferred_hash = hash_spec(inferred)
        if inferred_hash == annotated_hash:
            return
        if _annotation_conflicts_with_sample(
            fn, annotations, *args, **kwargs
        ) or _inferred_has_perf_advantage(spec, inferred):
            add_spec(TRITON, fn, inferred, inferred_hash)


# =====================================================================
# spec inference (Triton-specific)
# =====================================================================


def _ensure_multi_config_autotuner(fn: KernelInterface[List[Any]]) -> None:
    """Duplicate a single ``@triton.autotune`` Config so Triton runs the
    benchmark path and populates ``Autotuner.cache``.

    Triton's ``Autotuner.run()`` skips benchmarking when
    ``len(self.configs) == 1`` and never writes ``self.cache``. AOT-T
    enumerates kernel variants from ``autotuner.cache``, so without this
    a single-Config kernel collects zero specs.
    """
    if not is_autotuner(fn):
        return
    # pyre-ignore[16]
    if len(fn.configs) != 1:
        return
    # pyre-ignore[16]
    fn.configs = list(fn.configs) + [copy.copy(fn.configs[0])]
    logger.info(
        "TritonAOT: duplicated single Config on %r so autotune cache populates.",
        fn,
    )


def _unwrap_triton_fn(
    fn: KernelInterface[List[Any]],
) -> Callable[..., Any]:
    while isinstance(fn, KernelInterface):
        # pyre-ignore[16]: KernelInterface has `fn` attribute at runtime
        fn = fn.fn
    return fn


def _inferred_has_perf_advantage(
    annotated_spec: Dict[str, List[Any]],
    inferred_spec: Dict[str, List[Any]],
) -> bool:
    """True if inferred spec has alignment/divisibility hints the annotated lacks.

    A tuple element ``(type, N)`` carries alignment or divisibility info
    that a bare string does not.  When inference adds such hints (e.g.,
    tensor alignment from ``data_ptr() % 16 == 0``), the inferred spec
    produces a more optimized cubin worth keeping as a perf variant.
    """
    for ann_elem, inf_elem in zip(
        annotated_spec["signature"], inferred_spec["signature"]
    ):
        if isinstance(inf_elem, tuple) and not isinstance(ann_elem, tuple):
            return True
    return False


# Triton-internal kwargs injected by KernelInterface.__getitem__
# (triton/runtime/jit.py).  These are not kernel parameters and must
# be stripped before getcallargs.
_TRITON_INTERNAL_KWARGS: frozenset[str] = frozenset({"warmup", "grid"})


def _resolve_call_args(
    fn: KernelInterface[List[Any]],
    *args: Any,
    **kwargs: Any,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Unwrap kernel and bind every kernel param to a value via ``getcallargs``.

    ``getcallargs`` errors on unknown kwargs OR on missing required params,
    so ``**kwargs`` must be cleaned before binding.  Positional ``*args``
    (kernel tensors / scalars) pass through untouched.

    | Source                                   | Issue                       | Handling                                 |
    | ---------------------------------------- | --------------------------- | ---------------------------------------- |
    | Triton-internal kwargs                   | Not in kernel signature     | Drop via ``_TRITON_INTERNAL_KWARGS``     |
    | (``warmup``, ``grid``, ...)              |                             |                                          |
    | ``@autotune`` constexprs                 | Caller doesn't supply →     | Fill ``-1`` (replaced later              |
    | (``BLOCK_M``, ``GROUP_M``, ...)          | "missing required kwarg"    | by ``_autotune_specs``)                  |
    | Backend opts in ``cfg.kwargs``           | Not in kernel signature →   | Excluded via set intersection            |
    | (AMD: ``matrix_instr_nonkdim``, ...)     | "unexpected kwarg"          | ``tuned_kwargs & fn_params``             |
    """
    triton_fn = _unwrap_triton_fn(fn)
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in _TRITON_INTERNAL_KWARGS}
    if is_autotuner(fn):
        # pyre-ignore[16]: Attributes checked by is_autotuner
        tuned_kwargs = set(fn.configs[0].kwargs.keys())
        fn_params = kernel_param_names(fn)
        for arg_name in tuned_kwargs & fn_params:
            if arg_name not in clean_kwargs:
                clean_kwargs[arg_name] = -1
    return triton_fn, getcallargs(triton_fn, *args, **clean_kwargs)


_I32_MIN: int = -(2**31)
_I32_MAX: int = 2**31 - 1


def _sample_satisfies_int_type(sample: int, ann_type: str) -> bool:
    """True if sample int fits the annotated type range."""
    if ann_type == "i32":
        return _I32_MIN <= sample <= _I32_MAX
    return True


def _sample_satisfies_annotation(sample: Any, ann: Annotation) -> bool:
    """True if a single sample value satisfies its annotation constraint."""
    if isinstance(ann, AnnotationHint):
        if isinstance(sample, torch.Tensor):
            return sample.data_ptr() % ann.hint == 0
        if isinstance(sample, int):
            if ann.hint == 1:
                return sample == 1
            if not _sample_satisfies_int_type(sample, ann.dtype):
                return False
            if ann.hint > 1:
                return sample % ann.hint == 0
        return True
    if isinstance(ann, str) and not ann.startswith("*") and isinstance(sample, int):
        return _sample_satisfies_int_type(sample, ann)
    return True


def _annotation_conflicts_with_sample(
    fn: KernelInterface[List[Any]],
    annotations: Dict[str, Annotation],
    *args: Any,
    **kwargs: Any,
) -> bool:
    """True if any annotated param's sample value doesn't satisfy the annotation.

    Used by ``_collect_triton_spec`` to decide whether to generate an inferred
    fallback spec.  When the sample satisfies all annotations, only the
    annotated spec is needed (the user's constraints hold for this input).
    """
    _, sample_args = _resolve_call_args(fn, *args, **kwargs)

    for param_name, ann in annotations.items():
        sample = sample_args.get(param_name)
        if sample is None:
            continue
        if not _sample_satisfies_annotation(sample, ann):
            return True

    return False


def infer_spec(  # noqa: C901
    fn: KernelInterface[List[Any]],
    annotations: Dict[str, Annotation],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, List[Any]]:
    """Infer kernel spec from sample args.

    Tensor dtype: ``mangle_type``, alignment: ``data_ptr() % 16``.
    Scalar int: always ``"i64"`` (safe default; user can annotate ``"i32"``
    to get a narrower variant via annotation-as-variant).
    Float: ``mangle_type`` → fp32.
    """
    triton_fn, call_args = _resolve_call_args(fn, *args, **kwargs)
    fn_sig = signature(triton_fn)
    arg_annotations = {
        name: param.annotation for name, param in fn_sig.parameters.items()
    }
    spec = []

    for arg_name in fn_sig.parameters.keys():
        arg = call_args[arg_name]
        if arg_annotations[arg_name] != Parameter.empty:
            if arg_annotations[arg_name] == tl.constexpr:
                spec.append(arg)
            else:
                raise RuntimeError(
                    f"TritonAOT: unsupported scalar annotation {arg_annotations[arg_name]}."
                )
        elif arg_name in annotations:
            ann = annotations[arg_name]
            # Convert to tuple for raw spec format (shared/spec_conversion
            # processes plain tuples).
            spec.append(ann.to_tuple() if isinstance(ann, AnnotationHint) else ann)
        elif arg is None:
            spec.append(None)
        elif isinstance(arg, torch.Tensor):
            # Reject dtypes SCALAR_TYPES can't render (e.g. *u16, *fp8e5)
            # so codegen doesn't KeyError downstream.
            type_str = mangle_type(arg)
            if type_str not in SCALAR_TYPES:
                raise RuntimeError(
                    f"TritonAOT: unsupported tensor type for {arg_name}: "
                    f"{arg.dtype} (Triton mangled to {type_str!r}). "
                    f"Supported tensor dtypes: {sorted(SCALAR_TYPES.keys())}."
                )
            if arg.data_ptr() % 16 == 0:
                spec.append((type_str, 16))
            else:
                spec.append(type_str)
        elif isinstance(arg, bool):
            # bool is subclass of int; must check before int.
            # Non-constexpr bools have no CTYPES entry for codegen.
            raise RuntimeError(
                f"TritonAOT: parameter {arg_name} is a bool without "
                f"tl.constexpr annotation.  Add `{arg_name}: tl.constexpr` "
                f"to the kernel signature."
            )
        elif isinstance(arg, int):
            # Always i64 for safety; users annotate "i32" for narrower
            # variant via annotation-as-variant.
            # TODO: observe scalar divisibility (arg % 16, arg % 8) like
            # tensor alignment — produces optimized spec with divisible_by_16
            # guard, potential QPS improvement from skipping boundary checks.
            if not -(2**63) <= arg <= 2**63 - 1:
                raise RuntimeError(
                    f"TritonAOT: unsupported int value for {arg_name}: "
                    f"value exceeds i64 range. "
                    f"Use a smaller value or tl.constexpr."
                )
            spec.append("i64")
        elif isinstance(arg, float):
            spec.append("fp32")
        else:
            raise RuntimeError(f"TritonAOT: parameter {arg_name} needs annotation.")
    return {"signature": spec}


# =====================================================================
# autotune cache overrides + default values (compile-time, Triton-only)
# =====================================================================


def _resolve_autotune_cache(
    fn: KernelInterface[List[Any]],
    fn_name: str,
    fn_dir: str,
    overrides: dict[str, AutotuneCache],
) -> None:
    """Apply override (if matched) and dump the autotune cache to fn_dir.

    ``cache`` is an ``Autotuner``-only concept, so this is a no-op for
    non-autotuned kernels.
    """
    autotuner = try_get_autotuner(fn)
    if autotuner is None:
        return

    override = overrides.get(fn_name)
    if override is not None:
        logger.info(f"[AOTT]: Overriding autotune cache for {fn_name}")
        autotuner.cache = override

    # Dump the resolved autotune cache to disk so tests can inspect it;
    # the compile itself never reads this file back.
    if is_non_empty_mapping_of_type(autotuner.cache, Config):
        with open(f"{fn_dir}/{fn_name}_autotune_cache", "wb") as data:
            # @lint-ignore PYTHONPICKLEISBAD
            pickle.dump(autotuner.cache, data)


def _extract_default_values(jit_fn: JITFunction[List[Any]]) -> dict[str, Any]:
    """Extract default values from the Triton JIT function's Python signature.

    These defaults are emitted in the C++ op schema so TorchScript
    treats them as optional parameters instead of required ones.

    Takes an already-unwrapped ``jit_fn`` (the caller unwraps once).
    """
    python_fn = getattr(jit_fn, "fn", None)
    if python_fn is None or not callable(python_fn):
        return {}
    try:
        sig = inspect.signature(python_fn)
        return {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
    except (ValueError, TypeError):
        return {}


# =====================================================================
# DSL
# =====================================================================


def _warn_if_host_mismatches_target(gpu_target: GPUTarget) -> None:
    """Warn if the compile host arch differs from the target arch.

    Downstream ``OpsUnit.smem_cap`` trusts the host driver to report the
    target's cap; cross-arch silently bakes cubins against the wrong cap (the
    C++ runtime guard still catches them at load).
    """
    # TODO: investigate AMD/HIP -- add an analogous host/target arch check
    # (gfx target string) once the AMD SMEM/LDS path is wired up.
    if gpu_target.backend != "cuda" or not torch.cuda.is_available():
        return
    host_major, host_minor = torch.cuda.get_device_capability(0)
    host_arch = host_major * 10 + host_minor
    if host_arch != gpu_target.arch:
        logger.warning(
            f"[AOTT] Compile host sm_{host_arch} != target "
            f"sm_{gpu_target.arch}; SMEM filter may be miscalibrated."
        )


@dataclass(frozen=True)
class TritonCompileConfig(DslCompileConfig):
    """Triton-specific compile options, threaded via ``CompileContext.dsl_config``.

    auto_tune_cache_overrides: pre-resolved autotune overrides keyed by kernel
        name (an ``AutotuneCache`` each). ``None``/empty disables overrides. The
        internal entry resolves these (e.g. from Manifold) before compile; core
        never fetches them.
    gpu_target: GPU target to compile for. ``None`` (prod default) resolves to
        ``driver.active.get_current_target()`` at compile time (host arch ==
        target arch). Set explicitly only for UT cross-arch coverage.
    drift_check: optional backend-opts drift guardrail invoked with the resolved
        ``gpu_target`` before compile. ``None`` (OSS default) is a no-op; the
        internal entry injects the Meta implementation (guardrails stays fb-only,
        core never imports it).
    """

    auto_tune_cache_overrides: dict[str, AutotuneCache] | None = None
    gpu_target: GPUTarget | None = None
    drift_check: Callable[[GPUTarget], None] | None = None


class TritonAdapter(AOTTAdapter[TritonAOT]):
    """AOT-T DSL integration for native Triton kernels (``@triton_aot``)."""

    name: str = TRITON

    def compile_and_build(self, ctx: CompileContext) -> None:
        cfg = ctx.find_config(TritonCompileConfig)
        gpu_target = (
            cfg.gpu_target
            if cfg and cfg.gpu_target is not None
            else driver.active.get_current_target()
        )
        _warn_if_host_mismatches_target(gpu_target)
        if cfg is not None and cfg.drift_check is not None:
            cfg.drift_check(gpu_target)

        kernel_specs = get_kernel_specs(self.name)
        auto_tune_overrides = (cfg.auto_tune_cache_overrides if cfg else None) or {}

        logger.info(f"[AOTT]: compiling {len(kernel_specs)} kernels")

        for fn, kspec in kernel_specs.items():
            jit_fn = unwrap_to_jit(fn)
            fn_name = jit_fn.__name__

            logger.info(f"[AOTT]: compiling {fn_name} with specs: {kspec.specs}")

            module_suffix = jit_fn.__module__.rsplit(".", 1)[-1]
            fn_dir = f"{ctx.compile_path}/{module_suffix}_{fn_name}"
            os.makedirs(fn_dir, exist_ok=True)

            _resolve_autotune_cache(fn, fn_name, fn_dir, auto_tune_overrides)

            default_values = _extract_default_values(jit_fn)

            compile_to_cpp(
                func=fn,
                base_specs=kspec.specs,
                install_dir=f"{fn_dir}",
                prefix=f"{fn_name}",
                gpu_target=gpu_target,
                tuner_fallback=True,
                import_module=ctx.import_module,
                default_values=default_values,
            )

            build_triton_aot_extension(
                source_dir=fn_dir,
                kernel_name=fn_name,
                output_dir=fn_dir,
                build_config=ctx.extension_build_config,
            )

    def find_kernel(self, node_target: Any) -> TritonAOT | None:
        """Return the single TritonAOT kernel referenced in *node_target*'s
        globals, or ``None`` if it references none. Delegates the scan/
        validation to the DSL-agnostic ``find_sole_marker_in_globals``
        (one-kernel-per-wrapper invariant enforced there); only the Triton
        spec lookup + error message vary.
        """
        kernel_specs = get_kernel_specs(self.name)
        match = find_sole_marker_in_globals(
            node_target,
            TritonAOT,
            in_specs=lambda var: var.fn in kernel_specs,
            missing_spec_error=lambda var: (
                f"Cannot find TritonAOT kernel {var.fn} in TRITON_AOT_KERNEL_SPECS"
            ),
        )
        return match[0] if match is not None else None

    def generate_wrapper_files(
        self,
        node_target: Any,
        match: TritonAOT,
        compile_path: str,
        package_importer: package.PackageImporter | None,
    ) -> None:
        """Generate ``_original.py`` and ``_wrapper.py`` for a single kernel.

        Only the Triton-specific output dir name and the
        ``TritonAOTOperatorTransform`` (which rewrites ``kernel[grid](...)`` into
        ``torch.ops.triton_aot.*``) vary; the file-writing skeleton is shared via
        ``generate_wrapper_files_skeleton``.
        """
        jit_fn = unwrap_to_jit(match)
        module_base = _get_clean_module_basename(jit_fn.__module__)
        kernel_dir = f"{module_base}_{get_kernel_name(jit_fn)}"
        generate_wrapper_files_skeleton(
            node_target,
            kernel_dir=kernel_dir,
            transformer=TritonAOTOperatorTransform(kernel=match),
            compile_path=compile_path,
            package_importer=package_importer,
        )


def collect(
    marker: TritonAOT,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Collect a Triton kernel spec. Invoked via the spec collector that
    ``enable_spec_collection`` registers on ``TritonAOT`` (fired from
    ``TritonAOT.run`` while compile is active). Registers ``TritonAdapter`` as active
    (so the orchestrator dispatches compile to it) and records the spec.
    """
    register_active(TritonAdapter)
    _collect_triton_spec(marker.fn, marker.annotations, *args, **kwargs)


# =====================================================================
# wrapper codegen (Triton): AST rewrite of kernel[grid](...) launch sites
# into torch.ops.triton_aot.* calls. Driven by transform/kernel_wrapper_codegen
# via TritonAdapter.find_kernel / generate_wrapper_files (runtime dispatch).
# =====================================================================


def _calls_triton_aot_kernel(node: ast.FunctionDef, kernel_name: str) -> bool:
    """
    kernel_name is the JIT function name (e.g. "_weighted_layer_norm_fwd"),
    which may differ from the wrapper function name (e.g.
    "_triton_aot_swish_layer_norm").  We match by looking for a
    Subscript-call ``kernel_name[grid](...)`` inside the function body.
    """
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Subscript)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == kernel_name
        ):
            return True
    return False


class TritonAOTOperatorTransform(ast.NodeTransformer):
    def __init__(self, kernel: Any, gpu_target: Optional[GPUTarget] = None) -> None:
        super().__init__()
        self._kernel: Any = kernel
        self.gpu_target: GPUTarget = gpu_target or driver.active.get_current_target()
        self._kernel_jit_fn: JITFunction[List[Any]] = unwrap_to_jit(kernel)
        self._kernel_autotuner: Optional[Autotuner] = try_get_autotuner(kernel)
        self._kernel_name: str = get_kernel_name(self._kernel_jit_fn)
        self._autotune_params: List[str] = compute_autotune_param_names(
            self._kernel_autotuner, self.gpu_target.backend
        )

        self._lambda_arg_name: Optional[str] = None
        self._grid_name: Optional[str] = None
        self._autotune_key_id: Optional[Dict[str, int]] = None
        self._autotune_key_map: Optional[Dict[str, ast.expr]] = None
        self._kernel_meta: Optional[ast.Assign] = None

        if self._kernel_autotuner is not None:
            autotune_key_id: Dict[str, int] = {}
            self._autotune_key_id = autotune_key_id
            for key in self._kernel_autotuner.keys:
                autotune_key_id[key] = self._kernel_jit_fn.arg_names.index(key)

    def generate_function_meta(self) -> None:
        targets = [
            ast.Name(id=param, ctx=ast.Store()) for param in self._autotune_params
        ]
        autotune_key_map = self._autotune_key_map
        kernel_autotuner = self._kernel_autotuner
        call = ast.Call(
            func=ast.Name(id=f"{self._kernel_name}_meta", ctx=ast.Load()),
            args=[
                none_throws(autotune_key_map)[key]
                for key in none_throws(kernel_autotuner).keys
            ]
            if kernel_autotuner is not None
            else [],
            keywords=[],
        )
        self._kernel_meta = ast.Assign(
            # pyre-ignore[6]: ast.Assign targets type
            targets=[ast.Tuple(elts=targets, ctx=ast.Store())],
            value=call,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        strip_jit_unused_decorator(
            node, lambda n: _calls_triton_aot_kernel(n, self._kernel_name)
        )

        new_body: List[ast.stmt] = []
        stmts = node.body
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == self._grid_name:
                        assert self._kernel_meta is not None
                        self._kernel_meta.lineno = stmt.lineno
                        new_body.append(self._kernel_meta)
            new_body.append(self.visit(stmt))
        node.body = new_body
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Lambda):
                lambda_node = node.value
                self._lambda_arg_name = lambda_node.args.args[0].arg
                lambda_body = lambda_node.body
                assert isinstance(lambda_body, ast.Tuple)
                new_elts: List[ast.expr] = []
                for elt in lambda_body.elts:
                    new_elts.append(self.visit(elt))
                node.value = ast.Tuple(elts=new_elts, ctx=ast.Load())
                self._lambda_arg_name = None
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if isinstance(node.value, ast.Name) and node.value.id == self._lambda_arg_name:
            assert isinstance(node.slice, ast.Constant)
            assert isinstance(node.slice.value, str)
            var_name = node.slice.value
            # pyre-ignore
            node = ast.Name(id=var_name, ctx=ast.Load())
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        if isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Subscript)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == self._kernel_name
            ):
                grid_arg = call.func.slice
                new_func = ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="torch", ctx=ast.Load()),
                            attr="ops",
                            ctx=ast.Load(),
                        ),
                        attr="triton_aot",
                        ctx=ast.Load(),
                    ),
                    attr=self._kernel_name,
                    ctx=ast.Load(),
                )
                new_args = [grid_arg] + call.args
                new_keywords = call.keywords + [
                    ast.keyword(arg=param, value=ast.Name(id=param, ctx=ast.Load()))
                    for param in self._autotune_params
                ]
                node.value = ast.Call(
                    func=new_func,
                    args=new_args,
                    keywords=new_keywords,
                )
        return node

    def contains_triton_call(self, node: ast.AST) -> bool:
        """
        A **dual-purpose** scanner:
        1. Detect a ``kernel_name[grid](...)`` invocation in *node*;
        2. on hit, harvest call-site metadata for the wrapper rewrite.

        * ``self._grid_name`` -- the variable name used as the grid
          expression (e.g. ``"grid"`` in ``kernel[grid](...)``).
        * ``self._autotune_key_map`` -- ``{autotune_key_name: ast_expr}``
          mapping each autotune key (e.g. ``"N"``, ``"K"``) to the AST
          expression actually passed at the call site.  Built only when
          the kernel is autotuned.
        * ``self._kernel_meta`` -- populated indirectly via
          ``generate_function_meta()``, which uses the map above to build
          ``(BLOCK_M, ..., num_warps, num_stages) = _kernel_meta(N_expr, K_expr)``.
        """
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Subscript)
                # pyre-ignore[16]: ast.expr may have `id` attribute at runtime
                and child.func.value.id == self._kernel_name
            ):
                # pyrefly: ignore [missing-attribute]
                self._grid_name = child.func.slice.id

                if self._kernel_autotuner is not None:
                    autotune_key_map: Dict[str, ast.expr] = {}
                    self._autotune_key_map = autotune_key_map
                    for key in self._kernel_autotuner.keys:
                        # Prefer kwargs: ``kernel[grid](.., N=N_expr, ..)``.
                        found_key = False
                        for keyword in child.keywords:
                            if keyword.arg == key:
                                autotune_key_map[key] = keyword.value
                                found_key = True
                                break

                        # Fallback: autotune key was passed positionally
                        # (``kernel[grid](.., N_expr, K_expr)``).  Resolve
                        # via signature-index table from __init__.
                        if not found_key:
                            autotune_key_id = self._autotune_key_id
                            assert autotune_key_id is not None
                            assert key in autotune_key_id
                            key_id = autotune_key_id[key]
                            autotune_key_map[key] = child.args[key_id]

                self.generate_function_meta()
                return True
        return False

    def contains_lambda(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Lambda):
                return True
        return False

    def _get_grid_name(self, node: ast.AST) -> Optional[str]:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Subscript)
                # pyre-ignore[16]: ast.expr may have `id` attribute at runtime
                and child.func.value.id == self._kernel_name
            ):
                # pyrefly: ignore [missing-attribute]
                return child.func.slice.id
        return None

    def generate_so_loading_code(
        self,
        node: ast.AST,
        abs_triton_aot_path: str,
    ) -> str:
        """Return auto-generated code to load the compiled kernel at runtime.

        If *node* contains a call to this transformer's kernel, returns
        ``import importlib.util`` + meta-module loading + ``torch.ops.load_library``
        code.  Otherwise returns an empty string.

        This method also sets up internal transformer state (grid name,
        autotune key map, etc.) via ``contains_triton_call`` as a side effect.

        Example for _addmm_fwd kernel:
            kernel_dir = "triton_addmm__addmm_fwd"
            meta_module_path = "/path/to/triton_aot_compile/triton_addmm__addmm_fwd/_addmm_fwd_meta.py"
            so_path = "/path/to/triton_aot_compile/triton_addmm__addmm_fwd/addmm_fwd.so"
        """
        if not self.contains_triton_call(node):
            return ""

        kernel_dir = f"{_get_clean_module_basename(self._kernel_jit_fn.__module__)}_{self._kernel_name}"

        meta_module_path = os.path.join(
            abs_triton_aot_path, kernel_dir, f"{self._kernel_name}_meta.py"
        )

        so_path = os.path.join(
            abs_triton_aot_path,
            kernel_dir,
            f"{self._kernel_name.lstrip('_')}.so",
        )

        return f"""
# Auto-generated by triton_aot.kernel_wrapper_codegen
import importlib.util
_meta_spec = importlib.util.spec_from_file_location("{self._kernel_name}_meta", "{meta_module_path}")
_meta_module = importlib.util.module_from_spec(_meta_spec)
_meta_spec.loader.exec_module(_meta_module)
{self._kernel_name}_meta = _meta_module.{self._kernel_name}_meta

torch.ops.load_library("{so_path}")
"""
