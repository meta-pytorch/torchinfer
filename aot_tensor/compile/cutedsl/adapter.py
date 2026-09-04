# Copyright (c) Meta Platforms, Inc. and affiliates.


from __future__ import annotations

import ast
import logging
import os
from typing import Any

from aot_tensor.build.cutedsl.extension_builder import build_cutedsl_aot_extension
from aot_tensor.compile.adapter_base import AOTTAdapter, CompileContext, DslSpecStore
from aot_tensor.compile.compile_state import (
    add_spec,
    get_aott_compile_state,
    get_kernel_specs,
    register_active,
)
from aot_tensor.constants import CUTEDSL
from aot_tensor.cute_specs import sample_spec_and_hash
from aot_tensor.transform.wrapper_codegen_utils import (
    find_sole_marker_in_globals,
    generate_wrapper_files_skeleton,
    strip_jit_unused_decorator,
)
from aot_tensor.types import CuTeAOT, get_cutedsl_aot_dir_name
from torch import package
from triton_aot.compile.cutedsl.pipeline import compile_cutedsl_to_cpp

logger: logging.Logger = logging.getLogger(__name__)


# =====================================================================
# spec collection (records into the singleton's per-DSL store)
# =====================================================================


def _collect_cutedsl_spec(
    op: CuTeAOT,
    *args: Any,
    **kwargs: Any,
) -> None:
    spec, hashed = sample_spec_and_hash(
        op.arg_specs, op.name, op.module_basename, *args, **kwargs
    )
    add_spec(CUTEDSL, op, spec, hashed)


# =====================================================================
# DSL
# =====================================================================


class CuTeAdapter(AOTTAdapter[tuple[CuTeAOT, set[str]]]):
    """AOT-T DSL integration for CuTeDSL kernels (``cutedsl_aot`` wrapped)."""

    name: str = CUTEDSL

    def compile_and_build(self, ctx: CompileContext) -> None:
        # TODO(autotune): CuteDSL has no built-in autotuner (unlike Triton's
        # @triton.autotune). To tune, enumerate a config search space
        # (mma_tiler, cluster_shape, use_2cta_instrs, use_tma_store, ...),
        # cute.compile + benchmark each variant here at compile time, and bake
        # only the winning config per spec (AOT: no benchmarking on the serving
        # box). See NVIDIA CuteDSL autotuning_gemm recipe.
        kernel_specs = get_kernel_specs(self.name)
        logger.info(f"[AOTT][CuTeDSL]: compiling {len(kernel_specs)} kernels")

        for op, kspec in kernel_specs.items():
            fn_name = op.name
            fn_dir = f"{ctx.compile_path}/{get_cutedsl_aot_dir_name(op)}"
            os.makedirs(fn_dir, exist_ok=True)

            logger.info(
                f"[AOTT][CuTeDSL]: compiling {fn_name} with specs: {kspec.specs}"
            )

            compile_cutedsl_to_cpp(
                op=op,
                specs=kspec.specs,
                install_dir=fn_dir,
                prefix=fn_name,
            )

            build_cutedsl_aot_extension(
                source_dir=fn_dir,
                kernel_name=fn_name,
                output_dir=fn_dir,
                build_config=ctx.extension_build_config,
            )

    def find_kernel(self, node_target: Any) -> tuple[CuTeAOT, set[str]] | None:
        """Return the single CuTeAOT kernel (paired with the global names it is
        bound to) referenced in *node_target*'s globals, or ``None`` if it
        references none. Delegates the scan/validation to the DSL-agnostic
        ``find_sole_marker_in_globals`` (one-kernel-per-wrapper invariant
        enforced there); only the CuTeDSL spec lookup + error message vary.
        """
        kernel_specs = get_kernel_specs(self.name)
        return find_sole_marker_in_globals(
            node_target,
            CuTeAOT,
            in_specs=lambda var: var in kernel_specs,
            missing_spec_error=lambda var: (
                f"Cannot find CuTeAOT kernel {var.name} in CUTEDSL_AOT_KERNEL_SPECS"
            ),
        )

    def generate_wrapper_files(
        self,
        node_target: Any,
        match: tuple[CuTeAOT, set[str]],
        compile_path: str,
        package_importer: package.PackageImporter | None,
    ) -> None:
        kernel, global_names = match
        generate_wrapper_files_skeleton(
            node_target,
            kernel_dir=get_cutedsl_aot_dir_name(kernel),
            transformer=CuTeAOTOperatorTransform(
                kernel=kernel, global_names=global_names
            ),
            compile_path=compile_path,
            package_importer=package_importer,
            import_filter=_filter_cutedsl_runtime_imports,
        )


def collect(op: CuTeAOT, *args: Any, **kwargs: Any) -> None:
    """Collect a CuTeDSL kernel spec. Invoked via the spec collector that
    ``enable_spec_collection`` registers on ``CuTeAOT`` (fired from
    ``CuTeAOT.__call__`` while compile is active). Registers ``CuTeAdapter`` as
    active (so the orchestrator dispatches compile to it) and records the spec.
    """
    register_active(CuTeAdapter)
    _collect_cutedsl_spec(op, *args, **kwargs)


def get_cutedsl_eager_cache() -> dict[Any, Any]:
    """CuTeDSL eager compile cache, on its ``dsl_state`` slice (created on first
    use). Lives here, not in ``compile_state``, so creating the slice can supply a
    real ``CuTeAdapter`` without a ``compile_state -> adapter`` cycle."""
    dsl_state = get_aott_compile_state().dsl_state
    if CUTEDSL not in dsl_state:
        dsl_state[CUTEDSL] = DslSpecStore(dsl=CuTeAdapter())
    return dsl_state[CUTEDSL].eager_compiled


# =====================================================================
# wrapper codegen (CuTeDSL): AST rewrite of bare CuTeAOT call sites into
# torch.ops.triton_aot.* calls. Driven by transform/kernel_wrapper_codegen
# via CuTeAdapter.find_kernel / generate_wrapper_files (runtime dispatch).
# =====================================================================


def _filter_cutedsl_runtime_imports(import_header: str) -> str:
    """Drop CuTeDSL-only Python imports from generated runtime wrappers."""
    tree = ast.parse(import_header)
    kept: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases = [
                alias
                for alias in node.names
                if not (
                    alias.name == "cutlass"
                    or alias.name.startswith("cutlass.")
                    or alias.name == "cuda"
                    or alias.name.startswith("cuda.")
                )
            ]
            if aliases:
                node.names = aliases
                kept.append(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (
                module == "cutlass"
                or module.startswith("cutlass.")
                or module == "cuda"
                or module.startswith("cuda.")
            ):
                continue
            kept.append(ast.unparse(node))
        else:
            kept.append(ast.unparse(node))
    return "\n".join(kept) + ("\n" if kept else "")


class CuTeAOTOperatorTransform(ast.NodeTransformer):
    def __init__(self, kernel: CuTeAOT, global_names: set[str]) -> None:
        super().__init__()
        self._kernel: CuTeAOT = kernel
        self._global_names: set[str] = global_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Strip @torch.jit.unused: the rewritten body becomes a runnable torch.ops call.
        strip_jit_unused_decorator(node, self._calls_cutedsl_kernel)
        self.generic_visit(node)
        return node

    def _calls_cutedsl_kernel(self, node: ast.FunctionDef) -> bool:
        # Scan only this function's own scope. Skip nested function/lambda/class
        # bodies so an outer wrapper is not flagged (and its @torch.jit.unused
        # stripped) for a call that lives only in a nested scope -- that nested
        # def is handled on its own visit.
        stack: list[ast.AST] = list(node.body)
        while stack:
            child = stack.pop()
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in self._global_names
            ):
                return True
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                continue
            stack.extend(ast.iter_child_nodes(child))
        return False

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self._global_names:
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
                attr=self._kernel.name,
                ctx=ast.Load(),
            )
            return ast.Call(func=new_func, args=node.args, keywords=node.keywords)
        return node

    def contains_cutedsl_call(self, node: ast.AST) -> bool:
        # Module-wide on purpose: called with the whole tree to decide whether to
        # emit SO-loading, so it must see calls in any (incl. nested) scope.
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in self._global_names
            ):
                return True
        return False

    def generate_so_loading_code(
        self,
        node: ast.AST,
        abs_triton_aot_path: str,
    ) -> str:
        if not self.contains_cutedsl_call(node):
            return ""

        kernel_dir = get_cutedsl_aot_dir_name(self._kernel)
        so_path = os.path.join(
            abs_triton_aot_path,
            kernel_dir,
            f"{self._kernel.name.lstrip('_')}.so",
        )

        return f"""
# Auto-generated by triton_aot.kernel_wrapper_codegen
import torch
torch.ops.load_library("{so_path}")
"""
