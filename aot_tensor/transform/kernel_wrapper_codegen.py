# pyre-strict
from typing import Any, Callable

from aot_tensor.compile.compile_state import (
    assert_aott_compile_session_completed,
    get_aott_compile_path,
    get_aott_compile_state,
)
from torch import package
from torch.fx import GraphModule


def kernel_wrapper_codegen(
    module: GraphModule, packageImporter: package.PackageImporter | None = None
) -> None:
    """
    Generate wrapper files for AOT-T kernels of every DSL that collected specs.
    Requirement: under wrapper.py, @triton.jit kernel/func is imported without 'as' alias.

    For each function containing AOT-T kernels, generates:
    - {fn_name}_original.py: Original source code with imports
    - {fn_name}_wrapper.py: Transformed wrapper that uses torch.ops.triton_aot

    DSL-agnostic: each active DSL (from ``dsl_state``) says whether it owns the
    kernel (``find_kernel``) and emits its own files (``generate_wrapper_files``);
    a wrapper may belong to at most one DSL. ``transform`` must not import
    ``compile/<dsl>/*`` -- adapters import ``transform``, not the reverse -- to
    keep the dependency acyclic.
    """
    assert_aott_compile_session_completed()
    compile_path = get_aott_compile_path()
    dsls = [store.dsl for store in get_aott_compile_state().dsl_state.values()]
    transformed_ops: set[Callable[..., Any]] = set()
    for node in module.graph.nodes:
        if node.op == "call_function" and hasattr(node.target, "__globals__"):
            if node.target in transformed_ops:
                continue
            transformed_ops.add(node.target)

            matches = [
                (dsl, match)
                for dsl in dsls
                if (match := dsl.find_kernel(node.target)) is not None
            ]

            if len(matches) > 1:
                names = [dsl.name for dsl, _ in matches]
                raise RuntimeError(
                    f"Wrapper function {node.target.__name__} contains kernels "
                    f"from multiple DSLs {names}; split them into separate "
                    "wrappers."
                )

            if matches:
                dsl, match = matches[0]
                dsl.generate_wrapper_files(
                    node.target, match, compile_path, packageImporter
                )
