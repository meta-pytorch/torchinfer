# Copyright (c) Meta Platforms, Inc. and affiliates.

"""DSL-agnostic helpers shared by the per-DSL wrapper-codegen logic.

Used by both ``compile/triton/adapter.py`` and ``compile/cutedsl/adapter.py``:
the globals kernel scan (``find_sole_marker_in_globals``), the file-writing
skeleton (``generate_wrapper_files_skeleton``), torch.package source extraction,
and the @torch.jit.unused strip. They live here -- not in the driver
(``kernel_wrapper_codegen.py``) -- so each DSL imports them without depending on
the driver, keeping the driver's only edge to the DSLs a runtime one
(``dsl_state``).
"""

import ast
import inspect
import os
from typing import Any, Callable, Optional, Protocol

from aot_tensor.constants import generated_header
from aot_tensor.transform.import_utils import (
    get_original_import_header,
    rewrite_package_imports,
)
from pyre_extensions import none_throws
from torch import package

# Module prefixes allowed on the torch.package source-extraction path
# (aot_tensor.ops is here ahead of the ops/ move).
_ALLOWED_MODULE_PREFIXES: tuple[str, ...] = (
    "triton_aot.ops",
    "prime_perf_optimizer",
    "aot_tensor.ops",
)


def _is_torch_package_module(module_name: str) -> bool:
    """Check if a module name is from torch.package namespace."""
    return module_name.startswith("<torch_package")


def _strip_torch_package_prefix(module_name: str) -> str:
    """Strip the torch.package namespace prefix from a module name.

    Example:
        '<torch_package_0>.triton_aot.ops.triton_layer_norm'
        -> 'triton_aot.ops.triton_layer_norm'
    """
    if _is_torch_package_module(module_name):
        # Remove '<torch_package_N>.' prefix
        return module_name.split(".", 1)[1]
    return module_name


def _get_clean_module_basename(module_name: str) -> str:
    """Get the basename of a module, stripping torch.package prefix if present.

    Example:
        '<torch_package_0>.triton_aot.ops.triton_layer_norm'
        -> 'triton_layer_norm'
        'triton_aot.ops.triton_layer_norm'
        -> 'triton_layer_norm'
    """
    clean_name = _strip_torch_package_prefix(module_name)
    return clean_name.rsplit(".", 1)[-1]


def _extract_function_source(module_source: str, fn_name: str) -> str:
    """Extract a function's source code from module source.

    Parses the module source and extracts just the function definition.
    """
    tree = ast.parse(module_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.unparse(node)
    raise ValueError(f"Function '{fn_name}' not found in module source")


def _get_module_and_source(
    target: Callable[..., Any],
    package_importer: Optional[package.PackageImporter],
) -> tuple[Any, str, str]:
    """Get module, module source, and function source for a callable.

    Handles both regular modules and torch.package loaded modules.

    Args:
        target: The callable (function) to get source for
        package_importer: Optional PackageImporter for torch.package modules

    Returns:
        Tuple of (module, module_source, function_source)
    """
    module_name = target.__module__
    fn_name = target.__name__

    if _is_torch_package_module(module_name) and package_importer is not None:
        # Handle torch.package namespace
        real_module_name = _strip_torch_package_prefix(module_name)
        assert real_module_name.startswith(_ALLOWED_MODULE_PREFIXES), (
            f"Expected module under one of {_ALLOWED_MODULE_PREFIXES}, "
            f"got: {real_module_name}"
        )

        # Get module source from package
        module_source = package_importer.get_source(real_module_name)

        # Import the module through the package importer
        fn_module = package_importer.import_module(real_module_name)

        # Extract function source from module source
        fn_source = _extract_function_source(module_source, fn_name)

        return fn_module, module_source, fn_source
    else:
        # Standard module handling
        fn_module = inspect.getmodule(target)
        module_source = inspect.getsource(none_throws(fn_module))
        fn_source = inspect.getsource(target)

        return fn_module, module_source, fn_source


def _is_torch_jit_unused(d: ast.expr) -> bool:
    """Check if a decorator AST node represents @torch.jit.unused."""
    return (
        isinstance(d, ast.Attribute)
        and d.attr == "unused"
        and isinstance(d.value, ast.Attribute)
        and d.value.attr == "jit"
        and isinstance(d.value.value, ast.Name)
        and d.value.value.id == "torch"
    )


def strip_jit_unused_decorator(
    node: ast.FunctionDef,
    calls_kernel: Callable[[ast.FunctionDef], bool],
) -> ast.FunctionDef:
    """Strip @torch.jit.unused from the launcher whose body becomes a runnable
    torch.ops call. ``calls_kernel`` decides whether ``node`` is that launcher
    (Triton: a ``kernel[grid](...)`` subscript call; CuTeDSL: a bare-name call to
    the CuTeAOT global), so the same strip logic serves both transforms.
    """
    if calls_kernel(node):
        node.decorator_list = [
            d for d in node.decorator_list if not _is_torch_jit_unused(d)
        ]
    return node


def find_sole_marker_in_globals(
    node_target: Any,
    marker_type: type,
    in_specs: Callable[[Any], bool],
    missing_spec_error: Callable[[Any], str],
) -> Optional[tuple[Any, set[str]]]:
    """Return the single ``marker_type`` marker referenced in
    ``node_target.__globals__`` paired with the global names it is bound to, or
    ``None`` if the wrapper references none.

    Shared by the per-DSL ``find_kernel`` methods; each injects only the
    DSL-specific spec lookup:
    - ``in_specs(var)``: whether the marker's spec was collected.
    - ``missing_spec_error(var)``: ``RuntimeError`` message when a referenced
      marker has no collected spec.

    Scans every ``marker_type`` binding in ``__globals__``, called or not (one
    code path for both DSLs). One-kernel-per-wrapper is enforced by the
    ``len(markers) == 1`` check: an imported-but-uncalled extra marker trips that
    assertion (or the missing-spec ``RuntimeError``) instead of being silently
    ignored -- intended, stricter than the old call-site-filtered CuTeDSL scan.
    """
    markers: set[Any] = set()
    global_names: set[str] = set()
    for name, var in node_target.__globals__.items():
        if not isinstance(var, marker_type):
            continue
        if not in_specs(var):
            raise RuntimeError(missing_spec_error(var))
        markers.add(var)
        global_names.add(name)

    if not markers:
        return None

    assert len(markers) == 1, (
        f"Expected exactly 1 kernel per wrapper function "
        f"'{node_target.__name__}', got {len(markers)}"
    )
    (marker,) = markers
    return marker, global_names


class AOTTOperatorTransform(Protocol):
    """Structural type for a per-DSL wrapper AST transformer, so
    ``generate_wrapper_files_skeleton`` is type-checked instead of ``Any``. Both
    ``TritonAOTOperatorTransform`` and ``CuTeAOTOperatorTransform`` already match
    it (they subclass ``ast.NodeTransformer`` and define
    ``generate_so_loading_code``), so neither inherits it explicitly.

    Ordering contract: ``generate_so_loading_code`` runs *before* ``visit`` and
    may prime transformer state for it (e.g. Triton harvests the grid name /
    autotune-key map while scanning for the kernel call), so the skeleton must
    keep that order.
    """

    def generate_so_loading_code(
        self, node: ast.AST, abs_triton_aot_path: str
    ) -> str: ...

    def visit(self, node: ast.AST) -> ast.AST: ...


def generate_wrapper_files_skeleton(
    node_target: Any,
    kernel_dir: str,
    transformer: AOTTOperatorTransform,
    compile_path: str,
    package_importer: Optional[package.PackageImporter],
    import_filter: Optional[Callable[[str], str]] = None,
) -> None:
    """Emit ``{fn}_original.py`` + ``{fn}_wrapper.py`` for one wrapper function.

    The DSL-invariant skeleton behind every ``AOTTAdapter.generate_wrapper_files``:
    write the untouched source as ``_original.py``, then rewrite the launcher
    body via ``transformer`` and write ``_wrapper.py``. Only three things vary
    per DSL and are injected:
    - ``kernel_dir``: per-kernel output subdir name (DSL-specific derivation).
    - ``transformer``: the DSL's ``ast.NodeTransformer`` (rewrites launch sites
      to ``torch.ops.triton_aot.*`` and emits the .so-loading preamble).
    - ``import_filter``: optional final pass over the wrapper import header
      (CuTeDSL drops cutlass/cuda-only imports; Triton passes ``None``).
    """
    fn_name = node_target.__name__
    output_dir = os.path.join(compile_path, kernel_dir)
    os.makedirs(output_dir, exist_ok=True)

    _, module_code, wrapper_code = _get_module_and_source(node_target, package_importer)
    import_header = get_original_import_header(module_code)

    with open(os.path.join(output_dir, f"{fn_name}_original.py"), "w") as f:
        f.write(generated_header("#"))
        f.write(import_header)
        f.write(wrapper_code)

    # Rewrite interned torch.package imports (e.g. hammer.*) before appending the
    # auto-generated .so-loading preamble, so stdlib imports there are untouched.
    if package_importer is not None:
        import_header = rewrite_package_imports(import_header, package_importer)
    if import_filter is not None:
        import_header = import_filter(import_header)

    tree = ast.parse(wrapper_code)
    # Must run before ``visit`` -- it may prime transformer state (see Protocol).
    import_header += transformer.generate_so_loading_code(tree, compile_path)
    tree = transformer.visit(tree)

    with open(os.path.join(output_dir, f"{fn_name}_wrapper.py"), "w") as f:
        f.write(generated_header("#"))
        f.write(import_header)
        f.write(ast.unparse(tree))
