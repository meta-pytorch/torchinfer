# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# One CuTeDSL op ships as two .so files. CuTeDSL itself builds the kernel
# (`cute.compile(...).export_to_c(prefix)` -> {prefix}.h + {prefix}.o); we
# only write the two small C++ shims that wrap it and call into it:
#
#   model calls  torch.ops.aot_tensor.{op}(Tensor, int, ...)
#        |
#        v
#   {op}_torch_op.cpp      generate_torch_op_content()    --> {ext}.so
#        - registers the torch op (stable ABI, no libtorch link)
#        - selects the variant matching the call (``spec_selector``), grabs
#          the current CUDA stream
#        - unpacks each Tensor into plain C args: data_ptr + sizes/strides
#        |
#        |  plain C call -- no torch types cross this line
#        v
#   {op}_entry.cpp         generate_sidecar_entry_content() --> {ext}_cutedsl_impl.so
#        - one C function {symbol}_entry(void* data, const int64_t* sizes, ...)
#          per variant, symbol = {op}__{spec_hash[:8]}
#        - repacks those flat args into the per-tensor structs from {symbol}.h
#        - loads the kernel and provides the cudart 12.5+ shims it needs
#        |
#        v
#   {symbol}.h + {symbol}.o  (CuTeDSL's export_to_c output -- not ours)
#        - the actual kernel launch
#
# Why two .so: the sidecar pulls in cudart 12.5+ (cudaLibrary* runtime API). The
# torch op .so is loaded RTLD_GLOBAL alongside PyTorch's own cudart, so keeping the
# kernel's cudart there risks interposing the wrong version. The sidecar instead
# dlopen's cudart RTLD_LOCAL to quarantine those symbols, leaving the torch op .so
# on the stable ABI / version-stable libcuda driver API. Both .so sit in one dir;
# the torch op locates the sidecar relative to itself via dladdr. Every variant
# links into the one sidecar: export_to_c prefixes all its global symbols.
#
# The per-arg codegen model lives in codegen.py; the eager path (the JIT
# used by CuTeAOT.__call__) lives in eager.py.
# =====================================================================

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from aot_tensor.arg_specs.common import RawSpec, resolve_runtime_args
from aot_tensor.compile.adapter_base import hash_spec
from aot_tensor.compile.cutedsl.codegen import (
    _codegens,
    _entry_args,
    _entry_struct_builds,
    _quote_cpp_string,
    ArgCodegen,
    build_cute_call_args,
    tensor_struct_fields,
    TensorFields,
    validate_cutedsl_op,
)
from aot_tensor.compile.spec_selector import distinct_variants, render_selector, Variant
from aot_tensor.compile.template_utils import CUTEDSL_TEMPLATES
from aot_tensor.constants import DEFAULT_OP_NAMESPACE_PREFIX
from aot_tensor.types import CuTeAOT

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CuTeVariant:
    """One exported specialization: its selector entry and the tensor struct
    fields its ``{symbol}.h`` declares."""

    variant: Variant
    fields: TensorFields


# =====================================================================
# sidecar entry (.cpp) codegen
# =====================================================================


def _entry_fn(codegens: list[ArgCodegen[Any]], cute: CuTeVariant) -> str:
    symbol = cute.variant.symbol
    runtime = [cg for cg in codegens if cg.is_runtime]
    entry_args = _entry_args(runtime, cute.fields)
    entry_params = ", ".join([a.decl for a in entry_args] + ["CUstream stream"])
    struct_builds = _entry_struct_builds(runtime, symbol, cute.fields)
    wrapper_parts = [cg.wrapper_arg() for cg in runtime]
    wrapper_args = ", ".join([p for p in wrapper_parts if p is not None] + ["stream"])
    return (
        f'extern "C" int32_t {symbol}_entry({entry_params}) {{\n'
        f"  static {symbol}_Kernel_Module_t module;\n"
        f"  static std::once_flag load_once;\n"
        f"  std::call_once(load_once, []() {{\n"
        f"    {symbol}_Kernel_Module_Load(&module);\n"
        f"  }});\n"
        f"  {struct_builds}\n"
        f"  return cute_dsl_{symbol}_wrapper(&module, {wrapper_args});\n"
        f"}}\n"
    )


def generate_sidecar_entry_content(
    op_name: str, codegens: list[ArgCodegen[Any]], variants: list[CuTeVariant]
) -> str:
    # Only the entry functions are per-op. The cudart loader/shim boilerplate is
    # static and lives verbatim in compile/cutedsl/templates/cutedsl_entry.cpp (op_name reaches
    # it through the CUTEDSL_OP_NAME macro defined in the OP_NAME region).
    return CUTEDSL_TEMPLATES.render(
        "cutedsl_entry.cpp",
        {
            "HEADER_INCLUDE": "".join(
                f'#include "{cute.variant.symbol}.h"\n' for cute in variants
            ),
            "OP_NAME": f'#define CUTEDSL_OP_NAME "{op_name}"\n',
            "ENTRY_FN": "\n".join(_entry_fn(codegens, cute) for cute in variants),
        },
    )


# =====================================================================
# torch op (.cpp) codegen
# =====================================================================


def _run_fn(
    op_name: str, codegens: list[ArgCodegen[Any]], cute: CuTeVariant, cpp_params: str
) -> str:
    symbol = cute.variant.symbol
    runtime = [cg for cg in codegens if cg.is_runtime]
    entry_args = _entry_args(runtime, cute.fields)
    entry_types = ", ".join([a.ctype for a in entry_args] + ["CUstream"])
    entry_call_args = ", ".join([a.call for a in entry_args] + ["stream"])
    lines = [
        f"void {symbol}_run({cpp_params}) {{",
        f"  using EntryFn = int32_t (*)({entry_types});",
        f'  static EntryFn entry = cutedsl_load_entry<EntryFn>("{symbol}_entry");',
        "  CUstream stream = cutedsl_aot_get_current_stream();",
        f"  int32_t result = entry({entry_call_args});",
        "  if (result != 0) {",
        f'    throw std::runtime_error("CuTeAOT {op_name}: generated sidecar returned CUDA error " + std::to_string(result));',
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def generate_torch_op_content(
    op_name: str,
    codegens: list[ArgCodegen[Any]],
    variants: list[CuTeVariant],
    sidecar_so_name: str,
) -> str:
    cpp_params = ", ".join(cg.torch_cpp_param() for cg in codegens)
    schema_params = ", ".join(
        cg.torch_schema_param(idx) for idx, cg in enumerate(codegens)
    )
    call_args = ", ".join(cg.name for cg in codegens)
    selector = render_selector(
        [cute.variant for cute in variants],
        {
            cute.variant.symbol: f"{cute.variant.symbol}_run({call_args})"
            for cute in variants
        },
        fail="cutedsl_fail",
    )

    # The loader boilerplate (stream getter, dladdr sidecar lookup, dlopen/dlsym)
    # is static and lives verbatim in compile/cutedsl/templates/cutedsl_torch_op.cpp; the per-op
    # name and sidecar basename reach it through the OP_NAME macros. Only the
    # per-variant run functions and the dispatching op body are per-op.
    op_lines = [f"void cutedsl_op({cpp_params}) {{"]
    op_lines += [f"  {line}" for line in selector]
    op_lines += ["}", "", f"void cutedsl_dummy_op({cpp_params}) {{}}"]
    run_fns = [_run_fn(op_name, codegens, cute, cpp_params) for cute in variants]
    op_fn = "\n".join([*run_fns, *op_lines]) + "\n"

    # Registration is static template text; namespace, op name and schema come in
    # through the OP_NAME macros.
    return CUTEDSL_TEMPLATES.render(
        "cutedsl_torch_op.cpp",
        {
            "OP_NAME": (
                f"#define CUTEDSL_OP_NAMESPACE {DEFAULT_OP_NAMESPACE_PREFIX}\n"
                f'#define CUTEDSL_OP_NAME "{op_name}"\n'
                f"#define CUTEDSL_SIDECAR_NAME {_quote_cpp_string(sidecar_so_name)}\n"
                f'#define CUTEDSL_OP_SCHEMA "{op_name}({schema_params}) -> ()"\n'
            ),
            "OP_FN": op_fn,
        },
    )


# =====================================================================
# orchestrator
# =====================================================================


def compile_cutedsl_to_cpp(
    op: CuTeAOT,
    specs: list[RawSpec],
    install_dir: str,
) -> list[str]:
    """Export one kernel object per distinct specialization, plus the sidecar
    entry and the torch op dispatching between them; returns the object paths
    to link into the sidecar."""
    # Imported only in the CuTe compile path so regular Triton AOT users do not
    # need Cutlass/CuTeDSL at import time.
    import cutlass.cute as cute  # @manual=fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl
    import cutlass.cute.export  # @manual=fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl  # noqa: F401

    if torch.version.hip is not None:
        raise NotImplementedError("CuTeDSL AOT is only supported on NVIDIA CUDA.")
    # TODO: share the torch-op arg codegen (scalar/constexpr tables, schema,
    # C++ params) with FlyDSL.

    codegens = _codegens(op.arg_specs)
    validate_cutedsl_op(codegens, op.name)
    os.makedirs(install_dir, exist_ok=True)

    keyed = [(spec, hash_spec(spec.hash_key), spec.hash_key) for spec in specs]
    variants: list[CuTeVariant] = []
    for spec, variant in distinct_variants(op.name, keyed):
        # cute.compile args: per-arg values (dlpack tensors with dynamic layout,
        # CuTeDSL scalars, passthrough constexprs) plus the current stream.
        values = resolve_runtime_args(
            op.arg_specs, op.name, *spec.runtime_args, **spec.runtime_kwargs
        )
        cute_args = build_cute_call_args(op.arg_specs, values)

        logger.info("[AOTT][CuTeDSL]: compiling %s (%s)", op.name, variant.description)
        compiled: Any = cute.compile(op.jit_fn, *cute_args)
        compiled.export_to_c(install_dir, variant.symbol)

        with open(os.path.join(install_dir, f"{variant.symbol}.h")) as fp:
            header = fp.read()
        fields = tensor_struct_fields(codegens, variant.symbol, header)
        variants.append(CuTeVariant(variant, fields))

    with open(os.path.join(install_dir, f"{op.name}_entry.cpp"), "w") as fp:
        fp.write(generate_sidecar_entry_content(op.name, codegens, variants))

    sidecar_so_name = f"{op.name.lstrip('_')}_cutedsl_impl.so"
    with open(os.path.join(install_dir, f"{op.name}_torch_op.cpp"), "w") as fp:
        fp.write(
            generate_torch_op_content(op.name, codegens, variants, sidecar_so_name)
        )
    return [os.path.join(install_dir, f"{cute.variant.symbol}.o") for cute in variants]
