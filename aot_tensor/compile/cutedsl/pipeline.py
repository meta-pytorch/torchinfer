# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# One CuTeDSL op ships as two .so files. CuTeDSL itself builds the kernel
# (`cute.compile(...).export_to_c(prefix)` -> {prefix}.h + {prefix}.o); we
# only write the two small C++ shims that wrap it and call into it:
#
#   model calls  torch.ops.triton_aot.{op}(Tensor, int, ...)
#        |
#        v
#   {op}_torch_op.cpp      generate_torch_op_content()    --> {ext}.so
#        - registers the torch op (stable ABI, no libtorch link)
#        - checks dtypes, grabs the current CUDA stream
#        - unpacks each Tensor into plain C args: data_ptr + sizes/strides
#        |
#        |  plain C call -- no torch types cross this line
#        v
#   {op}_entry.cpp         generate_sidecar_entry_content() --> {ext}_cutedsl_impl.so
#        - C function {op}_entry(void* data, const int64_t* sizes, ...)
#        - repacks those flat args into the per-tensor structs from {op}.h
#        - loads the kernel and provides the cudart 12.5+ shims it needs
#        |
#        v
#   {op}.h + {op}.o        (CuTeDSL's export_to_c output -- not ours)
#        - the actual kernel launch
#
# Why two .so: the sidecar pulls in cudart 12.5+ (cudaLibrary* runtime API). The
# torch op .so is loaded RTLD_GLOBAL alongside PyTorch's own cudart, so keeping the
# kernel's cudart there risks interposing the wrong version. The sidecar instead
# dlopen's cudart RTLD_LOCAL to quarantine those symbols, leaving the torch op .so
# on the stable ABI / version-stable libcuda driver API. Both .so sit in one dir;
# the torch op locates the sidecar relative to itself via dladdr.
#
# The per-arg codegen model lives in codegen.py; the eager path (the JIT
# used by CuTeAOT.__call__) lives in eager.py.
# =====================================================================

import logging
import os
import textwrap
from typing import Any

import torch
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
from aot_tensor.compile.template_utils import CUTEDSL_TEMPLATES
from aot_tensor.cute_specs import CuTeRawSpec, resolve_runtime_args
from aot_tensor.types import CuTeAOT

logger: logging.Logger = logging.getLogger(__name__)


# =====================================================================
# sidecar entry (.cpp) codegen
# =====================================================================


def generate_sidecar_entry_content(
    op_name: str, codegens: list[ArgCodegen[Any]], prefix: str, fields: TensorFields
) -> str:
    runtime = [cg for cg in codegens if cg.is_runtime]
    entry_args = _entry_args(runtime, fields)
    entry_params = ", ".join([a.decl for a in entry_args] + ["CUstream stream"])
    struct_builds = _entry_struct_builds(runtime, prefix, fields)
    wrapper_parts = [cg.wrapper_arg() for cg in runtime]
    wrapper_args = ", ".join([p for p in wrapper_parts if p is not None] + ["stream"])

    # Only the entry function is per-op. The cudart loader/shim boilerplate is
    # static and lives verbatim in compile/cutedsl/templates/cutedsl_entry.cpp (op_name reaches
    # it through the CUTEDSL_OP_NAME macro defined in the OP_NAME region).
    entry_fn = (
        f'extern "C" int32_t {prefix}_entry({entry_params}) {{\n'
        f"  static {prefix}_Kernel_Module_t module;\n"
        f"  static std::once_flag load_once;\n"
        f"  std::call_once(load_once, []() {{\n"
        f"    {prefix}_Kernel_Module_Load(&module);\n"
        f"  }});\n"
        f"  {struct_builds}\n"
        f"  return cute_dsl_{prefix}_wrapper(&module, {wrapper_args});\n"
        f"}}\n"
    )

    return CUTEDSL_TEMPLATES.render(
        "cutedsl_entry.cpp",
        {
            "HEADER_INCLUDE": f'#include "{prefix}.h"\n',
            "OP_NAME": f'#define CUTEDSL_OP_NAME "{op_name}"\n',
            "ENTRY_FN": entry_fn,
        },
    )


# =====================================================================
# torch op (.cpp) codegen
# =====================================================================


def generate_torch_op_content(
    op_name: str,
    codegens: list[ArgCodegen[Any]],
    values: dict[str, Any],
    prefix: str,
    sidecar_so_name: str,
    fields: TensorFields,
) -> str:
    runtime = [cg for cg in codegens if cg.is_runtime]

    cpp_params = ", ".join(cg.torch_cpp_param() for cg in codegens)
    schema_params = ", ".join(
        cg.torch_schema_param(idx) for idx, cg in enumerate(codegens)
    )
    entry_args = _entry_args(runtime, fields)
    entry_types = ", ".join([a.ctype for a in entry_args] + ["CUstream"])
    entry_call_args = ", ".join([a.call for a in entry_args] + ["stream"])
    # Each guard() bakes in its own indentation; normalize to a 2-space block so
    # the rendered op body is consistent regardless of how many guards there are.
    guard_blocks = [
        textwrap.indent(
            textwrap.dedent(cg.guard(op_name, values[cg.name])).strip(), "  "
        )
        for cg in codegens
    ]
    guards = "\n".join(g for g in guard_blocks if g)

    # The loader boilerplate (stream getter, dladdr sidecar lookup, dlopen/dlsym)
    # is static and lives verbatim in compile/cutedsl/templates/cutedsl_torch_op.cpp; the per-op
    # name/entry symbol/sidecar basename reach it through the OP_NAME macros. Only
    # the typed entry signature and the op body are per-op.
    op_lines = [f"void cutedsl_op({cpp_params}) {{"]
    if guards:
        op_lines.append(guards)
    op_lines += [
        "  CUstream stream = cutedsl_aot_get_current_stream();",
        f"  int32_t result = cutedsl_load_entry()({entry_call_args});",
        "  if (result != 0) {",
        f'    throw std::runtime_error("CuTeAOT {op_name}: generated sidecar returned CUDA error " + std::to_string(result));',
        "  }",
        "}",
        "",
        f"void cutedsl_dummy_op({cpp_params}) {{}}",
    ]
    op_fn = "\n".join(op_lines) + "\n"

    # The torch library registration is static template text; only the op name and
    # schema vary, and they reach it through the OP_NAME macros (the op/dummy_op
    # function names are file-static, so the registration needs no codegen).
    return CUTEDSL_TEMPLATES.render(
        "cutedsl_torch_op.cpp",
        {
            "OP_NAME": (
                f'#define CUTEDSL_OP_NAME "{op_name}"\n'
                f'#define CUTEDSL_ENTRY_SYMBOL "{prefix}_entry"\n'
                f"#define CUTEDSL_SIDECAR_NAME {_quote_cpp_string(sidecar_so_name)}\n"
                f'#define CUTEDSL_OP_SCHEMA "{op_name}({schema_params}) -> ()"\n'
            ),
            "ENTRY_TYPE": f"using EntryFn = int32_t (*)({entry_types});\n",
            "OP_FN": op_fn,
        },
    )


# =====================================================================
# orchestrator
# =====================================================================


def compile_cutedsl_to_cpp(
    op: CuTeAOT,
    specs: list[CuTeRawSpec],
    install_dir: str,
    prefix: str,
) -> None:
    # Imported only in the CuTe compile path so regular Triton AOT users do not
    # need Cutlass/CuTeDSL at import time.
    import cutlass.cute as cute  # @manual=fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl
    import cutlass.cute.export  # @manual=fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl  # noqa: F401

    if torch.version.hip is not None:
        raise NotImplementedError("CuTeDSL AOT is only supported on NVIDIA CUDA.")
    # TODO: support multiple specializations per op (the Triton AOT path already
    # does). Needs a selector in the generated torch op that dispatches on the
    # runtime arg hash_key to the matching cubin/sidecar, like codegen.gen_selector.
    if len(specs) != 1:
        raise NotImplementedError(
            f"CuTeDSL AOT currently supports one specialization per op, got {len(specs)} for {op.name}."
        )

    codegens = _codegens(op.arg_specs)
    validate_cutedsl_op(codegens, op.name, prefix)
    os.makedirs(install_dir, exist_ok=True)

    spec = specs[0]

    # cute.compile args: per-arg values (dlpack tensors with dynamic layout,
    # CuTeDSL scalars, passthrough constexprs) plus the current stream.
    values = resolve_runtime_args(
        op.arg_specs, op.name, *spec.runtime_args, **spec.runtime_kwargs
    )
    cute_args = build_cute_call_args(op.arg_specs, values)

    logger.info("[AOTT][CuTeDSL]: compiling %s", op.name)
    compiled: Any = cute.compile(op.jit_fn, *cute_args)
    compiled.export_to_c(install_dir, prefix)

    with open(os.path.join(install_dir, f"{prefix}.h")) as fp:
        header = fp.read()
    fields = tensor_struct_fields(codegens, prefix, header)

    with open(os.path.join(install_dir, f"{prefix}_entry.cpp"), "w") as fp:
        fp.write(generate_sidecar_entry_content(op.name, codegens, prefix, fields))

    sidecar_so_name = f"{prefix.lstrip('_')}_cutedsl_impl.so"
    with open(os.path.join(install_dir, f"{prefix}_torch_op.cpp"), "w") as fp:
        fp.write(
            generate_torch_op_content(
                op.name, codegens, values, prefix, sidecar_so_name, fields
            )
        )
