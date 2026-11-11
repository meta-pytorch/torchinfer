# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# CuTeDSL AOT codegen: the per-arg ABI shared by the eager path (eager.py) and
# the compile path (pipeline.py), and the two C++ shims built from it. No
# cute.compile/export and no kernel execution, so it is CPU-testable.
#
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

import re
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import torch
from aot_tensor.arg_specs.common import ConstexprArg, ScalarArg
from aot_tensor.arg_specs.cute import CuTeArgSpec, CuTeTensorArg
from aot_tensor.compile.spec_selector import render_selector, Variant
from aot_tensor.compile.template_utils import CUTEDSL_TEMPLATES
from aot_tensor.constants import DEFAULT_OP_NAMESPACE_PREFIX

_C_IDENTIFIER: re.Pattern[str] = re.compile(r"^[A-Za-z_]\w*$")


@dataclass(frozen=True)
class ScalarDtype:
    """Per-scalar-dtype codegen info: single source of truth across the ABI."""

    cpp_param: str  # torch-op C++ param type
    entry_param: str  # sidecar entry param / function-pointer type
    schema: str  # torch op schema type
    entry_cast: str  # C++ expr casting cpp_param -> entry_param; `{name}` placeholder
    py_cast: Callable[[Any], Any]  # coerce a sampled runtime value (int/float/bool)
    cute_ctor: str = ""  # cutlass scalar class name; "" -> use py_cast directly


_SCALAR_DTYPES: dict[str, ScalarDtype] = {
    # cutedsl_i32 (templates/cutedsl_torch_op.cpp) throws instead of wrapping.
    "i32": ScalarDtype(
        "int64_t", "int32_t", "int", 'cutedsl_i32({name}, "{name}")', int, "Int32"
    ),
    "i64": ScalarDtype(
        "int64_t", "int64_t", "int", "static_cast<int64_t>({name})", int, "Int64"
    ),
    "fp32": ScalarDtype(
        "double", "float", "float", "static_cast<float>({name})", float, "Float32"
    ),
    "bool": ScalarDtype("bool", "bool", "bool", "{name}", bool),
}


# =====================================================================
# small codegen utilities
# =====================================================================


def _check_c_identifier(name: str, what: str) -> None:
    if _C_IDENTIFIER.fullmatch(name) is None:
        raise RuntimeError(f"CuTeAOT: invalid {what} C identifier {name!r}.")


def _quote_cpp_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def cute_scalar(value: Any, dtype: str) -> Any:
    """Build a representative CuTeDSL scalar from a runtime sample value."""
    info = _SCALAR_DTYPES.get(dtype)
    if info is None:
        raise NotImplementedError(
            f"CuTeAOT: default compile-arg scalar dtype {dtype!r} is not supported; "
            "extend _SCALAR_DTYPES to handle it."
        )
    # Dtypes without a cutlass scalar type (e.g. bool) pass through py_cast.
    if not info.cute_ctor:
        return info.py_cast(value)
    # Lazy import: only the cute paths reach here, keep CuTeDSL off the import path.
    import cutlass  # @manual=fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl

    ctor = getattr(cutlass, info.cute_ctor)
    return ctor(info.py_cast(value))


# =====================================================================
# tensor ABI: struct field parsing
# =====================================================================

# export_to_c writes one C struct per tensor, with arrays only for dynamic dims:
#   { void* data; int32_t dynamic_shapes[N]; int64_t dynamic_strides[M]; }
# We read (N, M) back from the header into ``fields`` so every helper agrees on
# which arrays a tensor has (a contiguous tensor has shapes but no strides).

# Per-tensor (num_dynamic_shapes, num_dynamic_strides).
TensorFields = dict[str, tuple[int, int]]

# Dynamic arrays in struct order: (member, torch::stable accessor, cast to i32?).
# The count comes from the matching slot of the tensor's ``fields`` tuple.
_TENSOR_ARRAYS: tuple[tuple[str, str, bool], ...] = (
    ("dynamic_shapes", "sizes", True),
    ("dynamic_strides", "strides", False),
)


def tensor_arrays(
    name: str, fields: TensorFields
) -> Generator[tuple[str, str, int, bool], None, None]:
    """Yield (member, accessor, count, cast_i32) for the arrays this tensor has."""
    for (member, accessor, cast_i32), count in zip(_TENSOR_ARRAYS, fields[name]):
        if count:
            yield member, accessor, count, cast_i32


def parse_tensor_struct_fields(header: str, prefix: str, name: str) -> tuple[int, int]:
    """(num_dynamic_shapes, num_dynamic_strides) read from the generated struct."""
    # Grab the {...} body tied to this struct name, then read array sizes:
    #   typedef struct { void* data; int32_t dynamic_shapes[2]; } pfx_Tensor_x_t;
    # ``[^{}]*`` (not ``.*?`` + DOTALL) keeps the body from spanning into a
    # neighboring struct: the member list is flat (no nested braces), so this
    # matches exactly one struct. A DOTALL ``.*?`` lets re.search anchor at the
    # FIRST struct and run through several, mis-reading a later tensor's array
    # sizes -- which breaks any op whose tensors have heterogeneous dynamic-dim
    # counts (a homogeneous op like the toy happens to dodge it).
    struct_name = f"{prefix}_Tensor_{name}_t"
    match = re.search(
        r"typedef struct\s*\{([^{}]*)\}\s*" + re.escape(struct_name) + r"\s*;",
        header,
    )
    if match is None:
        raise RuntimeError(
            f"CuTeAOT {prefix}: could not find struct {struct_name} in generated header."
        )
    body = match.group(1)
    shapes_m = re.search(r"dynamic_shapes\[(\d+)\]", body)
    strides_m = re.search(r"dynamic_strides\[(\d+)\]", body)
    return (
        int(shapes_m.group(1)) if shapes_m else 0,
        int(strides_m.group(1)) if strides_m else 0,
    )


# =====================================================================
# Each arg kind (tensor, scalar, constexpr) gets one ``ArgCodegen`` subclass that
# emits all of its pieces. ``make_codegen`` is the only place that picks the
# kind; generators just loop over the resulting list.
# =====================================================================


@dataclass(frozen=True)
class EntryArg:
    """One C arg of the sidecar entry: declaration, type, and call expression."""

    decl: str  # e.g. "void* x_data"
    ctype: str  # e.g. "void*"
    call: str  # value the torch op passes, e.g. "x.data_ptr()"


_SpecT = TypeVar("_SpecT", bound=CuTeArgSpec)


class ArgCodegen(Generic[_SpecT]):
    """Per-arg codegen behavior. Defaults cover compile-time (constexpr) args;
    runtime kinds override the pieces they contribute."""

    # Whether the arg crosses the sidecar entry / wrapper boundary.
    is_runtime: bool = True
    # Whether the arg is a ``cute.Tensor`` (emits a header struct).
    is_tensor: bool = False

    def __init__(self, spec: _SpecT) -> None:
        self.spec: _SpecT = spec

    @property
    def name(self) -> str:
        return self.spec.name

    # --- torch op surface (all kinds) ---
    def torch_cpp_param(self) -> str:
        raise NotImplementedError

    def torch_schema_param(self, index: int) -> str:
        raise NotImplementedError

    def torch_checks(self) -> list[str]:
        """C++ statements rejecting a value the compiled kernel can't take."""
        return []

    # --- sidecar entry surface (runtime kinds only) ---
    def entry_args(self, fields: TensorFields) -> list[EntryArg]:
        return []

    def struct_build_lines(self, prefix: str, fields: TensorFields) -> list[str]:
        return []

    def wrapper_arg(self) -> str | None:
        return None

    # --- cute.compile arg from a sampled runtime value ---
    def cute_arg(self, value: Any) -> Any:
        raise NotImplementedError

    # --- spec validation (identifier already checked by caller) ---
    def validate(self, op_name: str) -> None:
        return None


class TensorCodegen(ArgCodegen[CuTeTensorArg]):
    is_tensor = True

    def torch_cpp_param(self) -> str:
        return f"torch::stable::Tensor {self.name}"

    def torch_schema_param(self, index: int) -> str:
        # TODO: take Tensor? like Triton and FlyDSL, as the generated-cpp-stable-abi
        # rule requires for TorchScript; needs an NVIDIA host to validate.
        alias = chr(ord("a") + index)
        return f"Tensor({alias}!) {self.name}"

    def torch_checks(self) -> list[str]:
        # cute_arg compiles with assumed_align=alignment, and the exported C
        # wrapper doesn't check it (the eager JIT executor does).
        n, align = self.name, self.spec.alignment
        return [
            f"if (reinterpret_cast<uintptr_t>({n}.data_ptr()) % {align} != 0) "
            f'cutedsl_fail("{n} must be {align}-byte aligned");',
            *self._compact_checks(),
        ]

    def _compact_checks(self) -> list[str]:
        """mark_compact_shape_dynamic promises the compiler that the compact
        mode's size, and the strides of the modes outside it in stride_order,
        are multiples of divisibility. Eager checks this when marking; the
        exported C wrapper doesn't."""
        spec = self.spec
        order = spec.stride_order
        if spec.compact_mode is None or order is None or spec.divisibility == 1:
            return []
        n, m, div = self.name, spec.compact_mode, spec.divisibility
        checks = [
            f"if ({n}.size({m}) % {div} != 0) "
            f'cutedsl_fail("{n}.size({m}) must be a multiple of {div}");'
        ]
        # Only strides skip size-1 modes: a stride is never used to step over a
        # size-1 mode, but the size itself is the promise (eager rejects 1 too).
        checks += [
            f"if ({n}.size({d}) != 1 && {n}.stride({d}) % {div} != 0) "
            f'cutedsl_fail("{n}.stride({d}) must be a multiple of {div}");'
            for d in order[: order.index(m)]
        ]
        # The selector checks the rank later; this keeps size(d) in range.
        return [f"if ({n}.dim() == {len(order)}) {{", *(f"  {c}" for c in checks), "}"]

    def entry_args(self, fields: TensorFields) -> list[EntryArg]:
        n = self.name
        args = [EntryArg(f"void* {n}_data", "void*", f"{n}.data_ptr()")]
        for _member, accessor, _count, _cast in tensor_arrays(n, fields):
            args.append(
                EntryArg(
                    f"const int64_t* {n}_{accessor}",
                    "const int64_t*",
                    f"{n}.{accessor}().data()",
                )
            )
        return args

    def struct_build_lines(self, prefix: str, fields: TensorFields) -> list[str]:
        n = self.name
        lines = [f"{prefix}_Tensor_{n}_t {n}_t;", f"{n}_t.data = {n}_data;"]
        # Every dim gets a dynamic shape, so num_dynamic_shapes == tensor rank.
        rank = fields[n][0]
        for member, accessor, count, cast_i32 in tensor_arrays(n, fields):
            for slot, src in enumerate(self._dynamic_src_dims(member, count, rank)):
                rhs = f"{n}_{accessor}[{src}]"
                rhs = f"static_cast<int32_t>({rhs})" if cast_i32 else rhs
                lines.append(f"{n}_t.{member}[{slot}] = {rhs};")
        return lines

    def _dynamic_src_dims(self, member: str, count: int, rank: int) -> list[int]:
        """Torch source dim feeding each dynamic struct slot.

        export_to_c compacts the struct arrays to only the dynamic dims, in mode
        order. Every dim has a dynamic shape, so ``dynamic_shapes`` maps 1:1
        (identity). But the static unit-stride ``leading_dim`` is baked as a
        constant and OMITTED from ``dynamic_strides`` -- so each stride slot maps
        to the torch dim skipping ``leading_dim``. Without this, a tensor whose
        unit-stride dim is not dim 0 feeds the wrong (often the static) stride
        into the kernel layout and corrupts every address (misaligned 128-bit
        loads). Falls back to identity when ``leading_dim`` is unset (the
        historical last-dim-contiguous assumption).
        """
        if member == "dynamic_strides" and self.spec.leading_dim is not None and rank:
            dims = [d for d in range(rank) if d != self.spec.leading_dim]
            return dims[:count]
        return list(range(count))

    def wrapper_arg(self) -> str | None:
        return f"&{self.name}_t"

    def cute_arg(self, value: Any) -> Any:
        # Lazy import: CuTe compile path only (keep CuTeDSL off the import path).
        from cutlass.cute.runtime import from_dlpack  # @manual

        # mark_layout_dynamic keeps dtype/rank/alignment/contiguity static but
        # leaves shapes runtime, so one .so serves any shape. leading_dim pins
        # which mode is the static unit-stride dim (default: auto-detect the
        # unique stride-1 mode). compact_mode + divisibility additionally promise
        # the compiler that the compact strides are multiples of `divisibility`
        # elements -- required for kernels whose 128-bit cp.async atoms would
        # otherwise fail IR verification for lack of a provable stride alignment.
        tensor = from_dlpack(value, assumed_align=self.spec.alignment)
        # leading_dim is Optional (None -> auto-detect the unique stride-1 mode),
        # so pass it straight through instead of branching on None.
        tensor = tensor.mark_layout_dynamic(leading_dim=self.spec.leading_dim)
        if self.spec.compact_mode is not None:
            tensor = tensor.mark_compact_shape_dynamic(
                mode=self.spec.compact_mode,
                stride_order=self.spec.stride_order,
                divisibility=self.spec.divisibility,
            )
        return tensor


class ScalarCodegen(ArgCodegen[ScalarArg]):
    @property
    def _info(self) -> ScalarDtype:
        return _SCALAR_DTYPES[self.spec.dtype]

    def torch_cpp_param(self) -> str:
        return f"{self._info.cpp_param} {self.name}"

    def torch_schema_param(self, index: int) -> str:
        return f"{self._info.schema} {self.name}"

    def entry_args(self, fields: TensorFields) -> list[EntryArg]:
        info = self._info
        return [
            EntryArg(
                f"{info.entry_param} {self.name}",
                info.entry_param,
                info.entry_cast.format(name=self.name),
            )
        ]

    def wrapper_arg(self) -> str | None:
        return self.name

    def cute_arg(self, value: Any) -> Any:
        return cute_scalar(value, self.spec.dtype)

    def validate(self, op_name: str) -> None:
        if self.spec.dtype not in _SCALAR_DTYPES:
            raise RuntimeError(
                f"CuTeAOT {op_name}: unsupported scalar dtype {self.spec.dtype}."
            )


class ConstexprCodegen(ArgCodegen[ConstexprArg]):
    is_runtime = False

    def _cpp_type(self) -> str:
        python_type = self.spec.python_type
        if python_type is bool:
            return "bool"
        if python_type is int:
            return "int64_t"
        if python_type is float:
            return "double"
        raise RuntimeError(f"CuTeAOT: unsupported constexpr type for {self.name}.")

    def _schema_type(self) -> str:
        python_type = self.spec.python_type
        if python_type is bool:
            return "bool"
        if python_type is int:
            return "int"
        if python_type is float:
            return "float"
        raise RuntimeError(f"CuTeAOT: unsupported constexpr type for {self.name}.")

    def torch_cpp_param(self) -> str:
        return f"{self._cpp_type()} {self.name}"

    def torch_schema_param(self, index: int) -> str:
        return f"{self._schema_type()} {self.name}"

    def cute_arg(self, value: Any) -> Any:
        return value


def make_codegen(spec: CuTeArgSpec) -> ArgCodegen[Any]:
    """The single point that branches on CuTe arg kind."""
    if isinstance(spec, CuTeTensorArg):
        return TensorCodegen(spec)
    if isinstance(spec, ScalarArg):
        return ScalarCodegen(spec)
    if isinstance(spec, ConstexprArg):
        return ConstexprCodegen(spec)
    raise RuntimeError(f"CuTeAOT: unsupported arg spec {spec!r}.")


def make_codegens(specs: list[CuTeArgSpec]) -> list[ArgCodegen[Any]]:
    return [make_codegen(spec) for spec in specs]


def build_cute_call_args(
    arg_specs: list[CuTeArgSpec], values: dict[str, Any]
) -> tuple[Any, ...]:
    """Convert resolved torch values to CuTeDSL call args + the current stream.
    Shared by the eager path and the C++ compile path."""
    # Local import: cuda is stub-less/GPU-only; keep this module CPU-importable.
    import cuda.bindings.driver as cuda  # @manual  # pyre-ignore[21]: no stubs

    codegens = make_codegens(arg_specs)
    return tuple(cg.cute_arg(values[cg.name]) for cg in codegens) + (
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def entry_args(codegens: list[ArgCodegen[Any]], fields: TensorFields) -> list[EntryArg]:
    """Flat C arg list for ``{prefix}_entry`` (tensors expanded, scalars cast)."""
    return [arg for cg in codegens for arg in cg.entry_args(fields)]


def entry_struct_builds(
    codegens: list[ArgCodegen[Any]], prefix: str, fields: TensorFields
) -> str:
    """C statements filling each ``{prefix}_Tensor_{name}_t`` from the entry args."""
    lines = [line for cg in codegens for line in cg.struct_build_lines(prefix, fields)]
    return "\n  ".join(lines)


def tensor_struct_fields(
    codegens: list[ArgCodegen[Any]], prefix: str, header: str
) -> TensorFields:
    return {
        cg.name: parse_tensor_struct_fields(header, prefix, cg.name)
        for cg in codegens
        if cg.is_tensor
    }


def validate_cutedsl_op(codegens: list[ArgCodegen[Any]], op_name: str) -> None:
    # Variant symbols (``{op_name}__{hash}``) are valid whenever op_name is.
    _check_c_identifier(op_name, "op name")
    for cg in codegens:
        _check_c_identifier(cg.name, "arg name")
        cg.validate(op_name)


# =====================================================================
# variants: one exported specialization each
# =====================================================================


@dataclass(frozen=True)
class CuTeVariant:
    """One exported specialization. ``entry_args``: the flat C args of its
    sidecar entry; both the entry and the torch op's call to it use them."""

    variant: Variant
    fields: TensorFields
    entry_args: tuple[EntryArg, ...]


def make_variant(
    codegens: list[ArgCodegen[Any]], variant: Variant, header: str
) -> CuTeVariant:
    """Build a variant from the ``{symbol}.h`` that ``export_to_c`` wrote."""
    fields = tensor_struct_fields(codegens, variant.symbol, header)
    runtime = [cg for cg in codegens if cg.is_runtime]
    return CuTeVariant(variant, fields, tuple(entry_args(runtime, fields)))


# =====================================================================
# sidecar entry (.cpp) codegen
# =====================================================================


def entry_fn(codegens: list[ArgCodegen[Any]], cute: CuTeVariant) -> str:
    symbol = cute.variant.symbol
    runtime = [cg for cg in codegens if cg.is_runtime]
    entry_params = ", ".join([a.decl for a in cute.entry_args] + ["CUstream stream"])
    struct_builds = entry_struct_builds(runtime, symbol, cute.fields)
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
            "ENTRY_FN": "\n".join(entry_fn(codegens, cute) for cute in variants),
        },
    )


# =====================================================================
# torch op (.cpp) codegen
# =====================================================================


def run_fn(op_name: str, cute: CuTeVariant, cpp_params: str) -> str:
    symbol = cute.variant.symbol
    entry_types = ", ".join([a.ctype for a in cute.entry_args] + ["CUstream"])
    entry_call_args = ", ".join([a.call for a in cute.entry_args] + ["stream"])
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
    checks = [line for cg in codegens for line in cg.torch_checks()]
    op_lines = [f"void cutedsl_op({cpp_params}) {{"]
    op_lines += [f"  {line}" for line in [*checks, *selector]]
    op_lines += ["}", "", f"void cutedsl_dummy_op({cpp_params}) {{}}"]
    run_fns = [run_fn(op_name, cute, cpp_params) for cute in variants]
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
