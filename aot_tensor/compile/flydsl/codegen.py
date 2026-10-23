# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# Torch-op codegen for exported FlyDSL launchers, one per specialization. Each
# launcher's packed ABI (``AbiSlot``) drives its run function: one local and one
# ``void* args[]`` pointer per slot. The op picks a run function with
# ``spec_selector``. No flydsl import, so this is CPU-testable.

from dataclasses import dataclass
from typing import Any

from aot_tensor.arg_specs.common import ConstexprArg, ScalarArg
from aot_tensor.arg_specs.fly import FlyArgSpec, FlyTensorArg
from aot_tensor.compile.flydsl.abi import AbiSlot, C_IDENTIFIER, ExportedLauncher
from aot_tensor.compile.spec_selector import render_selector, Variant
from aot_tensor.compile.template_utils import FLYDSL_TEMPLATES
from aot_tensor.constants import DEFAULT_OP_NAMESPACE_PREFIX

# The generated regions of templates/flydsl_torch_op.cpp.
_OP_NAME_TMPL = """\
#define FLYDSL_OP_NAMESPACE {namespace}
#define FLYDSL_OP_NAME "{op_name}"
#define FLYDSL_OP_SCHEMA "{op_name}({schema_params}) -> ()"
"""

_EXTERNS_TMPL = """\
extern "C" void {entry}(void** args);
extern "C" void {module_init}(void** module, int32_t* err);
extern "C" void {module_load}(void** module, int32_t* err);
"""

_RUN_FN_TMPL = """\
void {symbol}_run({params}) {{
  {body}
}}
"""

_OP_FN_TMPL = """\
void flydsl_op({params}) {{
  {body}
}}

void flydsl_dummy_op({params}) {{}}
"""


@dataclass(frozen=True)
class _ScalarDtype:
    cpp_param: str  # torch-op C++ param type
    schema: str  # torch op schema type
    abi_ctype: str  # AbiSlot.ctype FlyDSL reports for this dtype
    storage: str  # C type written into the packed slot


_SCALAR_DTYPES: dict[str, _ScalarDtype] = {
    # SymInt so export keeps a dynamic size (e.g. numel()) dynamic; the stable
    # ABI still passes an int.
    "i32": _ScalarDtype("int64_t", "SymInt", "int32", "int32_t"),
    "i64": _ScalarDtype("int64_t", "SymInt", "int64", "int64_t"),
    "fp32": _ScalarDtype("double", "float", "float", "float"),
    "bool": _ScalarDtype("bool", "bool", "bool", "bool"),
}

_CONSTEXPR_TYPES: dict[type[Any], tuple[str, str]] = {
    bool: ("bool", "bool"),
    int: ("int64_t", "int"),
    float: ("double", "float"),
}


def _fail(op_name: str, message: str) -> RuntimeError:
    return RuntimeError(f"FlyAOT {op_name}: {message}")


def _require(condition: str, message: str) -> str:
    return f'if (!({condition})) flydsl_fail("{message}");'


# =====================================================================
# torch-facing surface (one entry per arg spec)
# =====================================================================


def _cpp_param(spec: FlyArgSpec) -> str:
    if isinstance(spec, FlyTensorArg):
        return f"torch::stable::Tensor {spec.name}"
    if isinstance(spec, ScalarArg):
        return f"{_SCALAR_DTYPES[spec.dtype].cpp_param} {spec.name}"
    return f"{_CONSTEXPR_TYPES[spec.python_type][0]} {spec.name}"


def _op_param(spec: FlyArgSpec) -> str:
    # Tensors arrive as Tensor? (TorchScript compatibility) and are unwrapped by
    # _unwrap_tensors before dispatch.
    if isinstance(spec, FlyTensorArg):
        return f"std::optional<torch::stable::Tensor> {spec.name}_opt"
    return _cpp_param(spec)


def _unwrap_tensors(arg_specs: list[FlyArgSpec]) -> list[str]:
    return [
        f'const torch::stable::Tensor {s.name} = flydsl_required({s.name}_opt, "{s.name}");'
        for s in arg_specs
        if isinstance(s, FlyTensorArg)
    ]


def _schema_param(index: int, spec: FlyArgSpec, mutated: frozenset[str]) -> str:
    if isinstance(spec, FlyTensorArg):
        if spec.name not in mutated:
            return f"Tensor? {spec.name}"
        return f"Tensor({chr(ord('a') + index)}!)? {spec.name}"
    if isinstance(spec, ScalarArg):
        return f"{_SCALAR_DTYPES[spec.dtype].schema} {spec.name}"
    return f"{_CONSTEXPR_TYPES[spec.python_type][1]} {spec.name}"


# =====================================================================
# packed slots (one entry per ABI slot)
# =====================================================================


def _layout_writes(slot: AbiSlot, local: str) -> list[str]:
    n = slot.arg_name
    lines: list[str] = []
    offset = 0
    for dim in slot.shape_dims:
        lines.append(
            f'flydsl_put<int32_t>({local}, sizeof({local}), {offset}, flydsl_i32({n}.size({dim}), "{n}.size({dim})"));'
        )
        offset += 4
    stride_type = "int32_t" if slot.stride_bits == 32 else "int64_t"
    for dim in slot.stride_dims:
        value = f"{n}.stride({dim})"
        if slot.stride_bits == 32:
            value = f'flydsl_i32({value}, "{n}.stride({dim})")'
        lines.append(
            f"flydsl_put<{stride_type}>({local}, sizeof({local}), {offset}, {value});"
        )
        offset += slot.stride_bits // 8
    return lines


def _layout_checks(slot: AbiSlot, spec: FlyTensorArg) -> list[str]:
    """The launcher was compiled assuming these (``mark_layout_dynamic``): the
    leading dim, the one without a stride in the ABI, has stride 1, and every
    other stride is a multiple of ``divisibility``."""
    n = slot.arg_name
    lines = [
        _require(f"{n}.stride({d}) == 1", f"{n} must have stride 1 in dim {d}")
        for d in range(len(slot.shape_dims))
        if d not in slot.stride_dims
    ]
    if spec.divisibility > 1:
        lines += [
            _require(
                f"{n}.stride({d}) % {spec.divisibility} == 0",
                f"{n}.stride({d}) must be a multiple of {spec.divisibility}",
            )
            for d in slot.stride_dims
        ]
    return lines


@dataclass(frozen=True)
class _Specs:
    tensors: dict[str, FlyTensorArg]
    scalars: dict[str, ScalarArg]


def _slot_lines(index: int, slot: AbiSlot, specs: _Specs) -> tuple[list[str], str]:
    """(statements filling slot ``index``, its ``void*`` in ``args[]``)."""
    local = f"slot{index}"
    if slot.kind == "stream":
        return [f"void* {local} = flydsl_current_stream(device_index);"], f"&{local}"
    if slot.kind == "tensor_data":
        spec = specs.tensors[slot.arg_name]
        n = slot.arg_name
        check = _require(
            f"reinterpret_cast<uintptr_t>({n}.data_ptr()) % {spec.alignment} == 0",
            f"{n} must be {spec.alignment}-byte aligned",
        )
        return [check, f"void* {local} = {n}.data_ptr();"], f"&{local}"
    if slot.kind == "tensor_layout":
        decl = f"alignas(8) uint8_t {local}[{max(slot.size, 1)}] = {{}};"
        return [
            *_layout_checks(slot, specs.tensors[slot.arg_name]),
            decl,
            *_layout_writes(slot, local),
        ], f"static_cast<void*>({local})"
    spec = specs.scalars[slot.arg_name]
    info = _SCALAR_DTYPES[spec.dtype]
    value = spec.name
    # i32 overflow would silently wrap, so it is checked. fp32 is not: double ->
    # float rounds (or goes to inf) exactly as eager's Float32 conversion does.
    if spec.dtype == "i32":
        value = f'flydsl_i32({spec.name}, "{spec.name}")'
    return [f"{info.storage} {local} = static_cast<{info.storage}>({value});"], (
        f"&{local}"
    )


# =====================================================================
# validation + render
# =====================================================================


def _validate_specs(op_name: str, arg_specs: list[FlyArgSpec]) -> None:
    for name, what in [(op_name, "op name")] + [
        (s.name, "arg name") for s in arg_specs
    ]:
        if C_IDENTIFIER.fullmatch(name) is None:
            raise _fail(op_name, f"invalid {what} C identifier {name!r}")
    for spec in arg_specs:
        if isinstance(spec, ScalarArg) and spec.dtype not in _SCALAR_DTYPES:
            raise _fail(op_name, f"unsupported scalar dtype {spec.dtype}")
        if isinstance(spec, ConstexprArg) and spec.python_type not in _CONSTEXPR_TYPES:
            raise _fail(op_name, f"unsupported constexpr type for {spec.name}")


def _validate_abi(op_name: str, specs: _Specs, abi: tuple[AbiSlot, ...]) -> None:
    runtime = specs.tensors.keys() | specs.scalars.keys()
    in_abi = {slot.arg_name for slot in abi if slot.kind != "stream"}
    if runtime != in_abi:
        raise _fail(
            op_name,
            f"arg_specs runtime args {sorted(runtime)} do not match the "
            f"launcher ABI args {sorted(in_abi)}",
        )
    for slot in abi:
        name = slot.arg_name
        if slot.kind in ("tensor_data", "tensor_layout") and name not in specs.tensors:
            raise _fail(op_name, f"ABI tensor {name!r} has no FlyTensorArg")
        if slot.kind != "scalar":
            continue
        spec = specs.scalars.get(name)
        if spec is None:
            raise _fail(op_name, f"ABI scalar {name!r} has no ScalarArg")
        if _SCALAR_DTYPES[spec.dtype].abi_ctype != slot.ctype:
            raise _fail(
                op_name,
                f"scalar {name} declared {spec.dtype} but the launcher takes "
                f"{slot.ctype}",
            )


@dataclass(frozen=True)
class FlyVariant:
    variant: Variant
    exported: ExportedLauncher


def _variant_fn(fly: FlyVariant, specs: _Specs, cpp_params: str) -> str:
    exported = fly.exported
    body = [
        "static std::once_flag loaded;",
        f"std::call_once(loaded, flydsl_load_module, {exported.module_init_symbol}, "
        f"{exported.module_load_symbol});",
        "const int32_t device_index = torch::stable::accelerator::getCurrentDeviceIndex();",
    ]
    pointers: list[str] = []
    for index, slot in enumerate(exported.abi):
        lines, pointer = _slot_lines(index, slot, specs)
        body.extend(lines)
        pointers.append(pointer)
    body += [
        f"void* args[] = {{{', '.join(pointers)}}};",
        f"{exported.entry_symbol}(args);",
    ]
    return _RUN_FN_TMPL.format(
        symbol=fly.variant.symbol, params=cpp_params, body=_block(body)
    )


def _block(lines: list[str]) -> str:
    # The template indents the first line; this indents the rest.
    return "\n  ".join(lines)


def _externs(exported: ExportedLauncher) -> str:
    return _EXTERNS_TMPL.format(
        entry=exported.entry_symbol,
        module_init=exported.module_init_symbol,
        module_load=exported.module_load_symbol,
    )


def generate_torch_op_content(
    op_name: str,
    arg_specs: list[FlyArgSpec],
    variants: list[FlyVariant],
    mutated: frozenset[str],
) -> str:
    _validate_specs(op_name, arg_specs)
    specs = _Specs(
        tensors={s.name: s for s in arg_specs if isinstance(s, FlyTensorArg)},
        scalars={s.name: s for s in arg_specs if isinstance(s, ScalarArg)},
    )
    for fly in variants:
        _validate_abi(op_name, specs, fly.exported.abi)
    cpp_params = ", ".join(_cpp_param(s) for s in arg_specs)
    op_params = ", ".join(_op_param(s) for s in arg_specs)
    schema_params = ", ".join(
        _schema_param(i, s, mutated) for i, s in enumerate(arg_specs)
    )
    call_args = ", ".join(spec.name for spec in arg_specs)

    selector = render_selector(
        [fly.variant for fly in variants],
        {
            fly.variant.symbol: f"{fly.variant.symbol}_run({call_args})"
            for fly in variants
        },
        fail="flydsl_fail",
    )
    run_fns = [_variant_fn(fly, specs, cpp_params) for fly in variants]
    op_fn = _OP_FN_TMPL.format(
        params=op_params, body=_block([*_unwrap_tensors(arg_specs), *selector])
    )
    return FLYDSL_TEMPLATES.render(
        "flydsl_torch_op.cpp",
        {
            "OP_NAME": _OP_NAME_TMPL.format(
                namespace=DEFAULT_OP_NAMESPACE_PREFIX,
                op_name=op_name,
                schema_params=schema_params,
            ),
            "EXTERNS": "".join(_externs(fly.exported) for fly in variants),
            "OP_FN": "\n".join([*run_fns, op_fn]),
        },
    )
