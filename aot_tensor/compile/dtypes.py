# Copyright (c) Meta Platforms, Inc. and affiliates.


"""Fundamental dtype mappings used across the AOT-T compiler."""

from typing import Any

# Mapping from Triton dtype names to C type names
CTYPES: dict[str, str] = {
    "i1": "bool",
    "u8": "uint8_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "fp16": "half",
    "fp32": "float",
    "fp64": "double",
    "bf16": "__nv_bfloat16",
    "fp8e4nv": "__nv_fp8_e4m3",
    "fp8e4b8": "__hip_fp8_e4m3_fnuz",
}

# Mapping from Triton pointer dtype names to ATen scalar types
ATYPES: dict[str, str] = {
    # Triton mangles ``torch.bool`` to ``*u1``, not ``*i1``. ``*i1`` is
    # unreachable (no dtype mangles to it) and is kept only so an out-of-tree
    # caller keying on it does not break; ``*u1`` is the one that fires.
    # Kept in sync with ``stable_types.SCALAR_TYPES`` -- see
    # ``aot_tensor/compile/tests/stable_types_test.py::test_keys_match_atypes``.
    "*u1": "at::kBool",
    "*i1": "at::kBool",
    "*u8": "at::kByte",
    "*i8": "at::kChar",
    "*i16": "at::kShort",
    "*i32": "at::kInt",
    "*i64": "at::kLong",
    "*fp16": "at::kHalf",
    "*fp32": "at::kFloat",
    "*fp64": "at::kDouble",
    "*bf16": "at::kBFloat16",
    "*fp8e4nv": "at::kFloat8_e4m3fn",
    "*fp8e4b8": "at::kFloat8_e4m3fnuz",
}

# Mapping from Python types to C++ type names
PY_TYPES_TO_CPP_TYPES: dict[type[Any], str] = {
    int: "int64_t",
    str: "at::string",
    float: "double",
}
