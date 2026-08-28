# Copyright (c) Meta Platforms, Inc. and affiliates.


"""Stable-ABI type mappings for AOTT codegen.

  * Triton pointer dtype -> ``torch::headeronly::ScalarType`` enum
  * Python type         -> C++ type string

Sibling of ``dtypes.ATYPES`` / ``dtypes.PY_TYPES_TO_CPP_TYPES`` but with zero
link dependency on ATen, so generated cpp can sit on the torch stable ABI.
"""

from typing import Any

# key = Triton pointer-mangled dtype ("*fp32", from triton ``mangle_type``).
# value = ``torch::headeronly::ScalarType`` (not ``c10::``) per libtorch_stable_abi.md.
# See SCALAR_TYPES test.
SCALAR_TYPES: dict[str, str] = {
    "*i1": "torch::headeronly::ScalarType::Bool",
    "*u8": "torch::headeronly::ScalarType::Byte",
    "*i8": "torch::headeronly::ScalarType::Char",
    "*i16": "torch::headeronly::ScalarType::Short",
    "*i32": "torch::headeronly::ScalarType::Int",
    "*i64": "torch::headeronly::ScalarType::Long",
    "*fp16": "torch::headeronly::ScalarType::Half",
    "*fp32": "torch::headeronly::ScalarType::Float",
    "*fp64": "torch::headeronly::ScalarType::Double",
    "*bf16": "torch::headeronly::ScalarType::BFloat16",
    "*fp8e4nv": "torch::headeronly::ScalarType::Float8_e4m3fn",
    "*fp8e4b8": "torch::headeronly::ScalarType::Float8_e4m3fnuz",
}

# Same values as SCALAR_TYPES, but key = ``str(torch.Tensor.dtype)``
# ("torch.float32"). The CuteDSL path reads dtype straight off raw torch tensors
# (no Triton mangling layer), so it keys on torch's native dtype string.
TORCH_DTYPE_TO_STABLE: dict[str, str] = {
    "torch.bool": "torch::headeronly::ScalarType::Bool",
    "torch.uint8": "torch::headeronly::ScalarType::Byte",
    "torch.int8": "torch::headeronly::ScalarType::Char",
    "torch.int16": "torch::headeronly::ScalarType::Short",
    "torch.int32": "torch::headeronly::ScalarType::Int",
    "torch.int64": "torch::headeronly::ScalarType::Long",
    "torch.float16": "torch::headeronly::ScalarType::Half",
    "torch.float32": "torch::headeronly::ScalarType::Float",
    "torch.float64": "torch::headeronly::ScalarType::Double",
    "torch.bfloat16": "torch::headeronly::ScalarType::BFloat16",
}

# Stable ABI override: str → "std::string" instead of "at::string".
PY_TYPES_TO_CPP_TYPES: dict[type[Any], str] = {
    int: "int64_t",
    str: "std::string",
    float: "double",
    bool: "bool",
}
