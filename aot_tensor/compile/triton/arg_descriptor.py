# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""ArgDescriptor — per-arg codegen descriptor for AOT-T.

Centralises arg classification (pointer / scalar / constant) so every
``gen_*`` function in ``codegen.py`` iterates descriptors instead of
doing its own dict lookups into ``OpsUnit`` fields.

Also provides type-mapping helpers that convert ``ArgDescriptor``
metadata into context-specific C++ / TorchScript type strings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from aot_tensor.compile.dtypes import CTYPES
from aot_tensor.compile.triton.spec_processing import OpsUnit
from triton.runtime.jit import JITFunction


# ---------------------------------------------------------------------------
# Type-mapping helpers
# ---------------------------------------------------------------------------

CONSTANT_SELECTOR_CTYPE: dict[type[Any], str] = {
    bool: "bool",
    int: "int",
    str: "const std::string&",
}

CONSTANT_CPP_OP_CTYPE: dict[type[Any], str] = {
    bool: "bool",
    int: "int64_t",
    str: "const std::string&",
}

CONSTANT_TORCH_SCHEMA: dict[type[Any], str] = {
    bool: "bool",
    int: "int",
    str: "str",
}


def scalar_cpp_op_ctype(triton_dtype: str) -> str:
    """Triton scalar dtype → widened C++ type for cpp_op / torch_op params."""
    if triton_dtype.startswith("i"):
        return "int64_t"
    if triton_dtype.startswith("f"):
        return "double"
    if triton_dtype == "bool":
        return "bool"
    raise ValueError(f"Unsupported scalar dtype for cpp_op: {triton_dtype}")


def scalar_torch_schema(triton_dtype: str) -> str:
    """Triton scalar dtype → TorchScript schema type string."""
    if triton_dtype.startswith("i"):
        return "int"
    if triton_dtype.startswith("f"):
        return "float"
    if triton_dtype == "bool":
        return "bool"
    raise ValueError(f"Unsupported scalar dtype for torch schema: {triton_dtype}")


# ---------------------------------------------------------------------------
# ArgDescriptor hierarchy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgDescriptor(ABC):
    """Base class for per-arg codegen descriptors.

    Built once by ``build_arg_descriptors`` and consumed by all ``gen_*``
    functions.  Arg kinds:

    - ``PointerArg`` — tensor pointer (required or optional)
    - ``ScalarArg`` — non-pointer signature arg with a Triton dtype
    - ``ConstantArg`` — compile-time constant with a Python type

    Each kind knows how to declare itself in the four ABI contexts AOT-T
    emits (launcher / selector / cpp op / torch schema), so ``codegen.py``
    never switches on arg kind to build a parameter list.  A new arg kind
    that forgets a context is an abstract-method error, not a silently
    dropped parameter.
    """

    name: str
    index: int

    @abstractmethod
    def selector_param(self) -> str:
        """Param declaration in the C++ selector signature."""

    @abstractmethod
    def cpp_op_param(self) -> str:
        """Param declaration in the C++ torch-op wrapper signature."""

    @abstractmethod
    def torch_schema_param(self, default: str) -> str:
        """TorchScript schema fragment.

        *default* is either ``""`` or a rendered ``" = <value>"`` suffix.
        """

    def launcher_param(self, spec_dtype: str) -> str:
        """Param declaration in a per-spec launcher signature.

        *spec_dtype* is that spec's signature type for this position, which
        may be narrower than the descriptor's invariant type.
        """
        return f"{CTYPES[spec_dtype]} {self.name}"


@dataclass(frozen=True)
class PointerArg(ArgDescriptor):
    """Tensor pointer arg (required or optional)."""

    is_optional: bool

    def selector_param(self) -> str:
        return f"const std::optional<torch::stable::Tensor>& {self.name}"

    def cpp_op_param(self) -> str:
        return f"std::optional<torch::stable::Tensor> {self.name}"

    def torch_schema_param(self, default: str) -> str:
        # Every tensor is Tensor? even when required, for TorchScript
        # compatibility; the alias annotation is keyed off arg position.
        # Tensors never carry a schema default.
        alias = chr(ord("a") + self.index)
        return f"Tensor({alias}!)? {self.name}"

    def launcher_param(self, spec_dtype: str) -> str:
        return f"void* {self.name}"


@dataclass(frozen=True)
class ScalarArg(ArgDescriptor):
    """Non-pointer signature arg with a Triton dtype (e.g., ``"i32"``, ``"fp32"``).

    ``triton_dtype`` is the **widest** type across all specs for this
    position, computed by ``_compute_invariants`` via ``_wider_type``.
    Individual specs may use a narrower type (e.g., ``"i32"`` when
    ``triton_dtype`` is ``"i64"``); codegen adds ``fits_i32`` guards
    and ``static_cast`` for narrowing.
    """

    triton_dtype: str

    def selector_param(self) -> str:
        return f"{CTYPES[self.triton_dtype]} {self.name}"

    def cpp_op_param(self) -> str:
        return f"{scalar_cpp_op_ctype(self.triton_dtype)} {self.name}"

    def torch_schema_param(self, default: str) -> str:
        return f"{scalar_torch_schema(self.triton_dtype)} {self.name}{default}"


@dataclass(frozen=True)
class ConstantArg(ArgDescriptor):
    """Compile-time constant arg with a Python type (``int``, ``str``, ``bool``)."""

    python_type: type[Any]

    def selector_param(self) -> str:
        return f"{CONSTANT_SELECTOR_CTYPE[self.python_type]} {self.name}"

    def cpp_op_param(self) -> str:
        return f"{CONSTANT_CPP_OP_CTYPE[self.python_type]} {self.name}"

    def torch_schema_param(self, default: str) -> str:
        return f"{CONSTANT_TORCH_SCHEMA[self.python_type]} {self.name}{default}"


def build_arg_descriptors(
    func: JITFunction[list[Any]],
    unit: OpsUnit,
) -> list[ArgDescriptor]:
    """Build ordered arg descriptors from func arg names + OpsUnit invariants.

    Single source of truth for arg classification.  Called once in
    ``compile_to_cpp`` and passed to all downstream codegen functions.
    """
    result: list[ArgDescriptor] = []
    for i, name in enumerate(func.arg_names):
        if i in unit.pointer_args:
            result.append(
                PointerArg(name=name, index=i, is_optional=i in unit.optional)
            )
        elif i in unit.scalar_dtypes:
            result.append(
                ScalarArg(name=name, index=i, triton_dtype=unit.scalar_dtypes[i])
            )
        elif i in unit.constant_types:
            result.append(
                ConstantArg(name=name, index=i, python_type=unit.constant_types[i])
            )
        else:
            raise ValueError(
                f"Arg {name} (index {i}) not classified as pointer, scalar, "
                f"or constant in OpsUnit"
            )
    return result
