# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

# One packed-argument slot of an exported FlyDSL launcher, in call order. The
# exported entry ``_mlir_{op}(void** args)`` takes one pointer per slot:
#   tensor_data   -> void* holding the tensor data pointer
#   tensor_layout -> packed little-endian bytes: int32 shape[d] for d in
#                    shape_dims, then int32/int64 stride[d] for d in stride_dims
#   scalar        -> the value as ``ctype``
#   stream        -> void* holding the HIP stream
SlotKind = Literal["tensor_data", "tensor_layout", "scalar", "stream"]

C_IDENTIFIER: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class AbiSlot:
    kind: SlotKind
    # Launcher parameter this slot belongs to.
    arg_name: str
    ctype: str
    size: int
    shape_dims: tuple[int, ...] = ()
    stride_dims: tuple[int, ...] = ()
    stride_bits: int = 64


@dataclass(frozen=True)
class ExportedLauncher:
    """Everything codegen and the builder need from one exported launcher."""

    object_path: str
    # Packed entry, ``void {entry_symbol}(void**)``.
    entry_symbol: str
    # GPU module loaders, ``void {sym}(void** module, int32_t* err)``, exported
    # by the explicit-module handler (``aot._use_explicit_module_loader``); call
    # init then load once before the first entry call.
    module_init_symbol: str
    module_load_symbol: str
    abi: tuple[AbiSlot, ...]


# The MLIR GPU runtime functions (``mgpu*``) templates/flydsl_torch_op.cpp
# defines. A launcher object calling any other one would leave it undefined in
# the .so, which only fails when the serving process loads it.
RUNTIME_FUNCTIONS: frozenset[str] = frozenset(
    {"mgpuModuleLoad", "mgpuModuleGetFunction", "mgpuLaunchKernel"}
)


def missing_runtime_functions(undefined_symbols: Iterable[str]) -> list[str]:
    """The ``mgpu*`` functions a launcher object needs that the template lacks."""
    needed = {name for name in undefined_symbols if name.startswith("mgpu")}
    return sorted(needed - RUNTIME_FUNCTIONS)
