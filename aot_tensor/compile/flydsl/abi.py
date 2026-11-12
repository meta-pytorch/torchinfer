# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from typing import Any, Literal

# One packed-argument slot of an exported FlyDSL launcher, in call order. The
# exported entry ``int32_t {symbol}(void** args)`` takes one pointer per slot:
#   tensor_data   -> void* holding the tensor data pointer
#   tensor_layout -> packed little-endian bytes: int{shape_bits} shape[d] for d
#                    in shape_dims, then int{stride_bits} stride[d] for d in
#                    stride_dims
#   scalar        -> the value as ``ctype``
#   stream        -> void* holding the HIP stream
SlotKind = Literal["tensor_data", "tensor_layout", "scalar", "stream"]

C_IDENTIFIER: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The FlyDSL ``export_to_c`` metadata schema this module reads.
METADATA_SCHEMA_VERSION = 2
_SLOT_KINDS: frozenset[str] = frozenset(
    {"tensor_data", "tensor_layout", "scalar", "stream"}
)


@dataclass(frozen=True)
class AbiSlot:
    kind: SlotKind
    # Launcher parameter this slot belongs to.
    arg_name: str
    ctype: str
    size: int
    shape_dims: tuple[int, ...] = ()
    stride_dims: tuple[int, ...] = ()
    shape_bits: int = 32
    stride_bits: int = 64


@dataclass(frozen=True)
class ExportedLauncher:
    """Everything codegen and the builder need from one exported launcher."""

    object_path: str
    # Packed entry, ``int32_t {entry_symbol}(void**)``.
    entry_symbol: str
    # ``int32_t {module_init_symbol}(void)`` once per process, then
    # ``int32_t {module_load_symbol}(int32_t device)`` before each entry call
    # (-1: the current device). Entry and lifecycle calls return 0 on success.
    module_init_symbol: str
    module_load_symbol: str
    abi: tuple[AbiSlot, ...]
    # What the .so linking the object needs, e.g. ``-lamdhip64``.
    linker_flags: tuple[str, ...] = ()


def _abi_slot(slot: dict[str, Any]) -> AbiSlot:
    # args.stream_param_name requires a stream parameter, so every slot is named.
    kind, name = slot["kind"], slot["arg_name"]
    if kind not in _SLOT_KINDS:
        raise NotImplementedError(
            "FlyDSL AOT supports tensor, scalar and stream arguments; got a "
            f"{kind!r} slot for {name!r}"
        )
    if "field" in slot or "encoding" in slot:
        raise NotImplementedError(
            f"FlyDSL AOT does not support struct or 16-bit float argument {name!r}"
        )
    abi_slot = AbiSlot(
        kind=kind,
        arg_name=name,
        ctype=slot["ctype"],
        size=slot["size"],
        shape_dims=tuple(slot.get("shape_dims", ())),
        stride_dims=tuple(slot.get("stride_dims", ())),
        shape_bits=slot.get("shape_bits") or 32,
        stride_bits=slot.get("stride_bits") or 64,
    )
    if kind == "tensor_layout":
        if not {abi_slot.shape_bits, abi_slot.stride_bits} <= {32, 64}:
            raise NotImplementedError(
                f"FlyDSL AOT packs 32- or 64-bit layouts; tensor {name!r} uses "
                f"{abi_slot.shape_bits}-bit shapes and "
                f"{abi_slot.stride_bits}-bit strides"
            )
    return abi_slot


def launcher_from_metadata(
    object_path: str, metadata: dict[str, Any]
) -> ExportedLauncher:
    """The ``ExportedLauncher`` an ``export_to_c`` object's metadata describes."""
    version = metadata.get("schema_version")
    if version != METADATA_SCHEMA_VERSION:
        raise NotImplementedError(
            f"FlyDSL AOT reads export metadata schema {METADATA_SCHEMA_VERSION}; "
            f"{object_path} has {version}"
        )
    return ExportedLauncher(
        object_path=object_path,
        entry_symbol=metadata["symbol"],
        module_init_symbol=metadata["module_init_symbol"],
        module_load_symbol=metadata["module_load_symbol"],
        abi=tuple(_abi_slot(slot) for slot in metadata["abi"]),
        linker_flags=tuple(metadata["linker_flags"]),
    )


# =====================================================================
# the metadata global of an ELF64 little-endian relocatable object
# =====================================================================

_SHT_SYMTAB = 2
_SYM_SIZE = 24
_SHN_UNDEF = 0
_SHN_LORESERVE = 0xFF00


def _section_headers(data: bytes) -> list[tuple[int, int, int, int]]:
    """(type, offset, size, link) of each section."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian ELF64 object")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    headers = []
    for i in range(shnum):
        _name, sh_type, _flags, _addr, offset, size, link = struct.unpack_from(
            "<IIQQQQI", data, shoff + i * shentsize
        )
        headers.append((sh_type, offset, size, link))
    return headers


def read_symbol_bytes(data: bytes, symbol: str) -> bytes:
    """The bytes the relocatable object ``data`` defines for ``symbol``."""
    sections = _section_headers(data)
    wanted = symbol.encode() + b"\0"
    for sh_type, offset, size, link in sections:
        if sh_type != _SHT_SYMTAB:
            continue
        _type, strtab, strtab_size, _link = sections[link]
        for entry in range(offset, offset + size, _SYM_SIZE):
            st_name, _info, _other, shndx, value, st_size = struct.unpack_from(
                "<IBBHQQ", data, entry
            )
            # Undefined and special-index (SHN_ABS, SHN_COMMON) symbols have no
            # bytes in the object.
            if shndx == _SHN_UNDEF or shndx >= _SHN_LORESERVE:
                continue
            if data.startswith(wanted, strtab + st_name, strtab + strtab_size):
                # In a relocatable object the value is the section offset.
                start = sections[shndx][1] + value
                return data[start : start + st_size]
    raise ValueError(f"symbol {symbol!r} not found")


def read_exported_launcher(object_path: str, symbol: str) -> ExportedLauncher:
    """The launcher ``export_to_c`` wrote to ``object_path`` with entry
    ``{symbol}``, from the JSON it embeds as ``const char {symbol}__metadata[]``."""
    with open(object_path, "rb") as fp:
        raw = read_symbol_bytes(fp.read(), f"{symbol}__metadata")
    launcher = launcher_from_metadata(object_path, json.loads(raw.rstrip(b"\0")))
    if launcher.entry_symbol != symbol:
        raise RuntimeError(
            f"FlyDSL AOT: {object_path} exports {launcher.entry_symbol!r}, "
            f"expected {symbol!r}"
        )
    return launcher
