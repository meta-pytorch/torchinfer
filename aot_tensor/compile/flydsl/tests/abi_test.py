# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import glob
import os
import unittest
from typing import Any

from aot_tensor.compile.flydsl.abi import (
    AbiSlot,
    launcher_from_metadata,
    METADATA_SCHEMA_VERSION,
    read_exported_launcher,
    read_symbol_bytes,
)
from aot_tensor.constants import FLYDSL_SYMBOL_PREFIX
from libfb.py import parutil
from parameterized import parameterized

# export_to_c objects of the toy silu-gate op, one per SUBTRACT variant.
_FIXTURE_DIR = (
    "aot_tensor/compile/flydsl/tests/fixtures/resources/"
    "flydsl_toy_flydsl__flydsl__silu_gate_launch"
)


def _metadata(**overrides: Any) -> dict[str, Any]:
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "symbol": "aott_op__aaaa",
        "module_init_symbol": "aott_op__aaaa__module_init",
        "module_load_symbol": "aott_op__aaaa__module_load",
        "linker_flags": ["-lamdhip64", "-pthread"],
        "abi": [
            {"kind": "tensor_data", "arg_name": "x", "ctype": "pointer", "size": 8},
            {
                "kind": "tensor_layout",
                "arg_name": "x",
                "ctype": "bytes",
                "size": 16,
                "shape_dims": [0, 1],
                "stride_dims": [0],
                "shape_bits": 32,
                "stride_bits": 64,
            },
            {"kind": "scalar", "arg_name": "n", "ctype": "int32", "size": 4},
            {"kind": "stream", "arg_name": "stream", "ctype": "pointer", "size": 8},
        ],
    }
    return {**metadata, **overrides}


class LauncherFromMetadataTest(unittest.TestCase):
    def test_reads_symbols_slots_and_link_flags(self) -> None:
        launcher = launcher_from_metadata("/out/op.o", _metadata())
        self.assertEqual(launcher.entry_symbol, "aott_op__aaaa")
        self.assertEqual(launcher.module_init_symbol, "aott_op__aaaa__module_init")
        self.assertEqual(launcher.module_load_symbol, "aott_op__aaaa__module_load")
        self.assertEqual(launcher.linker_flags, ("-lamdhip64", "-pthread"))
        self.assertEqual(
            launcher.abi,
            (
                AbiSlot("tensor_data", "x", "pointer", 8),
                AbiSlot("tensor_layout", "x", "bytes", 16, (0, 1), (0,), 32, 64),
                AbiSlot("scalar", "n", "int32", 4),
                AbiSlot("stream", "stream", "pointer", 8),
            ),
        )

    @parameterized.expand(
        [
            ("schema", {"schema_version": 1}, "metadata schema"),
            (
                "pointer_arg",
                {"abi": [{"kind": "pointer", "arg_name": "p", "ctype": "pointer"}]},
                "supports tensor, scalar and stream",
            ),
            (
                "half_scalar",
                {
                    "abi": [
                        {
                            "kind": "scalar",
                            "arg_name": "h",
                            "ctype": "uint16",
                            "size": 2,
                            "encoding": "float16_bits",
                        }
                    ]
                },
                "16-bit float",
            ),
            (
                "struct_field",
                {
                    "abi": [
                        {
                            "kind": "scalar",
                            "arg_name": "s",
                            "ctype": "int32",
                            "size": 4,
                            "field": "x",
                        }
                    ]
                },
                "struct",
            ),
            (
                "layout_width",
                {
                    "abi": [
                        {
                            "kind": "tensor_layout",
                            "arg_name": "x",
                            "ctype": "bytes",
                            "size": 2,
                            "shape_dims": [0],
                            "shape_bits": 16,
                        }
                    ]
                },
                "32- or 64-bit layouts",
            ),
        ]
    )
    def test_rejects_what_codegen_cannot_pack(
        self, _name: str, overrides: dict[str, Any], error: str
    ) -> None:
        with self.assertRaisesRegex(NotImplementedError, error):
            launcher_from_metadata("/out/op.o", _metadata(**overrides))


class ReadExportedLauncherTest(unittest.TestCase):
    def test_reads_the_launchers_of_export_to_c_objects(self) -> None:
        objects = sorted(
            glob.glob(os.path.join(parutil.get_dir_path(_FIXTURE_DIR), "*.o"))
        )
        self.assertEqual(len(objects), 2)
        for path in objects:
            symbol = f"{FLYDSL_SYMBOL_PREFIX}{os.path.basename(path)[: -len('.o')]}"
            launcher = read_exported_launcher(path, symbol)
            self.assertEqual(
                [(s.kind, s.arg_name) for s in launcher.abi],
                [
                    *(
                        (kind, name)
                        for name in ("A", "B", "C", "Out")
                        for kind in ("tensor_data", "tensor_layout")
                    ),
                    ("scalar", "n"),
                    ("stream", "stream"),
                ],
            )

    @parameterized.expand(
        [
            ("missing_symbol", True, "no_such_symbol", "not found"),
            # Referenced by the embedded AOT runtime but defined in libamdhip64.
            ("undefined_symbol", True, "hipGetDevice", "not found"),
            ("not_elf", False, "x", "ELF64"),
        ]
    )
    def test_rejects(
        self, _name: str, from_fixture: bool, symbol: str, error: str
    ) -> None:
        data = b"#!/bin/sh\n"
        if from_fixture:
            (path, *_) = glob.glob(
                os.path.join(parutil.get_dir_path(_FIXTURE_DIR), "*.o")
            )
            with open(path, "rb") as fp:
                data = fp.read()
        with self.assertRaisesRegex(ValueError, error):
            read_symbol_bytes(data, symbol)
