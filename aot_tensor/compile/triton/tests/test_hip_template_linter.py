# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from aot_tensor.compile.triton.gen_hip_templates import generate
from aot_tensor.compile.triton.hip_template_linter import check_templates, main
from parameterized import parameterized


class HipTemplateLinterTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("header", ("kernel.h",)),
            ("source", ("kernel.cpp",)),
            ("both", ("kernel.h", "kernel.cpp")),
        ]
    )
    def test_missing_copies_report_regeneration_without_writing(
        self, _case: str, missing: tuple[str, ...]
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            templates = Path(directory)
            (templates / "kernel.h").write_bytes(b"#pragma once\n#include <cuda.h>\n")
            (templates / "kernel.cpp").write_bytes(
                b'#include "kernel.h"\nCUdevice device;\n'
            )
            expected = generate(templates)
            hip = templates / "hip"
            hip.mkdir()
            for name, text in expected.items():
                (hip / name).write_bytes(text.encode("utf-8"))
            for name in missing:
                (hip / name).unlink()
            if len(missing) == 2:
                hip.rmdir()

            messages = check_templates(templates, [str(templates / "kernel.h")])

            self.assertEqual(
                [(str(hip / name), "", expected[name]) for name in missing],
                [
                    (message["path"], message["original"], message["replacement"])
                    for message in messages
                ],
            )
            self.assertTrue(
                all(message["code"] == "HIPTEMPLATES" for message in messages)
            )
            for name, text in expected.items():
                if name in missing:
                    self.assertFalse((hip / name).exists())
                else:
                    self.assertEqual(text.encode("utf-8"), (hip / name).read_bytes())

    @parameterized.expand(
        [
            ("gen_hip_templates.py",),
            ("templates/kernel.h",),
            ("templates/kernel.cpp",),
            ("templates/hip/kernel.h",),
            ("templates/hip/kernel.cpp",),
        ]
    )
    def test_replacements_do_not_write(self, filename: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triton_dir = Path(directory)
            templates = triton_dir / "templates"
            (templates / "hip").mkdir(parents=True)
            header = templates / "hip" / "kernel.h"
            source = templates / "hip" / "kernel.cpp"
            header.write_bytes(b"old\r\n")
            source.write_bytes(b"fresh\n")
            with patch(
                "aot_tensor.compile.triton.hip_template_linter.generate",
                return_value={"kernel.h": "new\n", "kernel.cpp": "fresh\n"},
            ) as generate:
                messages = check_templates(templates, [str(triton_dir / filename)])
            generate.assert_called_once_with(templates.resolve())
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["path"], str(header.resolve()))
            self.assertEqual(messages[0]["original"], "old\r\n")
            self.assertEqual(messages[0]["replacement"], "new\n")
            self.assertEqual(header.read_bytes(), b"old\r\n")
            self.assertEqual(source.read_bytes(), b"fresh\n")

    @parameterized.expand(
        [("torch_op.cpp",), ("hip/other.h",), ("../other/templates/kernel.h",)]
    )
    def test_unrelated_paths_do_not_generate(self, filename: str) -> None:
        with patch(
            "aot_tensor.compile.triton.hip_template_linter.generate"
        ) as generate:
            self.assertEqual(
                check_templates(Path("templates"), [str(Path("templates") / filename)]),
                [],
            )
        generate.assert_not_called()

    @parameterized.expand(
        [(None,), ("error",), ("warning",), ("advice",), ("disabled",)]
    )
    def test_cli_pathsfile_and_fresh_noop(self, severity: str | None) -> None:
        with tempfile.TemporaryDirectory() as directory:
            templates = Path(directory) / "templates"
            (templates / "hip").mkdir(parents=True)
            for name in ("kernel.h", "kernel.cpp"):
                (templates / "hip" / name).write_bytes(b"stale\n")
            pathsfile = Path(directory) / "paths"
            pathsfile.write_text(
                "\n".join(
                    str(path)
                    for path in (
                        templates.parent / "gen_hip_templates.py",
                        templates / "kernel.h",
                        templates / "kernel.cpp",
                        templates / "hip" / "kernel.h",
                        templates / "hip" / "kernel.cpp",
                    )
                )
                + "\n"
            )
            argv = ["--templates-dir", str(templates), f"@{pathsfile}"]
            if severity is not None:
                argv.extend(["--severity", severity])
            with patch(
                "aot_tensor.compile.triton.hip_template_linter.generate",
                return_value={"kernel.h": "fresh\n", "kernel.cpp": "fresh\n"},
            ) as generate:
                output = io.StringIO()
                with redirect_stdout(output):
                    main(argv)
                messages = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(len(messages), 2)
                generate.assert_called_once_with(templates.resolve())
                for message in messages:
                    self.assertEqual((message["line"], message["char"]), (1, 1))
                    self.assertEqual(message["severity"], severity or "warning")
                    self.assertEqual(Path(message["path"]).read_bytes(), b"stale\n")
                    Path(message["path"]).write_bytes(message["replacement"].encode())
                output = io.StringIO()
                with redirect_stdout(output):
                    main(argv)
                self.assertEqual(output.getvalue(), "")
