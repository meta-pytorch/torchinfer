# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import unittest
from pathlib import Path

from aot_tensor.compile.triton.gen_hip_templates import generate


class TemplateHipSyncTest(unittest.TestCase):
    def test_generated_hip_templates_are_current(self) -> None:
        root = Path(__file__).with_name("sources")
        if not root.is_dir():
            root = Path(__file__).parents[1] / "templates"
        for name, expected in generate(root).items():
            with self.subTest(template=name):
                self.assertEqual(
                    expected.encode("utf-8"),
                    (root / "hip" / name).read_bytes(),
                    "Edit the CUDA source and run gen_hip_templates --write.",
                )
