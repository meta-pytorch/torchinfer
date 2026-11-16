# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""Checks across the CuTeDSL and FlyDSL torch-op templates."""

import re
import unittest

from aot_tensor.compile.template_utils import CUTEDSL_TEMPLATES, FLYDSL_TEMPLATES


def nearest_entry_definition(template: str) -> str:
    """``aott_nearest_entry`` from its comment to its closing brace."""
    match = re.search(r"// Autotune routing .*?\n}\n", template, re.DOTALL)
    if match is None:
        raise AssertionError("aott_nearest_entry not found")
    return match.group(0)


class NearestEntryTest(unittest.TestCase):
    def test_both_torch_op_templates_carry_the_same_definition(self) -> None:
        # render_selector calls it for either DSL.
        self.assertEqual(
            nearest_entry_definition(CUTEDSL_TEMPLATES.load("cutedsl_torch_op.cpp")),
            nearest_entry_definition(FLYDSL_TEMPLATES.load("flydsl_torch_op.cpp")),
        )
