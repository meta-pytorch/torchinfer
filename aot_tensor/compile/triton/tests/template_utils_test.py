# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-strict

import unittest
from typing import Dict

from aot_tensor.compile.template_utils import render_template, TRITON_TEMPLATES
from parameterized import parameterized


class TemplateUtilsTest(unittest.TestCase):
    """Test suite for template_utils module."""

    @parameterized.expand(
        [
            (
                "embedded_cubins.cpp",
                [
                    "__TRITON_AOT_GENERATE_BEGIN__ CUBIN_ARRAYS",
                    "__TRITON_AOT_GENERATE_END__ CUBIN_ARRAYS",
                    'extern "C"',
                ],
            ),
            (
                "kernel.cpp",
                [
                    "__TRITON_AOT_GENERATE_BEGIN__",
                    "__TRITON_AOT_GENERATE_END__",
                    "HEADER_INCLUDE",
                    "KERNEL_SPECS",
                    "SELECTOR",
                ],
            ),
            (
                "kernel.h",
                [
                    "#pragma once",
                    "#include <torch/csrc/stable/tensor.h>",
                    "namespace triton {",
                    "struct gridDims",
                    "// __TRITON_AOT_GENERATE_BEGIN__ TUNER_META_CPP",
                    "// __TRITON_AOT_GENERATE_BEGIN__ SELECTOR_PROTO",
                ],
            ),
        ]
    )
    def test_load_triton_template(self, name: str, expected: list[str]) -> None:
        template = TRITON_TEMPLATES.load(name)
        for s in expected:
            self.assertIn(s, template)

    @parameterized.expand([("TRITON_AOT",), ("CUTEDSL_AOT",)])
    def test_render_template_replaces_block(self, tag: str) -> None:
        template = (
            "// header\n"
            f"// __{tag}_GENERATE_BEGIN__ FOO\n"
            "placeholder\n"
            f"// __{tag}_GENERATE_END__ FOO\n"
            "// footer"
        )
        result = render_template(template, {"FOO": "replaced content\n"}, tag=tag)
        self.assertIn("replaced content", result)
        self.assertIn("// header", result)
        self.assertIn("// footer", result)
        self.assertNotIn("placeholder", result)
        # markers kept in place
        self.assertIn(f"// __{tag}_GENERATE_BEGIN__ FOO", result)

    def test_render_template_multiple_keys(self) -> None:
        """Test rendering template with multiple keys and backslash escapes."""
        template = """Hello // __TRITON_AOT_GENERATE_BEGIN__ NAME
dummy name
// __TRITON_AOT_GENERATE_END__ NAME! Value: // __TRITON_AOT_GENERATE_BEGIN__ VALUE
dummy value
// __TRITON_AOT_GENERATE_END__ VALUE"""
        result = render_template(
            template,
            {
                "NAME": "World",
                "VALUE": 'printf("result=%d\\n", 42);',
            },
        )
        expected = """Hello // __TRITON_AOT_GENERATE_BEGIN__ NAME
World// __TRITON_AOT_GENERATE_END__ NAME! Value: // __TRITON_AOT_GENERATE_BEGIN__ VALUE
printf("result=%d\\n", 42);// __TRITON_AOT_GENERATE_END__ VALUE"""
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            # BEGIN marker without END marker
            (
                "mismatched_markers",
                "// __TRITON_AOT_GENERATE_BEGIN__ NAME\ndummy",
                {"NAME": "value"},
                "Mismatched BEGIN/END markers",
            ),
            # Missing key in replacements
            (
                "missing_key",
                "// __TRITON_AOT_GENERATE_BEGIN__ NAME\ndummy\n// __TRITON_AOT_GENERATE_END__ NAME",
                {},
                "Keys mismatch",
            ),
            # Extra key in replacements
            (
                "extra_key",
                "// __TRITON_AOT_GENERATE_BEGIN__ NAME\ndummy\n// __TRITON_AOT_GENERATE_END__ NAME",
                {"NAME": "value", "EXTRA": "extra"},
                "Keys mismatch",
            ),
            # Duplicate BEGIN marker
            (
                "duplicate_begin",
                "// __TRITON_AOT_GENERATE_BEGIN__ NAME\n// __TRITON_AOT_GENERATE_BEGIN__ NAME\n// __TRITON_AOT_GENERATE_END__ NAME",
                {"NAME": "value"},
                "Duplicate BEGIN marker",
            ),
            # Duplicate END marker
            (
                "duplicate_end",
                "// __TRITON_AOT_GENERATE_BEGIN__ NAME\n// __TRITON_AOT_GENERATE_END__ NAME\n// __TRITON_AOT_GENERATE_END__ NAME",
                {"NAME": "value"},
                "Duplicate END marker",
            ),
            # BEGIN marker not followed by newline
            (
                "no_newline_after_begin",
                "// __TRITON_AOT_GENERATE_BEGIN__ NAME// __TRITON_AOT_GENERATE_END__ NAME",
                {"NAME": "value"},
                "must be followed by newline",
            ),
        ]
    )
    def test_render_template_error_cases(
        self,
        name: str,
        template: str,
        replacements: Dict[str, str],
        expected_error: str,
    ) -> None:
        """Test that render_template asserts on invalid input."""
        with self.assertRaises(AssertionError) as ctx:
            render_template(template, replacements)
        self.assertIn(expected_error, str(ctx.exception))
