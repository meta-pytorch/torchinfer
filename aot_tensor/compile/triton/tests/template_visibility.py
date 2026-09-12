# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-strict

"""Shared visibility assertions for the CUDA and AMD template-load tests.

Both tests read the SAME template sources through different buck
configurations (the AMD target forces `ovr_config//gpu:amd`, so it sees the
hipified copy), so the assertions are identical and have to stay in lockstep.
"""

import unittest

# @dep=//aot_tensor/compile/triton/templates:triton_templates
from aot_tensor.compile.template_utils import TemplateSet

# Every generated region named after the kernel alone: one .so per forward
# method defines the same `triton::aot::<kernel>` and `<kernel>_meta`, and the
# predictor dlopens them all into one process.
_HIDDEN_REGIONS: tuple[tuple[str, str], ...] = (
    ("kernel.h", "TUNER_META_CPP"),
    ("kernel.h", "SELECTOR_PROTO"),
    ("kernel.cpp", "SELECTOR"),
)

_PUSH: str = "#pragma GCC visibility push(hidden)"
_POP: str = "#pragma GCC visibility pop"
_ANCHOR: str = "__triton_aot_anchor_get_stream"


def assert_generated_regions_are_hidden(
    test: unittest.TestCase, templates: TemplateSet
) -> None:
    """Every kernel-named generated region sits inside a hidden region, and
    the stream anchor stays outside it."""
    for name, marker in _HIDDEN_REGIONS:
        with test.subTest(template=name, marker=marker):
            content = templates.load(name)
            begin = content.find(f"// __TRITON_AOT_GENERATE_BEGIN__ {marker}\n")
            test.assertNotEqual(-1, begin, f"{name}: no {marker} marker")
            # The push that encloses THIS marker, not the file's first one.
            push = content.rfind(_PUSH, 0, begin)
            test.assertNotEqual(-1, push, f"{name}: {marker} precedes every push")
            pop = content.find(_POP, push)
            test.assertNotEqual(-1, pop, f"{name}: no pop after the enclosing push")
            test.assertLess(begin, pop, f"{name}: {marker} follows the pop")

    # The stream anchor is deliberately exported: it is what keeps
    # `triton_aot_get_current_stream` alive when KERNEL_SPECS is empty at
    # buck-build time. Widening the pragma over it would hide it again.
    kernel_cpp = templates.load("kernel.cpp")
    anchor = kernel_cpp.find(_ANCHOR)
    test.assertNotEqual(-1, anchor, "stream anchor missing")
    last_pop = kernel_cpp.rfind(_POP, 0, anchor)
    test.assertNotEqual(-1, last_pop, "stream anchor is inside a hidden region")
    test.assertEqual(
        -1,
        kernel_cpp.find(_PUSH, last_pop, anchor),
        "a hidden region reopens before the stream anchor",
    )
    # Tied to the anchor's own declaration, not merely present in the file.
    decl = kernel_cpp.rfind('extern "C"', 0, anchor)
    test.assertNotEqual(-1, decl, 'stream anchor has no extern "C" declaration')
    test.assertIn(
        'visibility("default")',
        kernel_cpp[decl:anchor],
        "the stream anchor must carry an explicit default visibility",
    )
