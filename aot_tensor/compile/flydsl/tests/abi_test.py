# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import re
import unittest

from aot_tensor.compile.flydsl.abi import missing_runtime_functions, RUNTIME_FUNCTIONS
from aot_tensor.compile.template_utils import FLYDSL_TEMPLATES


class RuntimeFunctionsTest(unittest.TestCase):
    def test_template_defines_exactly_the_listed_functions(self) -> None:
        template = FLYDSL_TEMPLATES.load("flydsl_torch_op.cpp")
        defined = set(re.findall(r'extern "C" [^\n(]*\b(mgpu\w+)\(', template))
        self.assertEqual(defined, RUNTIME_FUNCTIONS)

    def test_reports_only_the_unimplemented_mgpu_functions(self) -> None:
        undefined = [
            "_GLOBAL_OFFSET_TABLE_",
            "mgpuLaunchKernel",
            "mgpuLaunchClusterKernel",
            "mgpuMemAlloc",
            "mgpuModuleGetFunction",
        ]
        self.assertEqual(
            missing_runtime_functions(undefined),
            ["mgpuLaunchClusterKernel", "mgpuMemAlloc"],
        )
