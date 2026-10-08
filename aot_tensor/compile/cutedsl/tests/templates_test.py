# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""CPU checks for the CuTeDSL C++ skeleton templates (cutedsl_entry.cpp /
cutedsl_torch_op.cpp): the static shims plus the GENERATE regions the codegen
fills per op. No GPU or real CuTeDSL compile.
"""

import unittest

from aot_tensor.compile.template_utils import CUTEDSL_TEMPLATES


class CutedslTemplateTest(unittest.TestCase):
    def test_cutedsl_entry_cpp_has_cudart_shims(self) -> None:
        entry = CUTEDSL_TEMPLATES.load("cutedsl_entry.cpp")

        # GENERATE regions the codegen renders.
        for key in ("HEADER_INCLUDE", "OP_NAME", "ENTRY_FN"):
            self.assertIn(f"// __CUTEDSL_AOT_GENERATE_BEGIN__ {key}", entry)
            self.assertIn(f"// __CUTEDSL_AOT_GENERATE_END__ {key}", entry)

        # op_name reaches the static shims through the macro, not interpolation.
        self.assertIn("CUTEDSL_OP_NAME", entry)

        # Static cudart 12.5+ loader + shims live verbatim in the template.
        self.assertIn("cudaLibraryLoadData", entry)
        self.assertIn("cutedsl_cudart_symbol", entry)
        self.assertIn("_cudaLaunchKernelEx", entry)

        # Stays off the libtorch stable-ABI-forbidden namespaces.
        self.assertNotIn("c10::", entry)
        self.assertNotIn("at::", entry)

    def test_cutedsl_torch_op_cpp_has_dladdr_loader(self) -> None:
        torch_op = CUTEDSL_TEMPLATES.load("cutedsl_torch_op.cpp")

        # GENERATE regions the codegen renders.
        for key in ("OP_NAME", "ENTRY_TYPE", "OP_FN"):
            self.assertIn(f"// __CUTEDSL_AOT_GENERATE_BEGIN__ {key}", torch_op)
            self.assertIn(f"// __CUTEDSL_AOT_GENERATE_END__ {key}", torch_op)

        # Per-op name/symbol/sidecar/schema reach the static code through macros.
        self.assertIn("CUTEDSL_OP_NAME", torch_op)
        self.assertIn("CUTEDSL_ENTRY_SYMBOL", torch_op)
        self.assertIn("CUTEDSL_SIDECAR_NAME", torch_op)
        self.assertIn("CUTEDSL_OP_SCHEMA", torch_op)

        # The torch library registration is static template text, driven by macros.
        self.assertIn("STABLE_TORCH_LIBRARY_IMPL(triton_aot, CUDA, m)", torch_op)
        self.assertIn("m.impl(CUTEDSL_OP_NAME, TORCH_BOX(&cutedsl_op))", torch_op)

        # Static relocatable-sidecar loader lives verbatim in the template.
        self.assertIn("dladdr", torch_op)
        self.assertIn("cutedsl_sidecar_path", torch_op)
        self.assertIn("cutedsl_load_entry", torch_op)
        # Sidecar resolved relative to this .so, never an absolute build path.
        self.assertNotIn("/tmp", torch_op)
