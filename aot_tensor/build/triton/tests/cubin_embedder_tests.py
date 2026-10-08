# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-strict

import os
import tempfile
import unittest

from aot_tensor.build.triton.cubin_embedder import generate_cpp_for_kernel_binaries
from aot_tensor.compile.triton.utils import hash_kernel_name
from parameterized import parameterized


class KernelBinaryEmbedderTest(unittest.TestCase):
    """Test suite for cubin_embedder module."""

    @parameterized.expand(
        [
            ("single_kernel", ["test_kernel"]),
            ("multiple_kernels", ["kernel_a", "kernel_b", "kernel_c"]),
        ]
    )
    def test_generate_cpp_for_kernel_binaries(
        self, name: str, kernel_variants: list[str]
    ) -> None:
        """Test C++ generation with various kernel configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.cpp")

            for kernel_variant in kernel_variants:
                binary_filename = f"{hash_kernel_name(kernel_variant)}.cubin"
                binary_path = os.path.join(tmpdir, binary_filename)
                with open(binary_path, "wb") as f:
                    f.write(b"\xff")

            generate_cpp_for_kernel_binaries(
                output_path, kernel_variants, binary_dir=tmpdir
            )

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, "r") as f:
                content = f.read()

            # The section/visibility/alignment attributes are what keep the
            # object output identical to a hex array; #embed only replaces the
            # initializer.
            self.assertIn("#include <cstddef>", content)
            self.assertIn('extern "C"', content)
            self.assertIn('section(".triton")', content)
            self.assertIn('visibility("default")', content)
            self.assertIn("aligned(8)", content)

            # Verify extern "C" block wraps array declarations
            extern_start = content.find('extern "C" {')
            extern_end = content.rfind("}")
            self.assertNotEqual(extern_start, -1)
            self.assertGreater(extern_end, extern_start)

            # Byte fidelity is the preprocessor's job now, so the remaining way
            # to get this wrong is pairing a symbol with another kernel's file --
            # which every presence-only assertion would still pass.
            blocks = content.split("unsigned char ")[1:]
            self.assertEqual(len(blocks), len(kernel_variants))
            for kernel_variant in kernel_variants:
                block = next(
                    b for b in blocks if b.startswith(f"{kernel_variant}_cubin[]")
                )
                self.assertIn(
                    f'#embed "{hash_kernel_name(kernel_variant)}.cubin"', block
                )

    def test_output_does_not_embed_the_binary_dir(self) -> None:
        """The compile dir is a per-run mkdtemp; embedding it would churn the
        output on every rebuild and make the checked-in fixtures undiffable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.cpp")
            binary_filename = f"{hash_kernel_name('test_kernel')}.cubin"
            with open(os.path.join(tmpdir, binary_filename), "wb") as f:
                f.write(b"\xff")

            generate_cpp_for_kernel_binaries(
                output_path, ["test_kernel"], binary_dir=tmpdir
            )

            with open(output_path, "r") as f:
                content = f.read()

            self.assertIn(f'#embed "{binary_filename}"', content)
            self.assertNotIn(tmpdir, content)

    def test_generate_cpp_for_kernel_binaries_missing_file(self) -> None:
        """Test that missing kernel binary file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.cpp")

            with self.assertRaises(FileNotFoundError) as context:
                generate_cpp_for_kernel_binaries(
                    output_path, ["nonexistent_kernel"], binary_dir=tmpdir
                )

            self.assertIn("Kernel binary file not found", str(context.exception))
