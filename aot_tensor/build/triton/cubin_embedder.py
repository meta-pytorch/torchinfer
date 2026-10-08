# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Generate a C++ translation unit that embeds kernel binaries (.cubin for NVIDIA,
.hsaco for AMD) with `#embed`, for static linking into the kernel .so.
"""

import logging
import os

# @dep=//aot_tensor/compile/triton/templates:triton_templates
from aot_tensor.compile.template_utils import TRITON_TEMPLATES
from aot_tensor.compile.triton.utils import hash_kernel_name

logger: logging.Logger = logging.getLogger(__name__)

# Template for one embedded kernel binary.
#
# IMPORTANT: The pointer must NOT be in .triton section!
#
# Why this matters for large binaries (>4GB):
# - The cubin data is in .triton, which is placed beyond 4GB via linker script
# - Code in .text accesses the pointer via R_X86_64_PC32 (±2GB range)
# - If pointer is in .triton (>4GB away), R_X86_64_PC32 overflows
# - Solution: pointer in .data.rel.ro (near .text), data in .triton
# - Pointer initialization uses R_X86_64_64 to reference .triton data (no limit)
#
# The pointer is marked volatile to prevent the optimizer from constant-propagating
# through the pointer with -O2. Without volatile, the compiler sees that _cubin_ptr
# is const and initialized with (const void*)_cubin, then constant-propagates:
# image = _cubin_ptr -> image = (const void*)_cubin -> emits R_X86_64_32 to _cubin
KERNEL_BINARY_ARRAY_TEMPLATE = """\
    __attribute__((section(".triton"), visibility("default"), aligned(8)))
    unsigned char {symbol_name}[] = {{
    #embed "{binary_name}"
    }};
    // Pointer to cubin data - placed in .data.rel.ro (near .text) so code can
    // access it via R_X86_64_PC32. The pointer initialization uses R_X86_64_64
    // to reference the cubin data in .triton, which has no distance limit.
    // volatile prevents -O2 from constant-propagating through the pointer.
    __attribute__((section(".data.rel.ro"), visibility("default")))
    const void* volatile {symbol_name}_ptr = (const void*){symbol_name};
"""


def _embed_kernel_binaries(
    kernel_variants: list[str], binary_dir: str, is_amd: bool = False
) -> str:
    """Generate kernel binary array definitions for all kernel variants.

    Args:
        kernel_variants: List of kernel variant names to embed
            (e.g., '_addmm_fwd_sm80_pfp32_pfp32_pfp32_pfp32_i32_').
        binary_dir: Directory containing kernel binary files.
        is_amd: If True, look for .hsaco files instead of .cubin files.

    Returns:
        Combined kernel binary array definitions as a string.
    """
    binary_type = "hsaco" if is_amd else "cubin"
    logger.info(
        f"Embedding {len(kernel_variants)} kernel binaries ({binary_type}) from {binary_dir}"
    )

    kernel_binary_arrays = []
    binary_suffix = "." + binary_type

    for kernel in kernel_variants:
        symbol_name = f"{kernel}_cubin"
        binary_path = os.path.join(
            binary_dir, f"{hash_kernel_name(kernel)}{binary_suffix}"
        )

        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Kernel binary file not found: {binary_path}")

        logger.info(f"  Embedding {os.path.basename(binary_path)} as {symbol_name}")

        kernel_binary_arrays.append(
            KERNEL_BINARY_ARRAY_TEMPLATE.format(
                # Basename, not the full path: the compile dir is a per-run
                # mkdtemp, so embedding it churns the output on every rebuild.
                binary_name=os.path.basename(binary_path),
                symbol_name=symbol_name,
            )
        )
    return "\n".join(kernel_binary_arrays)


def generate_cpp_for_kernel_binaries(
    output_filename: str,
    kernel_variants: list[str],
    binary_dir: str | None = None,
    is_amd: bool = False,
) -> None:
    """Generate a C++ file embedding multiple kernel binaries.

    Args:
        output_filename: Output .cpp file path (e.g., 'embedded_kernels.cpp').
        kernel_variants: List of kernel variant names to embed
            (e.g., '_addmm_fwd_sm80_pfp32_pfp32_pfp32_pfp32_i32_').
        binary_dir: Directory containing kernel binary files. Defaults to script directory.
        is_amd: If True, look for .hsaco files instead of .cubin files.
    """
    if binary_dir is None:
        binary_dir = os.path.dirname(os.path.realpath(__file__))

    kernel_binary_arrays = _embed_kernel_binaries(
        kernel_variants, binary_dir, is_amd=is_amd
    )
    cpp_content = TRITON_TEMPLATES.render(
        "embedded_cubins.cpp", {"CUBIN_ARRAYS": kernel_binary_arrays}
    )

    with open(output_filename, "w") as f:
        f.write(cpp_content)

    logger.info(f"Generated {output_filename}")
