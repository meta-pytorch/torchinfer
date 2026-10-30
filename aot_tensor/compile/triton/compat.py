# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
This module provides shared utilities that handle differences between
Triton versions.
"""

import math
from typing import Any

# Importing triton requires torch at runtime.
# @dep=fbsource//third-party/pypi/torch:torch
# @manual=//triton:triton
import triton
from packaging.version import Version
from triton.runtime.jit import JITFunction

TRITON_VERSION: str = triton.__version__


def version_gte(version: str, target: str) -> bool:
    """
    Check if version >= target using semantic version comparison.
    Simple string comparison fails for versions like "3.10" vs "3.5"
    """
    return Version(version) >= Version(target)


# Triton 3.5 ``HIPOptions`` defaults, for builds that do not ship the AMD
# backend (triton-mtia, triton-cpu, conda).
_FALLBACK_HIP_OPTION_DEFAULTS: dict[str, int] = {
    "matrix_instr_nonkdim": 0,
    "waves_per_eu": 1,
    "kpack": 1,
}


def hip_option_default(name: str) -> int:
    """Upstream ``HIPOptions`` default for ``name``.

    Channel-dependent (e.g. ``waves_per_eu``) and part of every HIP kernel
    name, so it is read from the AMD backend rather than hard-coded. Falls
    back to the Triton 3.5 values when the AMD backend is not importable.
    """
    try:
        from triton.backends.amd.compiler import HIPOptions
    except ImportError:
        return _FALLBACK_HIP_OPTION_DEFAULTS[name]
    default = HIPOptions.__dataclass_fields__[name].default
    assert isinstance(default, int), f"HIPOptions.{name} default: {default!r}"
    return default


def get_kernel_name(jit_fn: JITFunction[Any]) -> str:
    """
    Get the simple kernel name from a JITFunction.

    In Triton 3.5+, JITFunction._fn_name returns the full qualified name
    (e.g., "aot_tensor.ops.triton_addmm._addmm_fwd").
    In older versions, it returns just the simple name (e.g., "_addmm_fwd").

    This function normalizes the behavior to always return the simple name.

    Args:
        jit_fn: A Triton JITFunction

    Returns:
        The simple kernel name (e.g., "_addmm_fwd")
    """
    fn_name = jit_fn._fn_name
    if version_gte(TRITON_VERSION, "3.5"):
        # Triton 3.5+ uses get_full_name(fn) which returns qualified name
        return fn_name.rsplit(".", 1)[-1]
    else:
        # Older versions use fn.__name__ which is already simple
        return fn_name


def _get_cluster_dims(kernel: Any) -> tuple[int, int, int]:
    """Single source of truth for Triton's Hopper+ ``cluster_dims``
    contract. Missing attr (older Triton / AMD fork) → (1,1,1).
    Used by both codegen (cluster launch path) and ``_get_num_ctas``
    (scratch sizing).
    """
    return getattr(kernel.metadata, "cluster_dims", None) or (1, 1, 1)


def _get_num_ctas(kernel: Any) -> int:
    """Product of ``cluster_dims`` — number of CTAs per cluster on Hopper+."""
    return math.prod(_get_cluster_dims(kernel))


def get_scratch_parameters(
    kernel: Any,
    device_index_expr: str = "torch::stable::accelerator::getCurrentDeviceIndex()",
) -> tuple[str, list[str]]:
    """Emit C++ launcher declarations + arg pointers for the two scratch
    slots Triton 3.5+ kernels expect in their ABI.

    `global_scratch` holds on-device TMA descriptors (or any other state
    Triton materialises in global memory). For non-zero size we
    allocate via the stable C shim `aoti_torch_empty_strided` (error-checked
    by `STABLE_TORCH_ERROR_CODE_CHECK`, both from `torch/csrc/stable/macros.h`)
    and wrap in `torch::stable::Tensor` (stable-ABI replacement for AOT
    Inductor's `RAIIAtenTensorHandle`; same RAII delete via
    `aoti_torch_delete_tensor_object`); zero size passes a null pointer.
    The block is backend-agnostic: HIP launchers are hipified, and upstream's
    AMD driver allocates the same `grid * global_scratch_size` bytes.
    Total bytes scale by `num_ctas` (product of cluster dims): every CTA in a
    Hopper cluster runs `tl.make_tensor_descriptor` independently. AMD has no
    cluster dims, so `num_ctas` is 1 there and sizing matches upstream.
    AMD-fork builds may omit `global_scratch_size` from `KernelMetadata`;
    `getattr(..., 0)` keeps them on the null path. `profile_scratch` (proton)
    is always null since AOT-T does not enable proton.

    ``device_index_expr`` is the C++ expression for the allocation's device.
    AOT-T launchers pass the ``device_idx`` local they already read for this
    launch; other callers (TritonCC) keep the default fresh read.
    """
    declarations = []
    arg_pointers = []

    global_scratch_size = getattr(kernel.metadata, "global_scratch_size", 0) or 0
    # TODO(T293217907): cover the HIP scratch path with an MI350 test.
    if global_scratch_size > 0:
        num_ctas = _get_num_ctas(kernel)
        declarations.extend(
            [
                f"constexpr int64_t global_scratch_per_program = {global_scratch_size};",
                f"constexpr int64_t global_scratch_num_ctas = {num_ctas};",
                "int64_t _global_scratch_size[] = {global_scratch_per_program"
                " * global_scratch_num_ctas * grid.x * grid.y * grid.z};",
                "int64_t _global_scratch_stride[] = {1};",
                "AtenTensorHandle _global_scratch_handle;",
                "STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided("
                "1, _global_scratch_size, _global_scratch_stride, "
                "aoti_torch_dtype_uint8(), aoti_torch_device_type_cuda(), "
                f"{device_index_expr}, "
                "&_global_scratch_handle));",
                "torch::stable::Tensor _global_scratch_tensor(_global_scratch_handle);",
                "CUdeviceptr global_scratch = "
                "reinterpret_cast<CUdeviceptr>(_global_scratch_tensor.data_ptr());",
            ]
        )
    else:
        declarations.append("CUdeviceptr global_scratch = 0;")
    arg_pointers.append("&global_scratch")

    declarations.append("CUdeviceptr profile_scratch = 0;")
    arg_pointers.append("&profile_scratch")

    return ("\n            ".join(declarations), arg_pointers)
