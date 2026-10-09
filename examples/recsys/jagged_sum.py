# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Sum-pool variable-length rows of embeddings with a Triton kernel."""

import torch
import triton
import triton.language as tl
from aot_tensor.types import triton_aot


@triton_aot(annotations={"D": "i32"})
@triton.jit
def _jagged_sum_kernel(values_ptr, offsets_ptr, out_ptr, D, BLOCK_D: tl.constexpr):
    row = tl.program_id(axis=0)
    start = tl.load(offsets_ptr + row)
    end = tl.load(offsets_ptr + row + 1)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in range(start, end):
        acc += tl.load(values_ptr + i * D + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * D + cols, acc, mask=mask)


def jagged_sum(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Row ``b`` of the result is the sum of ``values[offsets[b]:offsets[b + 1]]``.

    ``values`` is a contiguous ``[total, D]`` tensor and ``offsets`` holds
    ``num_rows + 1`` increasing positions into it, starting at 0.
    """
    num_rows = offsets.numel() - 1
    D = values.shape[1]
    out = torch.empty((num_rows, D), device=values.device, dtype=values.dtype)
    # Exporting compiles this function with TorchScript, which cannot compile
    # triton.next_power_of_2.
    block_d = 1
    while block_d < D:
        block_d *= 2
    grid = (num_rows,)
    _jagged_sum_kernel[grid](values, offsets, out, D, BLOCK_D=block_d)
    return out
