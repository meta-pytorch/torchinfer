# Copyright (c) Meta Platforms, Inc. and affiliates.

"""A fused linear layer and ReLU as an autotuned Triton kernel."""

import torch
import triton
import triton.language as tl
from aot_tensor.types import triton_aot


@triton_aot(
    annotations={
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_xm": "i32",
        "stride_wn": "i32",
        "stride_ym": "i32",
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def _linear_relu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + ks[None, :],
            mask=(rows[:, None] < M) & (ks[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + cols[None, :] * stride_wn + ks[:, None],
            mask=(cols[None, :] < N) & (ks[:, None] < K),
            other=0.0,
        )
        # Full fp32 precision, so results match the PyTorch reference closely.
        acc += tl.dot(x, w, input_precision="ieee")
    acc += tl.load(b_ptr + cols, mask=cols < N, other=0.0)[None, :]
    acc = tl.maximum(acc, 0.0)
    tl.store(
        y_ptr + rows[:, None] * stride_ym + cols[None, :],
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def linear_relu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """``relu(x @ weight.T + bias)``, with ``weight`` in ``torch.nn.Linear`` layout.

    ``x`` and ``weight`` must have contiguous rows.
    """
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # The grid may read only the autotuned meta-parameters: lowering replaces
    # them with the configuration it selected for this layer's N and K.
    grid = lambda meta: (  # noqa: E731
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )
    _linear_relu_kernel[grid](
        x, weight, bias, y, M, N, K, x.stride(0), weight.stride(0), y.stride(0)
    )
    return y
