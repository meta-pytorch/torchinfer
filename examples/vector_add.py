# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Lower, export, and load a model that calls a small Triton kernel.

Run from the repository root: python examples/vector_add.py
"""

import tempfile

import torch
import triton
import triton.language as tl
from aot_tensor.api.exporting import export_model, ExportOptions
from aot_tensor.api.loading import load_model
from aot_tensor.api.lowering import lower_model
from aot_tensor.types import triton_aot


# Tensor argument types come from the example inputs; scalars need a type.
@triton_aot(annotations={"n_elements": "i32"})
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


# lower_model() rewrites this launcher to call the compiled kernel. Keep it one
# traced node, launch the kernel by its own name, and compute the grid from
# plain values rather than from the launch meta-parameters.
@torch.fx.wrap
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 1024
    grid = ((n_elements + block_size - 1) // block_size,)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out


class AddModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return vector_add(x, y)


def main() -> None:
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")
    inputs = [(x, y)]

    work_dir = tempfile.mkdtemp(prefix="aot_tensor_vector_add_")
    graph_module = torch.fx.symbolic_trace(AddModel())
    lowering = lower_model(graph_module, inputs, work_dir=work_dir)
    print(f"Lowered add_kernel for {lowering.gpu_target} in {lowering.work_dir}")

    model_path = export_model(lowering, options=ExportOptions(validation_inputs=inputs))
    print(f"Exported {model_path}")

    model = load_model(model_path)
    print(f"Loaded model matches eager PyTorch: {torch.allclose(model(x, y), x + y)}")


if __name__ == "__main__":
    main()
