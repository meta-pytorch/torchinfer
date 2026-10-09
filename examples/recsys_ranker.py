# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Lower, export, and load a small recommendation model with two Triton kernels.

Run from the repository root: python examples/recsys_ranker.py

The model is in recsys/. Each of its Triton kernels has its own module because
lowering accepts one triton_aot kernel per module.
"""

import tempfile

import torch
from aot_tensor.api.exporting import export_model, ExportOptions
from aot_tensor.api.loading import load_model
from aot_tensor.api.lowering import lower_model
from recsys.model import make_inputs, RecsysRanker


def main() -> None:
    torch.manual_seed(0)
    model = RecsysRanker().cuda().eval()
    inputs = make_inputs(batch_size=64)

    with torch.no_grad():
        work_dir = tempfile.mkdtemp(prefix="aot_tensor_recsys_ranker_")
        lowering = lower_model(
            torch.fx.symbolic_trace(model), [inputs], work_dir=work_dir
        )
        libraries = [a for a in lowering.artifacts if a.kind == "shared_library"]
        print(
            f"Lowered {len(libraries)} Triton kernels for {lowering.gpu_target} "
            f"in {lowering.work_dir}"
        )

        model_path = export_model(
            lowering, options=ExportOptions(validation_inputs=[inputs])
        )
        print(f"Exported {model_path}")

        loaded = load_model(model_path)
        matches = torch.allclose(loaded(*inputs), model.reference(*inputs), atol=1e-5)
        print(f"Loaded model matches the PyTorch reference: {matches}")

        # Lowering does not specialize on the batch size or history lengths.
        other = make_inputs(batch_size=37)
        matches = torch.allclose(loaded(*other), model.reference(*other), atol=1e-5)
        print(f"Loaded model matches it on a batch of 37: {matches}")


if __name__ == "__main__":
    main()
