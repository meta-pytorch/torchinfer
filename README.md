# torchinfer

TorchInfer contains AOT Tensor, an extensible pipeline for compiling explicitly
marked kernels from multiple DSLs ahead of time, lowering their call sites in a
`torch.fx.GraphModule`, and exporting the resulting model and native artifacts.
The public lowering API currently supports Triton kernels only.

## Requirements

- Linux with an NVIDIA GPU and compatible host driver
- CUDA Toolkit 13.0, including development headers
- Clang 20 or newer with `--embed-dir` support
- Python 3.12
- PyTorch `2.14.0+cu130` with its matched Triton `3.8.0`

This repository currently has no `pyproject.toml`, so expose the checkout and
configure the compiler explicitly:

```bash
export PYTHONPATH="$PWD"
export CUDA_HOME=/usr/local/cuda-13.0
export CC=/usr/bin/clang-20
export CXX=/usr/bin/clang++-20
export TORCH_USE_RTLD_GLOBAL=YES
```

## Public API reference

The APIs and configuration example below describe the currently supported
Triton lowering path only.

### Kernel declaration

| API | Description |
| --- | --- |
| `triton_aot(annotations)` | Marks a `@triton.jit` kernel for AOT compilation. Tensor dtypes can be inferred from example inputs; scalar types and specialization hints can be supplied by argument name. |
| `AnnotationHint(dtype, hint)` | Describes a dtype plus a specialization constraint. Supported hints are `1`, `8`, and `16`; pointer alignment currently supports `16`. |

### Compilation and lowering

| API | Description |
| --- | --- |
| `TritonCompileConfig` | Configures the target GPU, generated operator namespace, optional autotune-cache overrides, and an optional target drift check. With no `gpu_target`, the attached GPU is detected automatically. |
| `ExtensionBuildConfig` | Selects the C++ compiler and CUDA Toolkit, with optional additional Torch include and GPU library directories. |
| `LoweringOptions` | Combines the Triton and extension build configurations and controls post-lowering validation. |
| `lower_model(module, example_inputs, *, work_dir, options)` | Collects kernel specializations from representative inputs, compiles native artifacts, rewrites the graph in place, optionally validates it, and returns a `LoweringResult`. |
| `LoweringResult` | Contains the lowered graph module, artifact directory, artifact metadata, GPU target, Torch stable-ABI target, and operator namespace. |
| `LoweringArtifact` | Identifies one generated file by DSL, artifact kind, and path relative to the lowering work directory. |

### Export and loading

| API | Description |
| --- | --- |
| `ExportOptions` | Configures validation inputs for export. PT2 export is reserved but not currently implemented. |
| `export_model(lowering, *, options)` | Scripts the lowered module and writes `model.pt` plus `manifest.json` into the lowering work directory. |
| `load_model(model_path)` | Loads every shared library listed in the manifest and returns the exported TorchScript module. |

`lower_model()` currently accepts exactly one `TritonCompileConfig`. The
public v0 lowering path does not yet support CuTeDSL kernels.

## Enable AOT Tensor

AOT Tensor is enabled for a lowering operation by passing `LoweringOptions` to
`lower_model()`. The graph module must contain a call to a function wrapping an
explicit `@triton_aot` kernel.

```python
from pathlib import Path

from aot_tensor.api.exporting import export_model, ExportOptions
from aot_tensor.api.loading import load_model
from aot_tensor.api.lowering import lower_model, LoweringOptions
from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.compile.triton.adapter import TritonCompileConfig


# Let Triton detect the attached GPU and compile for that target.
triton_config = TritonCompileConfig(
    op_namespace="aot_tensor",
)

# Point extension compilation at the public CUDA Toolkit and Clang.
build_config = ExtensionBuildConfig(
    compiler_path="/usr/bin/clang++-20",
    gpu_toolkit_path="/usr/local/cuda-13.0",
)

options = LoweringOptions(
    dsl_configs=(triton_config,),
    extension_build_config=build_config,
    validate=True,
)

# graph_module is a torch.fx.GraphModule whose forward path invokes an
# @triton_aot kernel. example_inputs is one tuple of representative CUDA inputs.
example_inputs = (input_a, input_b)
lowering = lower_model(
    graph_module,
    [example_inputs],
    work_dir=Path("aott_artifact"),
    options=options,
)

model_path = export_model(
    lowering,
    options=ExportOptions(validation_inputs=[example_inputs]),
)
loaded_model = load_model(model_path)
```

`work_dir` must be empty. With `validate=True`, AOT Tensor runs the lowered
module on every supplied input set before returning. `op_namespace` identifies
the generated custom operators and should be unique when multiple independently
compiled artifacts will be loaded into one process.

To compile for a specific target instead of detecting the attached GPU, provide
an explicit `GPUTarget`, for example:

```python
from triton.backends.compiler import GPUTarget

triton_config = TritonCompileConfig(
    gpu_target=GPUTarget(backend="cuda", arch=80, warp_size=32),
    op_namespace="aot_tensor_sm80",
)
```

Keep the PyTorch-provided Triton version rather than upgrading it independently.
Set `TORCH_USE_RTLD_GLOBAL=YES` before importing PyTorch so generated stable-ABI
libraries can resolve the required runtime symbols.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details about contributing to
TorchInfer.

## License

TorchInfer is BSD licensed, as found in the [LICENSE](LICENSE) file.
