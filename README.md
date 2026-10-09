# torchinfer

torchinfer contains AOT-Tensor, an extensible pipeline for compiling explicitly
marked kernels from multiple DSLs ahead of time, lowering their call sites in a
`torch.fx.GraphModule`, and exporting the resulting model and native artifacts.
The public lowering API currently supports Triton kernels only.

## Requirements

- Linux with an NVIDIA GPU and compatible host driver
- CUDA Toolkit 13.0, including development headers
- Clang 20 or newer with `--embed-dir` support
- Python 3.12
- PyTorch 2.10 through 2.14 built for CUDA 13.0, with the Triton version it
  pins
- [nlohmann/json](https://github.com/nlohmann/json) headers (C++ loader only)

Lowering, export, and loading are tested with PyTorch 2.10, 2.12, and 2.14.
The generated libraries target the PyTorch 2.12 stable ABI, so load an exported
model with PyTorch 2.12 or newer, or with the version that lowered it.

## Installation

From a clone of this repository:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu130 .
```

This installs the `aot_tensor` package and its dependencies, including PyTorch
2.14 built for CUDA 13.0 unless a supported PyTorch is already installed.
For development, install in editable mode with
`pip install -e . --config-settings editable_mode=compat`; the default editable
mode can't load the packaged C++ templates.

Lowering compiles C++ with Clang, and the generated libraries resolve PyTorch
symbols at load time, so set:

```bash
export CXX=/usr/bin/clang++-20
export TORCH_USE_RTLD_GLOBAL=YES
```

Lowering uses the CUDA Toolkit of the `nvcc` on your `PATH`, or else
`/usr/local/cuda`. If that is not CUDA 13.0, also set `CUDA_HOME`, for example
`export CUDA_HOME=/usr/local/cuda-13.0`.

## Quick start

[`examples/vector_add.py`](examples/vector_add.py) marks a small Triton kernel
with `triton_aot`, lowers a module that calls it, exports the result, and loads
it back. From the repository root:

```bash
python examples/vector_add.py
```

After the compiler and lowering logs, it prints the following. The GPU target
and the temporary directory depend on your machine.

```text
Lowered add_kernel for cuda:sm80 in /tmp/aot_tensor_vector_add_abcd1234
Exported /tmp/aot_tensor_vector_add_abcd1234/model.pt
Loaded model matches eager PyTorch: True
```

## Recommendation model example

[`examples/recsys_ranker.py`](examples/recsys_ranker.py) lowers a small
DLRM-style ranking model from [`examples/recsys`](examples/recsys). It combines
a PyTorch embedding table and output layer with two Triton kernels: one sums
each user's variable-length item history, and an autotuned one runs the linear
and ReLU of two MLP layers. From the repository root:

```bash
python examples/recsys_ranker.py
```

After the compiler and lowering logs, it prints:

```text
Lowered 2 Triton kernels for cuda:sm80 in /tmp/aot_tensor_recsys_ranker_abcd1234
Exported /tmp/aot_tensor_recsys_ranker_abcd1234/model.pt
Loaded model matches the PyTorch reference: True
Loaded model matches it on a batch of 37: True
```

Write your own kernels the way the example does:

- Give each `triton_aot` kernel and its launcher their own module. Lowering
  accepts one `triton_aot` kernel per module.
- In the module whose model calls a launcher, register the launcher with
  `torch.fx.wrap` so tracing records it as one node.
- In the launcher, assign the grid to a variable and launch the kernel by the
  name of its `@triton.jit` function. A `lambda meta:` grid may read only
  autotuned meta-parameters.
- Keep launchers TorchScript-compatible: exporting compiles them with
  TorchScript.

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
| `ExtensionBuildConfig` | Overrides the CUDA Toolkit root and adds Torch include and GPU library directories for building the generated extensions. Its `compiler_path` selects only the linker; set `CXX` to choose the C++ compiler. |
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
| `aot_tensor::loadModel(model_path)` | C++ equivalent of `load_model()`. It loads the manifest's shared libraries before returning the exported `torch::jit::Module`. |

`lower_model()` currently accepts exactly one `TritonCompileConfig`. The
public v0 lowering path does not yet support CuTeDSL kernels.

## Lowering options

`lower_model()` takes an optional `LoweringOptions`. By default it compiles for
the attached GPU, registers the generated operators in the `aot_tensor`
namespace, and runs the lowered module on every example input before
returning. To set these explicitly:

```python
from aot_tensor.api.lowering import LoweringOptions
from aot_tensor.compile.triton.adapter import TritonCompileConfig
from triton.backends.compiler import GPUTarget

options = LoweringOptions(
    dsl_configs=(
        TritonCompileConfig(
            gpu_target=GPUTarget(backend="cuda", arch=80, warp_size=32),
            op_namespace="aot_tensor_sm80",
        ),
    ),
    validate=True,
)
```

Pass it as `lower_model(..., options=options)`. `work_dir` must be empty.
`op_namespace` should be unique when multiple independently compiled artifacts
will be loaded into one process.

Keep the PyTorch-provided Triton version rather than upgrading it independently.

### Load an exported model from C++

Keep `model.pt`, its sibling `manifest.json`, and all manifest-relative shared
libraries together. Pass the absolute `model.pt` path to the C++ loader:

```cpp
#include <aot_tensor/api/loading.h>

auto module = aot_tensor::loadModel("/path/to/aott_artifact/model.pt");
auto output = module.forward({input}).toTensor();
```

`loadModel()` validates every library path against the model directory, loads
the libraries in manifest order, and then deserializes the TorchScript model.

To build the loader, compile `aot_tensor/api/loading.cpp` with your program as
C++20, with the checkout root and the nlohmann/json headers on the include path
(Ubuntu's `nlohmann-json3-dev` package installs them on the default path), and
link LibTorch. The LibTorch in the PyTorch wheel works:

```bash
TORCH_DIR="$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')"
clang++-20 -std=c++20 -I. -I"$TORCH_DIR/include" \
  -I"$TORCH_DIR/include/torch/csrc/api/include" \
  main.cpp aot_tensor/api/loading.cpp \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -Wl,--no-as-needed \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Wl,--as-needed -ldl \
  -o main
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details about contributing to
torchinfer.

## License

torchinfer is licensed under the [BSD 3-Clause License](LICENSE).

## Contributors

AOT-Tensor is made possible by the following contributors (listed alphabetically by last name):

Chun-Wei Chen, Rui Jian, Yuhang Liu, Runming Lu, Chang Pan, Katie Tseng, Zhiyong Wang, Chenzhi Yu, Yingji Zhang, Zhuoran Zhao, Tim Zheng.

This work builds on an internal prototype created by Sijia Chen and Bert Maher.
