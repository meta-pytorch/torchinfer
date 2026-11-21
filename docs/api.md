# API reference

The public API covers the Triton lowering path. CuTeDSL kernels are not yet
supported.

## Kernel declaration

| API | Description |
| --- | --- |
| `triton_aot(annotations)` | Marks a `@triton.jit` kernel for AOT compilation. Tensor dtypes can be inferred from example inputs; scalar types and specialization hints can be supplied by argument name. |
| `AnnotationHint(dtype, hint)` | Describes a dtype plus a specialization constraint. Supported hints are `1`, `8`, and `16`; pointer alignment currently supports `16`. |

## Compilation and lowering

| API | Description |
| --- | --- |
| `TritonCompileConfig` | Configures the target GPU, generated operator namespace, optional autotune-cache overrides, an optional target drift check, and an optional provider of extra values the generated selectors accept for autotune options left at their defaults. With no `gpu_target`, the attached GPU is detected automatically. |
| `ExtensionBuildConfig` | Overrides the CUDA Toolkit root and adds Torch include and GPU library directories for building the generated extensions. Its `compiler_path` selects only the linker; set `CXX` to choose the C++ compiler. |
| `LoweringOptions` | Combines the Triton and extension build configurations and controls post-lowering validation. |
| `lower_model(module, example_inputs, *, work_dir, options)` | Collects kernel specializations from representative inputs, compiles native artifacts, rewrites the graph in place, optionally validates it, and returns a `LoweringResult`. |
| `LoweringResult` | Contains the lowered graph module, artifact directory, artifact metadata, GPU target, Torch stable-ABI target, and operator namespace. |
| `LoweringArtifact` | Identifies one generated file by DSL, artifact kind, and path relative to the lowering work directory. |

## Export and loading

| API | Description |
| --- | --- |
| `ExportOptions` | Configures validation inputs for export. PT2 export is reserved but not currently implemented. |
| `export_model(lowering, *, options)` | Scripts the lowered module and writes `model.pt` plus `manifest.json` into the lowering work directory. |
| `load_model(model_path)` | Loads every shared library listed in the manifest and returns the exported TorchScript module. |
| `aot_tensor::loadModel(model_path)` | C++ equivalent of `load_model()`; see [Loading from C++](cpp-loader.md). |

## Lowering options

By default `lower_model()` compiles for the attached GPU, registers the
generated operators in the `aot_tensor` namespace, and runs the lowered module
on every example input before returning. To set these explicitly, pass a
`LoweringOptions` as `lower_model(..., options=options)`:

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

- `dsl_configs` takes exactly one `TritonCompileConfig`.
- `work_dir` must be empty.
- Give each independently compiled artifact a unique `op_namespace` when
  several are loaded into one process.
