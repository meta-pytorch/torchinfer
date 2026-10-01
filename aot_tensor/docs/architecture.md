# AOT-T Architecture

Paths are repository-relative unless explicitly stated.

The DSL-agnostic compile core — spec collection (`infer_spec`), spec processing
(`OpsUnit`), C++ `gen_*` codegen — lives in this package. Its primary entry
points are
`fbcode/aot_tensor/compile/compile_state.py` (`get_aott_compile_path`),
`fbcode/aot_tensor/compile/aott_compile.py` (`AOTTCompileSession`), and
`fbcode/aot_tensor/compile/triton/adapter.py` (spec collection +
`TritonAdapter`).

## The 6 stages

1. **Kernel Wrapper Authoring** (`fbcode/triton_aot/ops/`) — hand-written source, checked in; nothing executes at this stage. Three parts per op: (a) wrap the upstream Triton kernel with `@triton_aot()`, imported from `aot_tensor.types`; (b) a `_triton_aot_*` function that allocates outputs, computes the grid, and launches `kernel[grid](...)`; (c) a public `aot_triton_kernel_wrapper_*` that routes eager-PyTorch vs the AOT path. The kernels themselves usually live upstream (`generative_recommenders`, `hammer`, `fbr`); `fbcode/triton_aot/ops/` only adds the marker and the wrappers.
2. **AOT Compile** (`fbcode/aot_tensor/compile/`) — three-phase: (a) **spec collection** — inside `AOTTCompileSession`, the forward pass fires `TritonAOT.run()` on each marker authored in 1a, recording one `KernelSpec` per distinct call signature; (b) **Triton native compile** — upstream `triton.compiler.compile()` → `.cubin`, one per spec; (c) **AOTT C++ codegen** — generates `.cpp`, `.h`, `_torch_op.cpp`, `_meta.py` glue around the cubins from the C++ templates.
3. **Build** (`fbcode/aot_tensor/build/`) — compiles `.cubin` + generated C++ → `.so` via setuptools (NVIDIA CUDA + AMD ROCm via hipify).
4. **Kernel Transform** (`fbcode/aot_tensor/transform/`) — three-part: (a) **Python wrapper codegen** — AST-rewrites `kernel[grid](...)` → `torch.ops.triton_aot.kernel(...)`, generates `{fn}_wrapper.py`; (b) **Runtime swap** — loads wrappers and replaces FX graph node targets; (c) optional **graph transforms** (`fbcode/aot_tensor/fb/transform/graph_transforms.py`) — config-driven FX rewrites from `GraphTransformSettings` thrift (e.g. `fbcode/aot_tensor/fb/transform/matmul_kn_alignment.py` pads matmul K/N so cuBLAS picks the fast sm90 bf16 kernel).
5. **TGIF Publish** (`fbcode/tgif/publish/`) — bundles `.so` + transformed model into the .predictor archive.
6. **Runtime** (`fbcode/caffe2/caffe2/fb/predictor/lowered_module/`) — Predictor extracts `.so` from the archive, `dlopen`s it, loads the TorchScript module, updates weights; model calls `torch.ops.triton_aot.*`.

Only stages 2–4 write files you can inspect, which is why the artifact tables
below cover just those:

- **Stage 1** is source, checked in; nothing executes.
- **Stage 5** emits one archive, elsewhere. `fbcode/triton_aot/example/gen_toy_model_package.py`
  is the runnable example — it collects every built `.so` keyed by
  `AOTT_FILE_PREFIX`, scripts the transformed module as
  `AOTT_LOWERED_MODULE_NAME`, and writes a single TorchScript file via
  `torch.jit.save(..., _extra_files=...)`. (`fbcode/triton_aot/example/publish.py --serialization`
  does a torch.package variant uploaded to Manifold instead.)
- **Stage 6** writes nothing — Predictor reads the archive and `dlopen`s.

## Where each stage runs

Authoring on the TGIF client (CPU, no GPU — it imports only the type layer plus
`fbcode/triton_aot/ops/`); stages 2–4 in one process on an RE GPU host — the forward pass inside
`AOTTCompileSession` collects specs, its `__exit__` compiles and builds, then
`transform_kernels` runs *after* the context exits; stage 5 on TGIF publish
infra; stage 6 on the Sigrid Predictor serving host.

## AOT-T code map

| Path | Purpose |
|---|---|
| `fbcode/triton_aot/ops/` | Pre-built Triton kernel wrappers (~65) — where most new work lands. Read `fbcode/triton_aot/ops/.llms/skills/create-aott-ops/SKILL.md` before adding one |
| `fbcode/aot_tensor/fb/api.py` | Entry point: `aott_lower_full()` — full pipeline (eager lower → weight-signature check → TorchScript) in one call. Downstream users import this rather than assembling `AOTTCompileSession` + `transform_kernels` manually. The production caller is the RE consumer, `fbcode/aiplatform/modelstore/model_generation/utils/gpu_lowering_util.py` |
| `fbcode/aot_tensor/types.py` | Kernel marker types. `fbcode/triton_aot/types.py` remains a compatibility re-export for archives and readers that still use the legacy module name |
| `fbcode/aot_tensor/fb/transform/preprocess.py` | `unwrap_aott_wrapper_nodes` — retraces outer `{aot,ppo}_{triton,cutedsl}_kernel_wrapper_*` FX leaves to expose inner launcher calls (opt-in via `aott_lower_full(..., unwrap_wrapped_aott_nodes=True)`) |
| `fbcode/aot_tensor/build/`, `fbcode/aot_tensor/fb/runtime_deps.py` | Extension builders and the Meta runtime-link dependency closure |
| `fbcode/aot_tensor/transform/` | Wrapper-codegen driver (`kernel_wrapper_codegen.py`) + FX node swap (`replace_kernels.py`) + combined API (`transform_kernels.py`, exports `assert_weight_signature_unchanged`) |
| `fbcode/aot_tensor/fb/transform/` | Meta-only graph transforms (`graph_transforms.py`, `matmul_kn_alignment.py`) |
| `fbcode/aot_tensor/compile/cutedsl/` | CuTeDSL codegen (`codegen.py`, `pipeline.py`), eager JIT path (`eager.py`), C++ `templates/` |
| `fbcode/aot_tensor/fb/compile/guardrails/` | fb-only: `backend_opts_snapshot.py` — warns at compile time when upstream Triton `CUDAOptions`/`HIPOptions` fields drift from the checked-in `resources/{cuda,hip}.json` baseline. On a warning, see the **backend-opts-drift** skill |
| `fbcode/aot_tensor/fb/compile/triton/autotune_cache_overrides.py` | fb-only: resolves autotune-cache overrides from Manifold into the plain dict the core consumes |
| `fbcode/triton_aot/example/` | Toy models + compile/publish entry points (`compile.py`, `publish.py`). Note every `fbcode/triton_aot/ops/` wrapper pulls in `hammer.utils.should_trigger_eager_impl`, so no example is hammer-free |
| `fbcode/triton/cc/autotune_attrs.py` | TritonCC-only autotuning defaults (`AUTOTUNE_ATTRs`). See `fbcode/triton_aot/.llms/skills/SYNC_TRACKER/SKILL.md` |

CuTeDSL is loaded lazily — do NOT add `//aot_tensor/compile/cutedsl:*` to
Triton-only deps (pulls cutlass into
Triton-only paths and breaks APF Python 3.10 builds; see D106589601).

## Runtime (Sigrid Predictor side)

Not in this package — runtime loading lives in
`fbcode/caffe2/caffe2/fb/predictor/lowered_module/`: `AOTTPredictor.cpp/.h`
(strategy — `tryLoadAOTTPredictor()` picks from-scratch vs inplace delta update)
calls `AOTTModuleLoading.cpp/.h` (execution — extract `.so` from the archive,
`dlopen` via `AOTTLibraryRegistry` to avoid double-loading, load the TorchScript
module, update meta-device weights + compressed indices mapping).

## Build system

- **Framework code** is built by **BUCK** (`python_library`, `python_unittest`).
- **Generated `.so` files** are built by **setuptools + gcc/clang** (not buck) —
  the AOT tool cannot re-invoke buck at runtime. Torch headers are resolved from
  the statically-linked interpreter; `libcuda` is linked dynamically.

## Debug environment variables

| Var | Effect |
|---|---|
| `TRITON_AOT_PATH_PREFIX` | Parent dir for the compile output tree (default `/var/tmp`, a `mkdtemp` that is easy to lose). Read in `fbcode/aot_tensor/compile/compile_state.py` |
| `TRITON_AOT_DEBUG=1` | Compile specs sequentially instead of via `ThreadPoolExecutor`, so breakpoints work. Read in `fbcode/aot_tensor/compile/triton/pipeline.py` |
| `TRITON_AOT_FORCE_LEGACY_LAUNCHER=1` | Skip the Level-1 `launch.h` launcher; emit the legacy `cuLaunchKernel` path. Read in `fbcode/aot_tensor/compile/triton/codegen.py` |

Nothing sets `TRITON_AOT_PATH_PREFIX` in production, so RE compiles land in
`/var/tmp/triton_aot_compile_<random>/` and are never cleaned up — fine on an
ephemeral worker, worth an occasional `rm -rf` on a devserver.

## Tracing a kernel

A guide for reading this codebase for the first time. AOT-T is a codegen
pipeline, so **the generated artifacts are the ground truth** and usually
explain it faster than the source does. Make one inspectable run, read what it
wrote, then go to the source that wrote it.

### 1. Make an inspectable run

```bash
export TRITON_AOT_PATH_PREFIX=$HOME/aott_trace   # keep the output tree
export TRITON_AOT_DEBUG=1                        # sequential compile, breakpoints work
mkdir -p $HOME/aott_trace

# stages 2-4: publish compiles internally, then transforms
buck2 run @mode/opt fbcode//triton_aot/example:publish -- --model toy --cleanup false

find $HOME/aott_trace -type f | sort
```

The toy model is the smallest path — two `Linear` layers over
`aot_triton_kernel_wrapper_addmm`, no CuTeDSL, no HSTU. Needs a GPU host.

`--cleanup false` matters: it defaults to true and deletes the whole compile
tree on exit. Run from `fbcode/` so `@mode/opt` resolves.

`fbcode//triton_aot/example:compile` runs stages 2-3 only — it never calls `transform_kernels`, so
it produces no `_wrapper.py`. The two cannot be split across separate buck runs:
`AOTTCompileState` is a per-process singleton and `get_aott_compile_path()`
mkdtemps a fresh directory each run, so a second process would see no collected
specs. `:publish` compiles internally for this reason.

#### What you are compiling

The input chain, outermost first. Worth opening these before the run so the
generated output has something to correspond to:

| | |
|---|---|
| Entry point | `fbcode/triton_aot/example/compile.py::toy_model_compile` — `symbolic_trace`s the model, then runs one forward pass inside `AOTTCompileSession` with `x = randn(512, 1024)` on cuda |
| Model | `fbcode/triton_aot/example/models/toy.py` — `ToyModule` is two `Linear` layers, 1024→256→128. `Linear.forward` (line 38) is a single call to `aot_triton_kernel_wrapper_addmm(bias, x, weight)` |
| Op wrapper | `fbcode/triton_aot/ops/triton_addmm.py` — the three stage-1 parts: the `@triton_aot()` wrap producing `_addmm_fwd` (line 38), the launcher `_triton_aot_addmm_fwd` (line 57), and the public `aot_triton_kernel_wrapper_addmm` (line 105) |
| The kernel | `fbcode/generative_recommenders/ops/triton/triton_addmm.py:509` — `@triton.jit def _addmm_fwd(x_ptr, w_ptr, y_ptr, z_ptr, M, N, K, stride_…)`. **Not in this package**; `fbcode/triton_aot/ops/` only wraps it |

That kernel signature is worth a look before reading any output: four pointer
args and the scalars are exactly what shows up encoded in the cubin symbol name
and in the selector's guard chain.

Do **not** trace via a GPU test — `NviTestBase.tearDown` calls
`cleanup_compile_path()`, which deletes the tree before you can read it.

### 2. What each stage wrote

One directory per kernel: `{compile_path}/{module_basename}_{kernel_name}/` for
Triton (e.g. `triton_addmm__addmm_fwd/`), `cutedsl_{module_basename}_{op_name}/`
for CuTeDSL.

Names below are the `toy` model's, so they match what you actually see. Note the
two halves are named after *different things*: stage 2–3 artifacts take the
**kernel** name (`_addmm_fwd`), stage 4 artifacts take the **wrapper function**
name (`_triton_aot_addmm_fwd`, from `node_target.__name__` — the inner launcher, not the public `aot_triton_kernel_wrapper_addmm` router, because that is what FX leaves as the node).

Generators live in `fbcode/aot_tensor/compile/triton/` unless noted.

#### From compile + build (stages 2–3)

| File | Generated by | Worth reading |
|---|---|---|
| `_addmm_fwd.cpp` | `fbcode/aot_tensor/compile/triton/codegen.py::generate_kernel_cpp_content` + `fbcode/aot_tensor/compile/triton/templates/kernel.cpp` | **yes — the selector especially** |
| `_addmm_fwd.h` | `fbcode/aot_tensor/compile/triton/codegen.py::generate_header_content` + `fbcode/aot_tensor/compile/triton/templates/kernel.h` | skim |
| `_addmm_fwd_torch_op.cpp` | `fbcode/aot_tensor/compile/triton/codegen.py::generate_torch_op_content` + `fbcode/aot_tensor/compile/triton/templates/torch_op.cpp` | skim |
| `_addmm_fwd_meta.py` | `fbcode/aot_tensor/compile/triton/codegen.py::gen_tuner_meta_py` | skim |
| `kernel_<hash>.cubin` | `fbcode/aot_tensor/compile/triton/codegen.py::gen_cubin` (name hashed to stay under `NAME_MAX`) — **one per surviving spec, so the count varies run to run** | no (binary) |
| `aott_op_schemas.json` | `fbcode/aot_tensor/compile/triton/codegen.py::gen_torch_op_schema` | if debugging TorchScript |
| `_addmm_fwd_autotune_cache` | `fbcode/aot_tensor/compile/triton/adapter.py::_resolve_autotune_cache`, only if autotuned | if debugging variant explosion |
| `_addmm_fwd_embedded_kernels_autogen.cpp` | `fbcode/aot_tensor/build/triton/cubin_embedder.py::generate_cpp_for_kernel_binaries` | no (generated bulk) |
| `addmm_fwd.so` | `fbcode/aot_tensor/build/triton/nvidia_extension_builder.py::NvidiaExtensionBuilder.build`. Leading underscore stripped (`kernel_name.lstrip("_")`), unlike every sibling | no |
| `build_temp/` | setuptools scratch — `.o` files under a nested mirror of the absolute source path | no |
| `__pycache__/` | left by `replace_kernels` importing the generated wrapper | no |

**Read `fbcode/aot_tensor/compile/triton/pipeline.py::compile_to_cpp` alongside this table.** It is ~70 lines,
declares all five output paths up front, and makes one call per file — the map
for everything above.

#### From transform (stage 4)

| File | Generated by | Worth reading |
|---|---|---|
| `_triton_aot_addmm_fwd_original.py` | `fbcode/aot_tensor/transform/wrapper_codegen_utils.py::generate_wrapper_files_skeleton` | as the diff baseline |
| `_triton_aot_addmm_fwd_wrapper.py` | same | **yes — diff it against the original** |

Both land in the same per-kernel directory. If absent, you ran a
compile-only entry point — stage 4 needs compile and transform in one process.

**Cubin count is not stable across runs.** Autotune benchmarks at collection
time, so the same model can resolve its call sites to the same config (one spec
after dedup, one cubin) or to different configs (two specs, two cubins). Read
`_addmm_fwd_meta.py` to see how yours resolved — one line per autotune cache
entry.

#### The two worth real attention

**`diff _triton_aot_addmm_fwd_original.py _triton_aot_addmm_fwd_wrapper.py`**
— this single diff teaches the whole transform stage. You will see
`_addmm_fwd[grid](...)` become `torch.ops.triton_aot._addmm_fwd(grid, ...)`, the
`_addmm_fwd_meta` call injected above the grid assignment, and the
`.so`-loading preamble prepended.

**The selector at the bottom of `_addmm_fwd.cpp`** — a flat chain of
`if (guard) if (guard) return _addmm_fwd_sm80_...(args);`. That *is* the
dispatch model: one `.so` per kernel holding N specializations, picked by guards
on dtype, int range, constant values, autotune params, and divisibility. Once
you have seen it, `gen_guarded_calls` reads as transcription.

The cubin symbol name encodes the whole specialization —
`_addmm_fwd_sm80_pfp32_×4_i32_×7_<constexprs>_w2_s5_cta1_<divisibility>` — which
is why the guards can be generated mechanically from it.

### 3. Walk the five seams

Each is a place where the data changes shape. Roughly in dependency order; give
each one a sitting.

1. **Call interception** (stage 2a) — `fbcode/aot_tensor/types.py` (`TritonAOT.run`,
   `AOTTMarkerMeta`) and `fbcode/aot_tensor/compile/aott_compile.py` (`enable_spec_collection`).
   The dependency is inverted: the type layer never imports the compile layer;
   the session pushes a collector onto the marker class. Breakpoint on the
   `collector = type(self).spec_collector` line and watch it flip from `None`.

2. **Spec inference** (stage 2a) — `fbcode/aot_tensor/compile/triton/adapter.py`
   (`infer_spec`, `_collect_triton_spec`). Dtype from `mangle_type`, alignment
   from `data_ptr() % 16`, scalars always `i64`. To see the result:

   ```python
   from aot_tensor.compile.compile_state import get_kernel_specs
   with AOTTCompileSession():
       fx_m(x)
   print(get_kernel_specs("triton"))
   ```

3. **Spec → compile variants** (stage 2b) — `fbcode/aot_tensor/compile/triton/spec_processing.py`,
   the densest code in the system. **Read its module docstring first** — the
   A1/A2/A3/B argument taxonomy it defines is what makes `codegen.py` legible,
   and close to unreadable without. Then `OpsUnit.from_raw_specs`, a six-step
   pipeline of independently readable named helpers.

4. **Variants → C++** (stage 2c) — `fbcode/aot_tensor/compile/triton/pipeline.py`
   (`compile_to_cpp`), then the `gen_*` functions in `fbcode/aot_tensor/compile/triton/codegen.py`, then
   `fbcode/aot_tensor/compile/triton/templates/kernel.cpp` for the fixed scaffolding vs the
   `__TRITON_AOT_GENERATE_*__` holes codegen fills. The templates are real,
   readable C++; codegen only swaps the marked regions.

5. **FX rewrite** (stage 4) — `fbcode/aot_tensor/transform/transform_kernels.py`, then the
   `_original.py`/`_wrapper.py` diff you already read, then
   `fbcode/aot_tensor/compile/triton/adapter.py` (`TritonAOTOperatorTransform`)
   and `fbcode/aot_tensor/transform/replace_kernels.py`.

### Without a GPU

Most of seams 3–4 are reachable on CPU — `fbcode/aot_tensor/compile/tests/pipeline_test.py` mocks
`triton.compiler.compile` entirely:

```bash
buck2 test fbcode//aot_tensor/compile/tests:spec_processing_test
buck2 test fbcode//aot_tensor/compile/tests:codegen_test
buck2 test fbcode//aot_tensor/compile/tests:pipeline_test
```

`fbcode/aot_tensor/compile/tests/spec_processing_test.py` and `fbcode/aot_tensor/compile/tests/codegen_test.py` are the two largest files in
either package and serve as executable documentation for the two hardest seams.
When a spec-processing rule looks arbitrary, search these for the case — it is
usually there with a comment. You lose the artifact reading, which is the most
valuable part, so treat this as a fallback.

### Gotchas

- **`python3` from inside the compatibility package breaks.** `fbcode/triton_aot/types.py` shadows
  the stdlib `types` module, so any ad-hoc `python3` run with the package as cwd
  fails on `import enum` / `import re`. Run from elsewhere.
- **`fbcode/triton_aot/ops/` is the safe place to experiment.** It is where multi-author work
  happens; the rest of the package is largely a single-owner migration surface
  at the moment.

Hit an error rather than a confusion? [debugging.md](debugging.md) is indexed by
symptom — selector dispatch failures, SMEM caps, missing wrapper files, numeric
divergence vs TritonCC.
