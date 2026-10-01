# AOT-T Debugging

Indexed by symptom. For how the pipeline works and a first-time walkthrough,
see [architecture.md](architecture.md) — including the debug env vars, which are setup
for a trace rather than a diagnostic.

Paths are repository-relative unless explicitly stated.

## `[TritonAOT] No implementation found for <kernel>`

Thrown at launch when the generated selector matched no compiled spec. The
message is self-describing by design — `gen_failure_msg` dumps five sections:

```
  Tensors:  x_ptr=Float(aligned16=true) w_ptr=BFloat16(aligned16=false) ...
  Scalars:  M=512 N=256 ...
  Constants: BLOCK_M=64 ...
  Autotune: num_warps=4 num_stages=3 ...
  Device: cc=90
```

Compare that against the guard chains at the bottom of `{kernel}.cpp`. The
usual causes:

- **A dtype nobody traced.** The spec set only covers call signatures seen
  during the compile forward pass. A new dtype at serving time matches nothing.
- **Lost 16-byte alignment.** `aligned16=false` on a tensor whose specs all
  assume alignment. Non-contiguous or offset views do this.
- **Wrong `cc`.** Specs are compiled per compute capability — an `sm80` `.so`
  on an `sm90` host matches nothing.

Fix by re-running compile with inputs that cover the missing case, so the spec
gets collected.

## `All N specs exceeded SMEM cap <bytes>; nothing left to compile`

Compile-time, from `OpsUnit.drop_oversize_specs`. Every autotune variant needed
more shared memory than the device's per-block opt-in cap, so all were dropped.
Usually an autotune config with block sizes too large for the target — narrow
the `@triton.autotune` config list, or check you are compiling for the arch you
think (`_warn_if_host_mismatches_target` logs when host and target differ).

Individual over-cap specs are dropped with a warning and are not an error; this
message means *all* of them were.

## `AOT-T kernel needs N bytes SMEM but device D max opt-in is M bytes`

Load-time, from `enable_large_smem_or_throw` in the generated `.cpp`. The
compile-time filter uses the *compiling* host's cap; this is the runtime
backstop when the serving device is smaller. Recompile targeting the serving
arch.

## No `_original.py` / `_wrapper.py` after a trace

Those are stage-4 output. `fbcode/triton_aot/example/compile.py` only runs stages 2–3 — it never
calls `transform_kernels`. Run `fbcode//triton_aot/example:publish` too, and pass
`--cleanup false`, which otherwise defaults to true and deletes the whole
compile tree on exit. See the walkthrough in `architecture.md`.

## Model output differs from its TritonCC counterpart

Use the **aott_vs_tc** skill (`fbcode/triton_aot/ops/.llms/skills/aott_vs_tc/SKILL.md`). It
compares per-op intermediate outputs and pinpoints the first divergent op plus
the first divergent `triton_aot::*` kernel.

## Backend-opts drift warning at compile time

`fbcode/aot_tensor/fb/compile/guardrails/backend_opts_snapshot.py` warns when upstream Triton's
`CUDAOptions` / `HIPOptions` fields no longer match the checked-in
`fbcode/aot_tensor/fb/compile/guardrails/resources/{cuda,hip}.json` baseline. See the **backend-opts-drift** skill for
what is checked and how to regenerate.

---

Not yet covered, because nobody has written up an observed instance: `.so`
failing to `dlopen` on an older Predictor (a stable-ABI violation — see the
rule's stable-ABI section) and TorchScript scripting failures at publish (the
`fbcode/triton_aot/ops/.llms/skills/create-aott-ops/SKILL.md` has guidance under `pre_script_check`). Add them here
when you hit one.
