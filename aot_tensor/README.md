# aot_tensor

AOT-Tensor: pipeline that compiles GPU kernels in a PyTorch model ahead of
time (Triton, cuTeDSL, and FlyDSL backends), rewrites their call sites in the
`torch.fx` graph, and exports the model together with its native libraries.
Subfolders hold the public `api` (exporting/loading/lowering), per-DSL
`arg_specs`, `compile` adapters, `build` extension builders, and `transform`
kernel-call rewrites, plus `tests` and the `fb`/`github` packaging mirrors.
