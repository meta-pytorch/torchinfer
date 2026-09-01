# Copyright (c) Meta Platforms, Inc. and affiliates.


"""C++ and Python code generation for AOT-T compiled kernels.

Generates:
  - kernel.h      (header with gridDims, selector proto)
  - kernel.cpp    (cubin externs, loaders, launchers, selector)
  - _torch_op.cpp (torch op registration)
  - _meta.py      (Python autotuner meta function)
"""

import inspect
import os
import textwrap
from collections import Counter
from dataclasses import dataclass, Field
from typing import Any

# @manual=//triton:triton
import triton
from aot_tensor.compile.dtypes import CTYPES
from aot_tensor.compile.stable_types import PY_TYPES_TO_CPP_TYPES, SCALAR_TYPES

# @dep=//aot_tensor/compile/triton/templates:triton_templates
from aot_tensor.compile.template_utils import TRITON_TEMPLATES
from aot_tensor.compile.triton.arg_descriptor import (
    ArgDescriptor,
    ConstantArg,
    PointerArg,
    ScalarArg,
)
from aot_tensor.compile.triton.compat import _get_cluster_dims, get_scratch_parameters
from aot_tensor.compile.triton.launch_header import find_launch_header
from aot_tensor.compile.triton.spec_processing import AutotuneAttrs, KernelSpec, OpsUnit
from aot_tensor.compile.triton.utils import hash_kernel_name, unwrap_to_jit
from aot_tensor.constants import generated_header
from triton.runtime.jit import JITFunction

# Indent widths from the repo ``.clang-format``. Values interpolated into a
# multi-line slot are joined at these columns so they line up with the
# template text around them; keeping them named makes the two move together.
_INDENT = 2
_CONT = 4

# ---------------------------------------------------------------------------
# C++ templates
#
# Kept as module constants rather than inline f-strings so the shape of the
# emitted code is readable without stepping through the emitters, mirroring
# ``cubin_embedder.KERNEL_BINARY_ARRAY_TEMPLATE``.
#
# All are flush-left, so the ``indent=`` values their callers pass line up
# with the columns you can see here.
#
# C++ bodies follow the repo ``.clang-format``: ``IndentWidth: 2``,
# ``ContinuationIndentWidth: 4`` -- see ``_INDENT`` / ``_CONT``. Linters skip
# the emitted files (see ``constants.GENERATED_TOKEN``), so matching
# clang-format by hand is what keeps them readable next to the hand-written
# templates they are spliced into.
#
# Do not spell that marker out in this file: arc reads it as a header tag and
# silently stops linting the whole module, which is why the constant is
# assembled at runtime rather than written literally.
# ---------------------------------------------------------------------------

_CUBIN_EXTERN_TMPL = """\
extern "C" {{
extern unsigned char {sym}[];
extern const void* volatile {sym}_ptr;
}}"""

_LOADER_TMPL = """
CUfunction load_{kernel_name}(void)
{{
  thread_local std::unordered_map<int32_t, CUfunction> cache;
  auto idx = torch::stable::accelerator::getCurrentDeviceIndex();
  auto res = cache.find(idx);
  if (res != cache.end()) {{
    return res->second;
  }}
  CUfunction func;
  CUmodule mod_ptr;
  CUresult err;
  // Use pointer to cubin data to generate R_X86_64_64 relocation
  // instead of R_X86_64_32, allowing cubin data to be placed beyond 4GB
  const void *image = {kernel_name}_cubin_ptr;

  err = cuModuleLoadData(&mod_ptr, image);
  if (err != 0) {{
    const char* errStr;
    cuGetErrorString(err, &errStr);
    throw std::runtime_error(
        "cuModuleLoadData failed for "
        "{kernel_name}"
        ": error " + std::to_string(err)
        + " (" + (errStr ? errStr : "unknown") + ")");
  }}

  err = cuModuleGetFunction(&func, mod_ptr, "{cubin_name}");
  if (err != 0) {{
    const char* errStr;
    cuGetErrorString(err, &errStr);
    throw std::runtime_error(
        "cuModuleGetFunction failed for "
        "{kernel_name}"
        ": error " + std::to_string(err)
        + " (" + (errStr ? errStr : "unknown") + ")");
  }}

  enable_large_smem_or_throw({shared}, func);
  cache.emplace(idx, func);
  return func;
}}
"""

_LAUNCHER_FAST_TMPL = """
void {kernel_name}(
    {params}) {{
  CUfunction func = load_{kernel_name}();
  CUstream stream =
      grid.stream ? grid.stream : triton_aot_get_current_stream();
  {scratch_declarations}
  uint32_t grid_arr[3] = {{
      (uint32_t)grid.x, (uint32_t)grid.y, (uint32_t)grid.z}};
  {safe_name}_args_t args = {{
      {struct_init}}};
  TRITON_AOT_CU_CHECK(triton_launch_{safe_name}(
      grid_arr, stream, func, &args, global_scratch, profile_scratch));
}}
"""

_LAUNCHER_LEGACY_TMPL = """
void {kernel_name}(
    {params}) {{
  CUfunction func = load_{kernel_name}();
  cudaStream_t stream =
      grid.stream ? grid.stream : triton_aot_get_current_stream();
  {scratch_declarations}
  void *args[] = {{
      {args_str}
  }};
  {launch_call}
  TRITON_AOT_CU_CHECK(res);
}}
"""

_SELECTOR_TMPL = """
void {func_name}(
    {params}) {{
  auto cc = compute_capability();
  if (grid.x * grid.y * grid.z > 0) {{
{guarded_calls}
    std::stringstream ss;
    ss << "[TritonAOT] No implementation found for {func_name}"
{failure_msg};
    throw std::runtime_error(ss.str());
  }}
}}
"""

_TORCH_OP_TMPL = """
namespace {{
triton::aot::gridDims dims_from_vec(
    const std::vector<int64_t>& grid
) {{
  return triton::aot::gridDims(
      grid.size() > 0 ? grid[0] : 1,
      grid.size() > 1 ? grid[1] : 1,
      grid.size() > 2 ? grid[2] : 1
  );
}}

{type_comment}void {func_name}_op(
    std::vector<int64_t> grid,
    {cpp_params}
) {{
  triton::aot::{func_name}(
      dims_from_vec(grid),
      {args}
  );
}}

void {func_name}_dummy_op(
    std::vector<int64_t> grid,
    {cpp_params}
) {{
  // Do nothing.  The op is a dummy for model transform,
  // processing, and splitting services.
}}
}} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(triton_aot, m) {{
  m.def(
      {schema_literals});
}}

STABLE_TORCH_LIBRARY_IMPL(triton_aot, CUDA, m) {{
  m.impl("{func_name}", TORCH_BOX(&{func_name}_op));
}}

STABLE_TORCH_LIBRARY_IMPL(triton_aot, CPU, m) {{
  m.impl("{func_name}", TORCH_BOX(&{func_name}_dummy_op));
}}

STABLE_TORCH_LIBRARY_IMPL(triton_aot, Meta, m) {{
  m.impl("{func_name}", TORCH_BOX(&{func_name}_dummy_op));
}}
"""

# Python, so 4-space body -- the C++ constants above are the ones that follow
# clang-format.
_META_PY_TMPL = """\
def {meta_func_name}({in_args}):
    {returns_comment}
    {guards}
    {tail}
"""

_META_CPP_TMPL = """
inline std::tuple<{return_type}> {meta}({in_args}) {{
  {guards}
  {tail}
}}
"""


# ---------------------------------------------------------------------------
# Kernel naming and binary generation
# ---------------------------------------------------------------------------


def validate_unique_kernel_names(
    fn: Any,
    specs: list[KernelSpec],
    cc: int | str,
    autotune_fields: tuple[Field[Any], ...],
) -> None:
    """Fail fast when two specs map to one kernel name.

    Kernel names are cubin identity (``hash_kernel_name`` keys the binary
    files and the embedded symbols), so a collision silently overwrites
    the first spec's cubin on disk before the duplicate ``load_*`` loader
    even reaches the C++ compiler. Names are a pure function of
    ``(fn, spec, cc, autotune_fields)``, so this check costs no Triton
    compilation -- call it before ``compile_specs_parallel``.
    ``gen_guarded_calls`` re-checks at codegen time as a backstop (it also
    covers the post-``drop_oversize_specs`` spec set).
    """
    seen: dict[str, KernelSpec] = {}
    for spec in specs:
        name = gen_kernel_name(fn, spec, cc, autotune_fields)
        if name in seen:
            raise RuntimeError(
                f"[AOT-T] kernel name collision: {name!r} generated by two "
                f"distinct specs ({seen[name]} vs {spec}); distinct specs "
                f"must produce distinct names."
            )
        seen[name] = spec


def gen_kernel_name(
    fn: Any,
    spec: KernelSpec,
    cc: int | str,
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    name = fn.__name__
    sig = "_".join([p.replace("*", "p") for p in spec.signature.values()])
    const = "_".join(map(str, spec.constants.values()))
    cc_str = f"sm{cc}"
    autotune_configs = [
        f"{f.metadata['cubin_short']}{getattr(spec.autotune, f.name)}"
        for f in autotune_fields
        # Omit default-off auto_tma so enabling auto-TMA (beta) doesn't rename
        # every existing kernel; only auto_tma=True variants get a distinct name.
        if not (f.name == "auto_tma" and not getattr(spec.autotune, f.name))
    ]
    # See kernel_suffix in triton/compiler/code_generator.py.
    # ``spec.signature`` is keyed by ORIGINAL arg indices (constants
    # excluded), and the divisibility sets are in that same space -- so
    # iterate the KEYS. The previous ``enumerate(...)`` tested the dict
    # POSITION instead, garbling the specialization suffix whenever folded
    # args preceded a marked one: names misdescribed their cubins, and two
    # variant specs differing only in marks past the fold boundary could
    # COLLIDE on name (aliasing each other's cubin files). Purely a naming
    # bug -- the compiled attrs were always correct (P2467214344).
    suffix = ""
    for i in spec.signature:
        suffix += str(i)
        if i in spec.divisible_by_16:
            suffix += "d"
        if i in spec.divisible_by_8:
            suffix += "e"
    return "_".join([name, cc_str, sig, const] + autotune_configs + [suffix])


def gen_cubin(kernel_name: str, kernel: Any, install_dir: str, backend: str) -> str:
    """Generate kernel binary file (.cubin or .hsaco) and return extern declaration.

    Args:
        kernel_name: Full kernel name including specialization suffix.
        kernel: Compiled Triton kernel object containing binary in kernel.asm.
        install_dir: Directory to write binary file.
        backend: GPU backend ("cuda" or "hip").

    Returns:
        C++ extern declaration for the kernel binary array.
    """
    hashed = hash_kernel_name(kernel_name)
    if backend == "hip":
        binary_file = f"{install_dir}/{hashed}.hsaco"
        with open(binary_file, "wb") as hsaco:
            hsaco.write(kernel.asm["hsaco"])
        target_symbol_name = f"{kernel_name}_cubin"
    else:
        binary_file = f"{install_dir}/{hashed}.cubin"
        with open(binary_file, "wb") as cubin:
            cubin.write(kernel.asm["cubin"])
        target_symbol_name = f"{kernel_name}_cubin"

    # We return extern declarations for both the array and its pointer.
    # The pointer is used by gen_loader() to generate R_X86_64_64 relocations
    # instead of R_X86_64_32, which allows the .triton section to be placed
    # beyond the 4GB address limit in large binaries.
    # Note: The pointer is volatile to prevent optimizer constant-propagation.
    return _CUBIN_EXTERN_TMPL.format(sym=target_symbol_name)


def gen_loader(kernel_name: str, cubin_name: str, shared: int) -> str:
    # TODO(changpan): Extract inline cuModuleLoadData/cuModuleGetFunction error
    # handling into a shared helper to reduce generated code size.
    return _LOADER_TMPL.format(
        kernel_name=kernel_name, cubin_name=cubin_name, shared=shared
    )


# ---------------------------------------------------------------------------
# Launcher codegen (per-spec)
# ---------------------------------------------------------------------------


def gen_launcher_params(
    descriptors: list[ArgDescriptor],
    signature: dict[int, str],
    indent: int = 0,
) -> str:
    args = ["gridDims grid"]
    for d in descriptors:
        if d.index in signature:
            args.append(d.launcher_param(signature[d.index]))
    sep = ",\n" + " " * indent if indent else ", "
    return sep.join(args)


def gen_launch_args(
    func: JITFunction[list[Any]],
    spec: KernelSpec,
) -> list[str]:
    """Generate kernel launch argument list (pointers to non-constant arguments)."""
    args = []
    for i, arg in enumerate(func.arg_names):
        if i in spec.constants:
            continue
        assert i in spec.signature, f"Argument {i} ({arg}) does not appear in signature"
        args.append(f"&{arg}")
    return args


def _gen_launch_call(
    kernel: Any,
    backend: str,
    shared: int,
    warp_size: int,
    num_warps: int,
) -> str:
    """C++ launch call. Cluster path on cuda emits ``cuLaunchKernelEx`` +
    ``CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION``; otherwise legacy
    ``cuLaunchKernel``. User grid is in *programs* — multiply by
    ``cluster_dims`` to get the CTA grid CUDA expects (mirrors upstream
    Triton's C ``_launch``).
    """
    block_x = f"{warp_size} * {num_warps}"
    cluster_dims = _get_cluster_dims(kernel)
    if backend != "cuda" or tuple(cluster_dims) == (1, 1, 1):
        return (
            f"auto res = cuLaunchKernel(func, grid.x, grid.y, grid.z, "
            f"{block_x}, 1, 1, {shared}, stream, args, NULL);"
        )

    # upstream Triton contract: cluster_dims is always a 3-tuple.
    cx, cy, cz = cluster_dims
    lines = [
        "CUlaunchAttribute _cluster_attrs[1] = {};",
        "_cluster_attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;",
        f"_cluster_attrs[0].value.clusterDim = {{ {cx}, {cy}, {cz} }};",
        "CUlaunchConfig _launch_config = {};",
        f"_launch_config.gridDimX = grid.x * {cx};",
        f"_launch_config.gridDimY = grid.y * {cy};",
        f"_launch_config.gridDimZ = grid.z * {cz};",
        f"_launch_config.blockDimX = {block_x};",
        "_launch_config.blockDimY = 1;",
        "_launch_config.blockDimZ = 1;",
        f"_launch_config.sharedMemBytes = {shared};",
        "_launch_config.hStream = stream;",
        "_launch_config.attrs = _cluster_attrs;",
        "_launch_config.numAttrs = 1;",
        "auto res = cuLaunchKernelEx(&_launch_config, func, args, NULL);",
    ]
    # Joined at the column gen_launcher interpolates this at.
    return ("\n" + " " * _INDENT).join(lines)


def _launch_header_available() -> bool:
    """Whether the shared launch core header (launch.h) can be vendored into the
    generated extension build. Thin predicate over the shared candidate lookup
    so gen_launcher's Level-1 gate and
    NvidiaExtensionBuilder._vendor_launch_header always agree on the same paths.
    The Level-1 launcher_src calls ``triton_launch_<name>()`` declared in
    launch.h; when the header is not shippable, gen_launcher falls back to the
    legacy launcher instead of emitting an unresolvable ``#include``.
    """
    return find_launch_header() is not None


def gen_launcher(
    kernel_name: str,
    func: JITFunction[list[Any]],
    kernel: Any,
    shared: int,
    warp_size: int,
    spec: KernelSpec,
    descriptors: list[ArgDescriptor],
    backend: str,
) -> str:
    # Fast path: use compiler-generated Level 1 launcher source when available.
    # This gives us cuLaunchKernelEx, cluster dims, PDL support for free.
    # NOTE: The wrapper below uses CUDA types (CUfunction, CUstream, CUdeviceptr),
    # so we only enter this path for CUDA targets. A future CPU backend may emit
    # its own launcher_src, but would need a separate wrapper.
    launcher_src = kernel.asm.get("launcher_src", None)
    if (
        launcher_src is not None
        and "cubin" in kernel.asm  # Level 1 wrapper is CUDA-only for now
        and os.environ.get("TRITON_AOT_FORCE_LEGACY_LAUNCHER") != "1"
        # launch.h must be vendorable into the build; otherwise fall through to
        # the legacy launcher rather than emit an unresolvable #include.
        and _launch_header_available()
    ):
        # Strip #include lines from launcher_src — the downstream BUCK target
        # (triton_kernel.bzl) already depends on triton_launch_h, and the
        # include would be ill-formed inside the namespace triton::aot scope
        # where KERNEL_SPECS is emitted.
        launcher_src = "\n".join(
            line
            for line in launcher_src.splitlines()
            if not line.strip().startswith("#include")
        )
        params = gen_launcher_params(descriptors, spec.signature, indent=_CONT)

        # Scratch is NOT part of the args struct: the launch.h ABI passes
        # global_scratch / profile_scratch as separate trailing params of
        # triton_launch_<name> (see make_launcher_src). scratch_declarations
        # emits the local CUdeviceptr vars we forward by value to the call
        # below; this mirrors TritonCC's gen_launcher. (scratch_args is the
        # legacy-fallback void*[] form and is unused on this path.)
        scratch_declarations, _ = get_scratch_parameters(kernel, backend)

        # Build the args struct initializer: { (CUdeviceptr)ptr_arg, scalar_arg, ... }
        struct_fields = []
        for i, arg in enumerate(func.arg_names):
            if i in spec.constants:
                continue
            if i in spec.signature and spec.signature[i].startswith("*"):
                struct_fields.append(f"(CUdeviceptr){arg}")
            else:
                struct_fields.append(arg)
        struct_init = (",\n" + " " * (_INDENT + _CONT)).join(struct_fields)
        # See the legacy path below: compat.py joins these at its own indent.
        scratch_declarations = ("\n" + " " * _INDENT).join(
            line.strip() for line in scratch_declarations.splitlines()
        )

        # Extract the safe name used in the launcher_src for the args struct/function
        metadata_name = kernel.metadata.name
        safe_name = metadata_name.replace(".", "_")

        return (
            "\n"
            + launcher_src
            + _LAUNCHER_FAST_TMPL.format(
                kernel_name=kernel_name,
                params=params,
                scratch_declarations=scratch_declarations,
                safe_name=safe_name,
                struct_init=struct_init,
            )
        )

    # Fallback: original path (cuLaunchKernel, no cluster/PDL support)
    params = gen_launcher_params(descriptors, spec.signature, indent=_CONT)
    args = gen_launch_args(func, spec)

    scratch_declarations, scratch_args = get_scratch_parameters(kernel, backend)
    args.extend(scratch_args)

    # compat.py is shared with tritoncc and joins these at its own indent;
    # re-join at ours rather than depend on that constant.
    scratch_declarations = ("\n" + " " * _INDENT).join(
        line.strip() for line in scratch_declarations.splitlines()
    )
    args_str = (",\n" + " " * (_INDENT + _CONT)).join(args)
    launch_call = _gen_launch_call(
        kernel, backend, shared, warp_size, spec.autotune.num_warps
    )

    return _LAUNCHER_LEGACY_TMPL.format(
        kernel_name=kernel_name,
        params=params,
        scratch_declarations=scratch_declarations,
        args_str=args_str,
        launch_call=launch_call,
    )


# ---------------------------------------------------------------------------
# Selector codegen (invariant)
# ---------------------------------------------------------------------------


def gen_selector_params(
    descriptors: list[ArgDescriptor],
    autotune_fields: tuple[Field[Any], ...],
    *,
    with_defaults: bool = False,
    indent: int = 0,
) -> str:
    """Generate C++ selector function parameter list.

    If ``with_defaults`` is True, autotune-field params are emitted as
    ``T name=<default>`` (used by ``gen_selector_proto``).

    ``indent`` puts one parameter per line, continuation lines padded to that
    column.
    """
    args = ["gridDims grid"]
    args.extend(d.selector_param() for d in descriptors)

    py_types = AutotuneAttrs.field_python_types()
    for f in autotune_fields:
        if with_defaults:
            # C++ bool literals are lowercase (true/false), unlike Python's repr.
            dv = (
                ("true" if f.default else "false")
                if isinstance(f.default, bool)
                else f.default
            )
            suffix = f"={dv}"
        else:
            suffix = ""
        args.append(f"{py_types[f.name].__name__} {f.name}{suffix}")
    sep = ",\n" + " " * indent if indent else ", "
    return sep.join(args)


def gen_launcher_call_args(
    descriptors: list[ArgDescriptor],
    signature: dict[int, str],
    indent: int = 0,
) -> str:
    args = ["grid"]
    for d in descriptors:
        if d.index in signature:
            if isinstance(d, PointerArg):
                args.append(f"{d.name}.value().data_ptr()")
            elif isinstance(d, ScalarArg) and signature[d.index] != d.triton_dtype:
                args.append(f"static_cast<{CTYPES[signature[d.index]]}>({d.name})")
            else:
                args.append(d.name)
    sep = ",\n" + " " * indent if indent else ", "
    return sep.join(args)


@dataclass(frozen=True)
class Guard:
    """One ``if (...)`` predicate gating a launcher specialization.

    Guards are accumulated per spec and rendered as a single prefix chain, so
    the generated dispatch is a run of nested one-armed ``if``s ending in the
    ``return``.  Holding them as values rather than concatenated text keeps
    each producer independently testable.
    """

    cond: str

    def render(self) -> str:
        return f"if ({self.cond})"


def _dtype_guards(
    spec: KernelSpec,
    desc_by_idx: dict[int, ArgDescriptor],
) -> list[Guard]:
    """Tensor dtype guards; different specs may bind different dtypes."""
    guards = []
    for i, ttype in spec.signature.items():
        d = desc_by_idx[i]
        if not isinstance(d, PointerArg):
            continue
        guards.append(Guard(f"{d.name}.has_value()"))
        guards.append(Guard(f"{d.name}.value().scalar_type() == {SCALAR_TYPES[ttype]}"))
    return guards


def _narrowing_guards(
    spec: KernelSpec,
    desc_by_idx: dict[int, ArgDescriptor],
) -> list[Guard]:
    """Range guards for specs binding a narrower int type than the selector."""
    guards = []
    for i, dtype in spec.signature.items():
        d = desc_by_idx[i]
        if isinstance(d, ScalarArg) and dtype != d.triton_dtype and dtype == "i32":
            guards.append(Guard(f"fits_i32({d.name})"))
    return guards


def _constant_guards(
    spec: KernelSpec,
    desc_by_idx: dict[int, ArgDescriptor],
) -> list[Guard]:
    """Equality guards pinning each constexpr arg to this spec's value."""
    guards = []
    for i, val in spec.constants.items():
        arg = desc_by_idx[i].name
        if isinstance(val, bool):
            guards.append(Guard(arg if val else f"!({arg})"))
        elif isinstance(val, str):
            guards.append(Guard(f'{arg} == "{val}"'))
        elif val is None:
            guards.append(Guard(f"!{arg}.has_value()"))
        else:
            guards.append(Guard(f"{arg} == {val}"))
    return guards


def _autotune_guards(
    spec: KernelSpec,
    autotune_fields: tuple[Field[Any], ...],
) -> list[Guard]:
    guards = []
    for f in autotune_fields:
        v = getattr(spec.autotune, f.name)
        # C++ bool literals are lowercase (true/false), unlike Python repr.
        v_str = ("true" if v else "false") if isinstance(v, bool) else v
        guards.append(Guard(f"{f.name} == {v_str}"))
    return guards


def _divisible_by_16_guards(
    spec: KernelSpec,
    desc_by_idx: dict[int, ArgDescriptor],
) -> list[Guard]:
    guards = []
    for i in spec.divisible_by_16:
        arg = desc_by_idx[i].name
        if i in spec.signature:
            if spec.signature[i].startswith("*"):
                guards.append(
                    Guard(f"(((uintptr_t){arg}.value().data_ptr()) % 16) == 0")
                )
            else:
                guards.append(Guard(f"({arg} % 16) == 0"))
        elif i in spec.constants:
            assert (spec.constants[i] % 16) == 0
    return guards


def _divisible_by_8_guards(
    spec: KernelSpec,
    desc_by_idx: dict[int, ArgDescriptor],
) -> list[Guard]:
    guards = []
    for i in spec.divisible_by_8:
        arg = desc_by_idx[i].name
        if i in spec.signature:
            # divisible_by_8 is only applied to int
            if not spec.signature[i].startswith("*"):
                guards.append(Guard(f"({arg} % 8) == 0"))
        elif i in spec.constants:
            assert (spec.constants[i] % 8) == 0
    return guards


def spec_guards(
    spec: KernelSpec,
    desc_by_idx: dict[int, ArgDescriptor],
    autotune_fields: tuple[Field[Any], ...],
) -> list[Guard]:
    """All guards for one spec, in emission order.

    Order is load-bearing: it fixes the nesting of the generated ``if`` chain.
    """
    return [
        *_dtype_guards(spec, desc_by_idx),
        *_narrowing_guards(spec, desc_by_idx),
        *_constant_guards(spec, desc_by_idx),
        *_autotune_guards(spec, autotune_fields),
        *_divisible_by_16_guards(spec, desc_by_idx),
        *_divisible_by_8_guards(spec, desc_by_idx),
    ]


# Columns inside the text ``gen_guarded_calls`` returns. ``gen_selector``
# splices that block into ``_SELECTOR_TMPL``'s ``if`` body, so these are
# relative to it and stay correct wherever the template puts the block.
_CALL_ARG_INDENT = _INDENT + _CONT


def gen_guarded_calls(
    func: JITFunction[list[Any]],
    unit: OpsUnit,
    descriptors: list[ArgDescriptor],
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    desc_by_idx: dict[int, ArgDescriptor] = {d.index: d for d in descriptors}
    # Codegen-time backstop over the post-``drop_oversize_specs`` set;
    # single collision policy shared with the pre-compile call site in
    # ``pipeline``.
    validate_unique_kernel_names(func, unit.specs, unit.cc, autotune_fields)
    calls = []
    for spec in unit.specs:
        kernel_name = gen_kernel_name(func, spec, unit.cc, autotune_fields)
        args = gen_launcher_call_args(
            descriptors, spec.signature, indent=_CALL_ARG_INDENT
        )
        lines = [g.render() for g in spec_guards(spec, desc_by_idx, autotune_fields)]
        # Built as a list so a spec with no guards emits a bare return rather
        # than a leading blank line.
        lines.append(
            f"{' ' * _INDENT}return {kernel_name}(\n{' ' * _CALL_ARG_INDENT}{args});"
        )
        calls.append("\n".join(lines) + "\n")
    return "\n".join(calls)


def gen_selector_proto(
    descriptors: list[ArgDescriptor],
    func_name: str,
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    params = gen_selector_params(
        descriptors, autotune_fields, with_defaults=True, indent=_CONT
    )
    # Trailing newline so the template's __TRITON_AOT_GENERATE_END__ marker,
    # which follows directly, starts its own line.
    return f"void {func_name}(\n    {params});\n"


def gen_failure_msg(
    descriptors: list[ArgDescriptor],
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    """Generate C++ ``<<``-chain for the dispatch-failure error message.

    Groups parameters by category (Tensors / Scalars / Constants /
    Autotune / Device).  Tensor entries include aligned16 status.
    """
    tensors: list[str] = []
    scalars: list[str] = []
    constants: list[str] = []

    for d in descriptors:
        if isinstance(d, PointerArg):
            dtype_expr = (
                f"({d.name}.has_value()"
                f" ? torch::headeronly::toString({d.name}.value().scalar_type())"
                f' : "nullptr")'
            )
            align_expr = (
                f"(({d.name}.has_value()"
                f" && (((uintptr_t){d.name}.value().data_ptr()) % 16) == 0)"
                f' ? "true" : "false")'
            )
            # Split across lines: the two ternaries make a single-line entry
            # roughly 230 characters wide.
            tensors.append(
                f'" {d.name}="\n<< {dtype_expr}\n<< "(aligned16="'
                f'\n<< {align_expr}\n<< ")"'
            )
        elif isinstance(d, ScalarArg):
            scalars.append(f'" {d.name}=" << {d.name}')
        elif isinstance(d, ConstantArg):
            constants.append(f'" {d.name}=" << {d.name}')

    autotune: list[str] = [f'" {f.name}=" << {f.name}' for f in autotune_fields]

    # One operand per line -- this is what you read when dispatch fails.
    join = "\n<< "

    sections: list[str] = []
    if tensors:
        sections.append('"\\n  Tensors:"' + join + join.join(tensors))
    if scalars:
        sections.append('"\\n  Scalars:"' + join + join.join(scalars))
    if constants:
        sections.append('"\\n  Constants:"' + join + join.join(constants))
    sections.append('"\\n  Autotune:"' + join + join.join(autotune))
    sections.append('"\\n  Device: cc=" << cc')

    return join.join(sections)


def gen_selector(
    func: JITFunction[list[Any]],
    unit: OpsUnit,
    descriptors: list[ArgDescriptor],
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    params = gen_selector_params(descriptors, autotune_fields, indent=_CONT)
    guarded_calls = gen_guarded_calls(func, unit, descriptors, autotune_fields)
    failure_msg = gen_failure_msg(descriptors, autotune_fields)
    return _SELECTOR_TMPL.format(
        func_name=func.__name__,
        params=params,
        # Both land in _SELECTOR_TMPL's `if` body: one level for the
        # function body, one for the `if`.
        guarded_calls=textwrap.indent(guarded_calls, " " * (2 * _INDENT)),
        failure_msg=textwrap.indent("<< " + failure_msg, " " * (2 * _INDENT)),
    )


# ---------------------------------------------------------------------------
# Torch op codegen (invariant)
# ---------------------------------------------------------------------------


def _as_cpp_string_literals(text: str, indent: int, width: int = 72) -> str:
    """Render *text* as adjacent C++ string literals, one per line.

    The compiler concatenates them, and chunks keep their separators, so the
    value is unchanged. A chunk ending in an odd number of backslashes would
    escape its closing quote -- safe here because schema backslashes come from
    ``gen_str_wrap``'s ``\\"``, always followed by a quote, never by ``", "``.
    """
    parts = text.split(", ")
    chunks = [p + ", " for p in parts[:-1]] + parts[-1:]

    lines: list[str] = []
    current = ""
    for chunk in chunks:
        if current and len(current) + len(chunk) > width:
            lines.append(current)
            current = chunk
        else:
            current += chunk
    if current:
        lines.append(current)

    return ("\n" + " " * indent).join(f'"{line}"' for line in lines)


def gen_cpp_op_params(
    descriptors: list[ArgDescriptor],
    autotune_fields: tuple[Field[Any], ...],
    indent: int = 0,
) -> str:
    """Comma-joined C++ parameter list; ``indent`` puts one per line."""
    args = [d.cpp_op_param() for d in descriptors]
    py_types = AutotuneAttrs.field_python_types()
    for f in autotune_fields:
        args.append(f"{PY_TYPES_TO_CPP_TYPES[py_types[f.name]]} {f.name}")
    sep = ",\n" + " " * indent if indent else ", "
    return sep.join(args)


def gen_torch_op_params(
    descriptors: list[ArgDescriptor],
    default_values: dict[str, Any],
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    args = []

    def gen_str_wrap(value: Any) -> Any:
        return f'\\"{value}\\"' if isinstance(value, str) else value

    def gen_default_str(arg: str) -> str:
        return (
            f" = {gen_str_wrap(default_values[arg])}" if arg in default_values else ""
        )

    for d in descriptors:
        args.append(d.torch_schema_param(gen_default_str(d.name)))
    py_types = AutotuneAttrs.field_python_types()
    for f in autotune_fields:
        args.append(f"{py_types[f.name].__name__} {f.name}={f.default}")
    return ", ".join(args)


def gen_torch_op_schema(
    func: JITFunction[list[Any]],
    descriptors: list[ArgDescriptor],
    default_values: dict[str, Any],
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    return f"{func.__name__}(int[] grid, {gen_torch_op_params(descriptors, default_values, autotune_fields)}) -> ()"


def gen_torch_op(
    func: JITFunction[list[Any]],
    descriptors: list[ArgDescriptor],
    default_values: dict[str, Any],
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    cpp_params = gen_cpp_op_params(descriptors, autotune_fields, indent=_CONT)
    arg_names = list(func.arg_names) + [f.name for f in autotune_fields]
    args = (",\n" + " " * (_INDENT + _CONT)).join(arg_names)
    schema = gen_torch_op_schema(func, descriptors, default_values, autotune_fields)

    # Generate a comment noting which tensor params are non-optional but
    # promoted to Tensor? for TorchScript compatibility.
    promoted = [
        d.name for d in descriptors if isinstance(d, PointerArg) and not d.is_optional
    ]
    type_comment = ""
    if promoted:
        type_comment = (
            f"// Note: {', '.join(promoted)} are non-optional but use Tensor? "
            "for TorchScript compatibility.\n"
            "// Dispatch uses HAS_XXX constexpr ints, not tensor presence.\n"
        )
    return _TORCH_OP_TMPL.format(
        type_comment=type_comment,
        func_name=func.__name__,
        cpp_params=cpp_params,
        args=args,
        schema_literals=_as_cpp_string_literals(schema, indent=_INDENT + _CONT),
    )


# ---------------------------------------------------------------------------
# Tuner meta codegen
# ---------------------------------------------------------------------------


def key_names_and_idx(func: Any) -> tuple[list[str], list[int]]:
    if hasattr(func, "key_idx"):
        arg_names = [func.arg_names[idx] for idx in func.key_idx]
        key_idx = func.key_idx
    else:
        arg_names = func.keys
        key_idx = [func.arg_names.index(arg) for arg in arg_names]
    return arg_names, key_idx


def is_non_empty_mapping_of_type(obj: object, value_type: type[Any]) -> bool:
    """Check if object is a non-empty dict with all values of specific type"""
    if not obj or not isinstance(obj, dict):
        return False

    return all(isinstance(value, value_type) for value in obj.values())


def gen_tuner_meta_py(
    func: Any,
    tuner_fallback: bool,
    unit: OpsUnit,
    backend: str,
) -> str:
    vals = []

    guard_list = []
    autotune_fields = unit.autotune_fields

    # Use custom meta generation function if available (PPO hook).
    # Pass down the field table so PPO renders the same arity AOT-T
    # will unpack.  The signature probe keeps old packages working.
    if hasattr(func, "gen_autotune_select_meta_src"):
        hook = func.gen_autotune_select_meta_src
        if "autotune_field_defaults" in inspect.signature(hook).parameters:
            return hook(
                unit.constant_types,
                backend=backend,
                constexpr_keys=unit.constexpr_keys,
                autotune_field_defaults={f.name: f.default for f in autotune_fields},
            )
        return hook(unit.constant_types)

    if hasattr(func, "cache") and is_non_empty_mapping_of_type(
        func.cache, triton.runtime.autotuner.Config
    ):
        # auto tuned configs
        arg_names, key_idx = key_names_and_idx(func)

        in_args = ", ".join(
            [
                f"{name}: {unit.constant_types[idx].__name__ if idx in unit.constant_types else 'int'}"
                for idx, name in zip(key_idx, arg_names)
            ]
        )

        kernel_constexpr_keys = list(unit.constexpr_keys)
        return_names = kernel_constexpr_keys + [f.name for f in autotune_fields]

        for key, cfg in func.cache.items():
            attrs = AutotuneAttrs.from_cfg(cfg, autotune_fields)
            val = tuple(
                [cfg.kwargs[k] for k in kernel_constexpr_keys]
                + [getattr(attrs, f.name) for f in autotune_fields]
            )
            vals.append(val)
            equations = []
            for arg, value in zip(arg_names, key):
                if isinstance(value, str):
                    equations.append(f"{arg} == '{value}'")
                elif isinstance(value, bool):
                    equations.append(f"{arg} == {int(value)}")
                else:
                    equations.append(f"{arg} == {value}")
            guard_list.append(f"if {' and '.join(equations)}: return {val}")

    else:
        # default configs — single spec, use specs[0]
        in_args = ""
        arg_names = [f.name for f in autotune_fields]
        return_names = [f.name for f in autotune_fields]
        val = tuple(getattr(unit.specs[0].autotune, f.name) for f in autotune_fields)
        vals.append(val)

    name = unwrap_to_jit(func).__name__
    meta_func_name = name + "_meta"

    guards = "\n    ".join(guard_list)

    fmt_args = ", ".join([f"{{{arg_name}}}" for arg_name in arg_names])

    raise_runtime_error_str = (
        f"""raise RuntimeError(f"No autotuning config found for {name}({fmt_args})")"""
    )
    fallback_str = f"""return {Counter(vals).most_common(1)[0][0]}"""

    returns_comment = f"# Returns: ({', '.join(return_names)})"

    return generated_header("#") + _META_PY_TMPL.format(
        meta_func_name=meta_func_name,
        in_args=in_args,
        returns_comment=returns_comment,
        guards=guards,
        tail=fallback_str if tuner_fallback else raise_runtime_error_str,
    )


def gen_tuner_meta_cpp(
    func: Any,
    tuner_fallback: bool,
    constant_types: dict[int, type[Any]],
) -> str:
    # TODO(changpan): This C++ inline _meta is currently dead code — no C++ caller
    # invokes it.  The Python _meta.py (gen_tuner_meta_py) is the only consumer.
    # Double check, try remove this and the TUNER_META_CPP template region.
    def infer_arg_type(idx: int) -> str:
        if idx in constant_types:
            return PY_TYPES_TO_CPP_TYPES[constant_types[idx]]
        else:
            return "int64_t"

    arg_names, key_idx = key_names_and_idx(func)

    in_args = ", ".join(
        [f"{infer_arg_type(idx)} {name}" for idx, name in zip(key_idx, arg_names)]
    )

    vals = []
    guard_list = []
    for key, cfg in func.cache.items():
        val = list(cfg.kwargs.values()) + [cfg.num_warps, cfg.num_stages]
        val = tuple(val)
        vals.append(val)
        equations = []
        for arg, value in zip(arg_names, key):
            if isinstance(value, str):
                equations.append(f'{arg} == "{value}"')
            elif isinstance(value, bool):
                equations.append(f"{arg} == {int(value)}")
            else:
                equations.append(f"{arg} == {value}")
        guard_list.append(f"if ({' && '.join(equations)}) return std::make_tuple{val};")
    guards = ("\n" + " " * _INDENT).join(guard_list)
    name = unwrap_to_jit(func).__name__
    meta = name + "_meta"
    fmt_args = ", ".join([f"{arg_name}" for arg_name in arg_names])
    raise_runtime_error_str = f"""throw std::runtime_error("No autotuning config found for {name}({fmt_args})");"""
    fallback_str = f"""return std::make_tuple{Counter(vals).most_common(1)[0][0]};"""
    # Infer the return type from the actual values
    return_type = _infer_return_type(vals[0])
    return _META_CPP_TMPL.format(
        return_type=return_type,
        meta=meta,
        in_args=in_args,
        guards=guards,
        tail=fallback_str if tuner_fallback else raise_runtime_error_str,
    )


def _infer_return_type(vals: tuple[Any, ...]) -> str:
    types = [PY_TYPES_TO_CPP_TYPES.get(type(val)) for val in vals]
    try:
        # pyre-fixme[6]: For 1st argument expected
        #  `Iterable[typing_extensions.LiteralString]` but got `List[Optional[str]]`.
        return ", ".join(types)
    except TypeError:  # one of the types cannot be inferred, e.g. `None`
        raise ValueError("Cannot infer return type from `vals`")


# ---------------------------------------------------------------------------
# Top-level codegen entry points
# ---------------------------------------------------------------------------


def generate_header_content(
    tuned_func: triton.runtime.autotuner.Autotuner | None,
    func: JITFunction[list[Any]],
    unit: OpsUnit,
    descriptors: list[ArgDescriptor],
    tuner_fallback: bool,
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    """Generate the content of the .h header file."""
    tuner_meta_cpp = (
        gen_tuner_meta_cpp(tuned_func, tuner_fallback, unit.constant_types)
        if tuned_func
        else ""
    )
    selector_proto = gen_selector_proto(descriptors, func.__name__, autotune_fields)
    return TRITON_TEMPLATES.render(
        "kernel.h",
        {
            "TUNER_META_CPP": tuner_meta_cpp,
            "SELECTOR_PROTO": selector_proto,
        },
    )


def generate_kernel_cpp_content(
    func: JITFunction[list[Any]],
    unit: OpsUnit,
    descriptors: list[ArgDescriptor],
    prefix: str,
    generated_specs: list[str],
    autotune_fields: tuple[Field[Any], ...],
    backend: str,
) -> str:
    """Generate the content of the kernel .cpp file.

    *backend* is only used for hipification; per-backend autotune field
    selection is delegated to *autotune_fields*.
    """
    kernel_specs = "\n".join(generated_specs)
    selector = gen_selector(func, unit, descriptors, autotune_fields)

    # On AMD, apply hipification to generated code (KERNEL_SPECS, SELECTOR)
    # Templates are already hipified at load time (from hip/ subdirectory)
    if backend == "hip":
        from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper

        kernel_specs = maybe_hipify_code_wrapper(kernel_specs, force_hipify=True)
        selector = maybe_hipify_code_wrapper(selector, force_hipify=True)

    header_include = f'#include "{prefix}.h"\n'
    # The Level-1 launcher_src emitted into KERNEL_SPECS uses launch.h
    # (triton_kernel_launch_desc_t / triton_launch_kernel); its own #include is
    # stripped because it sits inside namespace triton::aot, so include it here
    # at file scope. Resolved via the triton_launch_h dep in the BUCK build and
    # via the vendored copy in the runtime extension build (see
    # nvidia_extension_builder._vendor_launch_header). Only emit it when a
    # Level-1 launcher was actually generated: gen_launcher falls back to the
    # legacy cuLaunchKernel launcher when launch.h is unavailable, and a
    # legacy-only build must not require the header.
    if "triton_launch_kernel" in kernel_specs:
        header_include += '#include "nvidia/backend/launch.h"\n'

    cpp_content = TRITON_TEMPLATES.render(
        "kernel.cpp",
        {
            "HEADER_INCLUDE": header_include,
            "KERNEL_SPECS": kernel_specs,
            "SELECTOR": selector,
        },
    )
    return cpp_content


def generate_torch_op_content(
    func: JITFunction[list[Any]],
    descriptors: list[ArgDescriptor],
    prefix: str,
    default_values: dict[str, Any],
    autotune_fields: tuple[Field[Any], ...],
) -> str:
    """Generate the content of the torch_op .cpp file."""
    torch_op_content = gen_torch_op(func, descriptors, default_values, autotune_fields)
    torch_content = TRITON_TEMPLATES.render(
        "torch_op.cpp",
        {
            "HEADER_INCLUDE": f'#include "{prefix}.h"\n',
            "TORCH_OP": torch_op_content,
        },
    )
    return torch_content
