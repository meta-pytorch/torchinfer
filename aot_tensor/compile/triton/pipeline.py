# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""AOT-T compilation pipeline.

Orchestrates: spec processing → Triton native compile → C++ / Python codegen.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from types import FrameType, ModuleType
from typing import Any, Callable

# @manual=//triton:triton
import triton
import triton.compiler
from aot_tensor.compile.triton.arg_descriptor import (
    ArgDescriptor,
    build_arg_descriptors,
)
from aot_tensor.compile.triton.codegen import (
    gen_cubin,
    gen_kernel_name,
    gen_launcher,
    gen_loader,
    gen_torch_op_schema,
    gen_tuner_meta_py,
    generate_header_content,
    generate_kernel_cpp_content,
    generate_torch_op_content,
)
from aot_tensor.compile.triton.spec_processing import (
    AutotuneAttrs,
    gen_compile_arg,
    KernelSpec,
    OpsUnit,
    RawKernelSpec,
)
from aot_tensor.compile.triton.utils import is_autotuner, unwrap_to_jit
from aot_tensor.constants import AOTT_OP_SCHEMAS_FILE_NAME
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import KernelInterface

logger: logging.Logger = logging.getLogger(__name__)


def compile_specs_parallel(
    specs: list[KernelSpec],
    install_dir: str,
    module: str,
    name: str,
    gpu_target: GPUTarget,
    import_module: Callable[[str], ModuleType],
    descriptors: list[ArgDescriptor],
) -> list[tuple[str, int]]:
    """Compile kernel specs in parallel; return per-spec ``(code, shared)``.

    Sequential when ``TRITON_AOT_DEBUG=1``. Pure dispatch -- the
    SMEM-cap filter lives in ``OpsUnit.drop_oversize_specs``.
    """

    debug = os.environ.get("TRITON_AOT_DEBUG", "0") == "1"
    if debug:
        return [
            spec_gen(
                install_dir,
                spec,
                module,
                name,
                gpu_target,
                import_module,
                descriptors,
            )
            for spec in specs
        ]
    max_workers = mp.cpu_count() // 2 + 1
    with ThreadPoolExecutor(max_workers=min(len(specs), max_workers)) as executor:
        return list(
            executor.map(
                lambda spec: spec_gen(
                    install_dir,
                    spec,
                    module,
                    name,
                    gpu_target,
                    import_module,
                    descriptors,
                ),
                specs,
            )
        )


# For each spec, generate a kernel:
# - cubin
# - loader
# - launcher
def spec_gen(
    install_dir: str,
    spec: KernelSpec,
    module: str,
    name: str,
    gpu_target: GPUTarget,
    import_module: Callable[[str], ModuleType],
    descriptors: list[ArgDescriptor],
) -> tuple[str, int]:
    """Compile one spec; return ``(cubin+loader+launcher, metadata.shared)``.

    Pure compile -- no policy decisions. The SMEM-cap filter lives in
    ``OpsUnit.drop_oversize_specs``.
    """
    func = unwrap_to_jit(getattr(import_module(module), name))

    # Generate cubin.
    autotune_fields = AutotuneAttrs.fields_for(gpu_target.backend)
    kernel_name = gen_kernel_name(func, spec, gpu_target.arch, autotune_fields)

    compile_arg = gen_compile_arg(spec, func)
    options = {f.name: getattr(spec.autotune, f.name) for f in autotune_fields}
    compile_kwargs = {
        "target": gpu_target,
        "options": options,
    }
    kernel = triton.compiler.compile(*compile_arg, **compile_kwargs)

    metadata_name = kernel.metadata.name
    metadata_shared = kernel.metadata.shared

    cubin = gen_cubin(kernel_name, kernel, install_dir, gpu_target.backend)
    out = [
        cubin,
        # Generate loader.
        gen_loader(kernel_name, metadata_name, metadata_shared),
        # Generate launcher.
        gen_launcher(
            kernel_name,
            func,
            kernel,
            metadata_shared,
            gpu_target.warp_size,
            spec,
            descriptors,
            gpu_target.backend,
        ),
    ]
    return "".join(out), metadata_shared


def sigchld_handler(signum: int, frame: FrameType | None) -> None:
    sketchy_signals = map(int, [signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS])
    try:
        # Consume all pending SIGCHLDs, looking for unexpected failures
        while True:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            if os.WIFSIGNALED(status) and os.WTERMSIG(status) in sketchy_signals:
                logger.error(
                    f"Child process {pid} exited catastrophically with signal {os.WTERMSIG(status)}, terminating!"
                )

                # Avoid triggering atexit etc which can get stuck and behave improperly
                # because multiprocessing sets up an atexit handler to join workers
                # (sigh).  We want to exit, now, so use os._exit instead of sys.exit.
                os._exit(1)
    except ChildProcessError:
        pass


def compile_to_cpp(
    func: KernelInterface[list[Any]] | triton.runtime.autotuner.Autotuner,
    base_specs: list[RawKernelSpec],
    install_dir: str,
    prefix: str,
    *,
    gpu_target: GPUTarget,
    import_module: Callable[[str], ModuleType],
    default_values: dict[str, Any] | None = None,
    tuner_fallback: bool = False,
) -> None:
    """Compile a Triton kernel into .cpp, .h, _torch_op.cpp, _meta.py files.

    Args:
        func: Triton JITFunction or Autotuner to compile.  If autotuned,
            ``@triton.autotune`` must be the *outermost* wrapper -- nested
            cases like ``heuristics(autotune(jit))`` would be misidentified
            as non-autotuned (silently dropping autotuner.cache expansion
            and miscompiling heuristic-injected constants).
        base_specs: List of kernel specialization specs.
        install_dir: Directory to output generated files.
        prefix: Kernel name prefix, e.g., "_addmm_fwd".
        gpu_target: GPU target for compilation.
        import_module: torch.package importer for loading kernels source code.
        default_values: Default values for kernel arguments.
        tuner_fallback: If True, generate fallback tuner code.
    """
    tuned_func = func if is_autotuner(func) else None
    # pyre-ignore[6]: Attributes verified by is_autotuner
    unit = OpsUnit.from_raw_specs(base_specs, gpu_target, tuned_func)
    default_values = {} if default_values is None else default_values

    func_unwrapped = unwrap_to_jit(func)
    descriptors = build_arg_descriptors(func_unwrapped, unit)

    # Python's multiprocessing.Pool class is not great at handling unexpected child
    # failures such as segfaults.  Account for this by temporarily installing a signal
    # handler that considers such signals a catastrophic compilation failure.  If not
    # for this, the Pool will deadlock.
    if threading.current_thread() is threading.main_thread():
        previous_child_handler = signal.signal(signal.SIGCHLD, sigchld_handler)
    else:
        previous_child_handler = None

    func = func_unwrapped

    # sanity check to make sure args with default values are always at the end
    has_default_value_arg = False
    for name in func.arg_names:
        if name in default_values:
            has_default_value_arg = True
        elif has_default_value_arg:
            raise RuntimeError(
                f"default values must be at the end of the argument list. {func.arg_names=} {default_values=}"
            )

    h_out = f"{install_dir}/{prefix}.h"
    cu_out = f"{install_dir}/{prefix}.cpp"
    torch_out = f"{install_dir}/{prefix}_torch_op.cpp"
    py_out = f"{install_dir}/{prefix}_meta.py"
    schema_out = f"{install_dir}/{AOTT_OP_SCHEMAS_FILE_NAME}"

    autotune_fields = unit.autotune_fields

    # Generate kernel.h file
    h_content = generate_header_content(
        tuned_func,  # pyre-ignore[6]: Autotuner when set (verified by is_autotuner)
        func,
        unit,
        descriptors,
        tuner_fallback,
        autotune_fields,
    )

    with open(h_out, "w") as fp:
        fp.write(h_content)

    generated_specs = compile_specs_parallel(
        unit.specs,
        install_dir,
        func.__module__,
        func.__name__,
        gpu_target,
        import_module,
        descriptors,
    )
    unit, filtered_specs = unit.drop_oversize_specs(generated_specs)

    # Generate kernel.cpp file
    cu_content = generate_kernel_cpp_content(
        func,
        unit,
        descriptors,
        prefix,
        filtered_specs,
        autotune_fields,
        gpu_target.backend,
    )
    with open(cu_out, "w") as fp:
        fp.write(cu_content)

    # Generate torch_op.cpp file
    torch_op_content = generate_torch_op_content(
        func, descriptors, prefix, default_values, autotune_fields
    )

    with open(torch_out, "w") as fp:
        fp.write(torch_op_content)

    with open(schema_out, "w") as fp:
        json.dump(
            [gen_torch_op_schema(func, descriptors, default_values, autotune_fields)],
            fp,
            indent=2,
        )
        fp.write("\n")

    meta_func = tuned_func if tuned_func else func
    with open(py_out, "w") as fp:
        fp.write(
            gen_tuner_meta_py(
                meta_func,
                tuner_fallback,
                unit,
            )
        )

    if previous_child_handler is not None:
        signal.signal(signal.SIGCHLD, previous_child_handler)
