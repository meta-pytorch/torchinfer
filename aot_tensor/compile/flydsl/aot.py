# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# AOT compile of a ``@flyc.jit`` launcher into a PIC object plus its packed ABI.
#
# FlyDSL compiles it (``COMPILE_ONLY``: normal trace + MLIR, no launch); the
# resulting IR is dumped as an object holding the launcher, its grid math and
# the embedded ``gpu.binary``, so the C++ side only packs arguments. The object
# calls three ``mgpu*`` runtime functions that the torch-op template defines, so
# no FlyDSL library ships with the .so. The ABI is read from FlyDSL's JIT
# arguments, not hand-mirrored, so a signature change cannot mislaunch.

import ctypes
import inspect
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aot_tensor.compile.flydsl.abi import (
    AbiSlot,
    C_IDENTIFIER,
    ExportedLauncher,
    missing_runtime_functions,
)
from aot_tensor.compile.flydsl.compat import private_attr
from flydsl._mlir import execution_engine, ir
from flydsl.compiler import jit_argument, jit_function, protocol
from flydsl.expr import numeric, typing as flydsl_typing

_GPU_MODULE_INIT = "flydsl_gpu_module_init"
_GPU_MODULE_LOAD_TO_DEVICE = "flydsl_gpu_module_load_to_device"


# =====================================================================
# ABI description
# =====================================================================


def _scalar_ctype_name(value: Any) -> str:
    if value.signed is not None:
        if value.width == 1:
            return "bool"
        return f"{'int' if value.signed else 'uint'}{value.width}"
    if isinstance(value, numeric.Float32):
        return "float"
    if isinstance(value, numeric.Float64):
        return "double"
    raise NotImplementedError(
        f"FlyDSL AOT does not support {type(value).__name__} scalar arguments"
    )


def argument_abi(arg_name: str, value: Any) -> list[AbiSlot]:
    """Packed slots one FlyDSL JIT argument occupies."""
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    if isinstance(value, jit_argument.MemRefJitArg):
        slots = [AbiSlot("tensor_data", arg_name, "pointer", pointer_size)]
        if value.is_layout_dynamic:
            shape_dims = tuple(value.shape_dyn_indices)
            stride_dims = tuple(value.stride_dyn_indices)
            stride_bits = 32 if value.use_32bit_stride else 64
            size = 4 * len(shape_dims) + stride_bits // 8 * len(stride_dims)
            slots.append(
                AbiSlot(
                    "tensor_layout",
                    arg_name,
                    "bytes",
                    size,
                    shape_dims=shape_dims,
                    stride_dims=stride_dims,
                    stride_bits=stride_bits,
                )
            )
        return slots
    if isinstance(value, flydsl_typing.Stream):
        return [AbiSlot("stream", arg_name, "pointer", pointer_size)]
    if not isinstance(value, numeric.Numeric):
        raise NotImplementedError(
            "FlyDSL AOT supports tensor, scalar and stream arguments; got "
            f"{type(value).__name__} for {arg_name!r}"
        )
    abi_spec = protocol.c_abi_spec(value)
    if len(abi_spec) != 1:
        raise NotImplementedError(
            f"FlyDSL AOT scalar {arg_name!r} must occupy exactly one C ABI slot"
        )
    ctype, _fill = abi_spec[0]
    return [
        AbiSlot("scalar", arg_name, _scalar_ctype_name(value), ctypes.sizeof(ctype))
    ]


def _launcher_abi(sig: Any, bound: Any, jit_args: list[Any]) -> tuple[AbiSlot, ...]:
    abi: list[AbiSlot] = []
    jit_arg_index = 0
    for param_name in bound.arguments:
        annotation = sig.parameters[param_name].annotation
        if annotation is not inspect.Parameter.empty and (
            flydsl_typing.Constexpr.is_constexpr_annotation(annotation)
            or jit_argument.is_type_param_annotation(annotation)
        ):
            continue
        abi.extend(argument_abi(param_name, jit_args[jit_arg_index]))
        jit_arg_index += 1
    return tuple(abi)


# =====================================================================
# compile
# =====================================================================


@contextmanager
def _compile_only() -> Iterator[None]:
    """FlyDSL's own compile-without-launch mode, with the disk cache off so an
    AOT compile never writes ``~/.flydsl``."""
    overrides = {"COMPILE_ONLY": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0"}
    saved = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _compile(
    launcher: Any, kwargs: dict[str, Any]
) -> tuple[str, str, tuple[AbiSlot, ...]]:
    """(IR text, entry name, packed ABI) of ``launcher`` compiled for the
    sample ``kwargs`` through FlyDSL's own JIT pipeline."""
    if not isinstance(launcher, jit_function.JitFunction):
        raise TypeError(
            f"FlyDSL AOT expects a @flyc.jit function, got {type(launcher).__name__}"
        )
    with _compile_only():
        launcher(**kwargs)
    if private_attr(launcher, "_has_self_param") is not False:
        raise NotImplementedError("FlyDSL AOT does not support bound-method launchers")

    sig = private_attr(launcher, "_sig")
    bound = sig.bind(**kwargs)
    bound.apply_defaults()
    # The artifact may come from an earlier eager call with the same signature
    # (compile-only then returns on FlyDSL's call-state fast path), so look it
    # up by FlyDSL's own cache key rather than relying on the last compile.
    cache_key = private_attr(launcher, "_build_full_cache_key")(
        bound.arguments,
        effective_hints=private_attr(launcher, "_effective_compile_hints")(),
    )
    artifact = private_attr(launcher, "_mem_cache").get(cache_key)
    if artifact is None:
        raise RuntimeError(
            f"FlyDSL AOT: no compiled artifact for {launcher.func.__name__}"
        )
    # CompiledArtifact's pickling state is the stable view of its contents.
    state = artifact.__getstate__()
    if state["processor_refs"]:
        raise NotImplementedError(
            "FlyDSL AOT does not support Python post-load processors"
        )
    if state["link_libs"]:
        raise NotImplementedError("FlyDSL AOT does not support extern-linked libs")

    with private_attr(jit_function, "_create_mlir_context")():
        _, jit_args, _, _ = jit_argument.convert_to_jit_arguments(sig, bound)
    return artifact.ir, state["entry"], _launcher_abi(sig, bound, jit_args)


def export_launcher(
    launcher: Any, kwargs: dict[str, Any], object_path: str, prefix: str
) -> ExportedLauncher:
    """Compile ``launcher`` for the sample ``kwargs`` (nothing is launched) and
    write it as a PIC object whose global symbols all carry ``prefix``, so
    several exported launchers can link into one .so."""
    if not C_IDENTIFIER.fullmatch(prefix):
        raise ValueError(f"invalid C symbol: {prefix!r}")
    ir_text, entry, abi = _compile(launcher, kwargs)
    with private_attr(jit_function, "_create_mlir_context")():
        module = ir.Module.parse(ir_text)
        _use_explicit_module_loader(module)
        # The wheel's .pyi lags the binding, which does take enable_pic.
        engine = execution_engine.ExecutionEngine(
            module,
            opt_level=3,
            enable_pic=True,  # pyrefly: ignore[unexpected-keyword]
        )
        engine.dump_to_object_file(object_path)
    object_file = Path(object_path)
    missing = missing_runtime_functions(_nm(object_file, "--undefined-only"))
    if missing:
        raise NotImplementedError(
            f"FlyDSL AOT: the launcher calls MLIR GPU runtime functions "
            f"{missing} that templates/flydsl_torch_op.cpp does not define; "
            "add them there and to abi.RUNTIME_FUNCTIONS"
        )
    renamed = _prefix_global_symbols(object_file, prefix)

    def exported(name: str) -> str:
        if name not in renamed:
            raise RuntimeError(f"FlyDSL AOT: exported object does not define {name!r}")
        return renamed[name]

    return ExportedLauncher(
        object_path=object_path,
        entry_symbol=exported(f"_mlir_{entry}"),
        module_init_symbol=exported(_GPU_MODULE_INIT),
        module_load_symbol=exported(_GPU_MODULE_LOAD_TO_DEVICE),
        abi=abi,
    )


# =====================================================================
# export helpers
# =====================================================================


def _use_explicit_module_loader(module: Any) -> None:
    """Switch each ``gpu.binary``'s loader (``offloadingHandler``) to FlyDSL's
    ``#fly.explicit_module``. The default ``#gpu.select_object`` loads the
    binary from an ORC JIT init function, which never runs in a plain .so; the
    explicit one exports ``flydsl_gpu_module_init`` /
    ``flydsl_gpu_module_load_to_device`` for the caller to run."""
    handler = ir.Attribute.parse("#fly.explicit_module")
    binaries = [
        op for op in module.body.operations if op.operation.name == "gpu.binary"
    ]
    if not binaries:
        raise RuntimeError("FlyDSL AOT: compiled module contains no gpu.binary")
    for binary in binaries:
        binary.operation.attributes["offloadingHandler"] = handler


def _tool(*names: str) -> str:
    """Prefer the ROCm toolchain's LLVM tools so RE and devservers agree."""
    rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    search = [os.path.join(rocm_home, "llvm", "bin")] if rocm_home else []
    for name in names:
        for directory in search:
            candidate = os.path.join(directory, name)
            if os.access(candidate, os.X_OK):
                return candidate
        found = shutil.which(name)
        if found is not None:
            return found
    raise RuntimeError(f"FlyDSL AOT export requires one of {names} on PATH")


def _nm(object_path: Path, *flags: str) -> list[str]:
    listing = subprocess.run(
        [_tool("llvm-nm", "nm"), "-P", *flags, str(object_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.split()[0] for line in listing.splitlines() if line]


def _prefix_global_symbols(object_path: Path, prefix: str) -> dict[str, str]:
    """Rename every global symbol ``object_path`` defines to
    ``{prefix}__{name}``; returns old -> new. References inside the object go
    through relocations, so they follow the rename."""
    renamed = {
        name: f"{prefix}__{name}"
        for name in _nm(object_path, "--defined-only", "--extern-only")
    }
    command = [_tool("llvm-objcopy", "objcopy")]
    for old_name, new_name in renamed.items():
        command.extend(["--redefine-sym", f"{old_name}={new_name}"])
    subprocess.run(
        [*command, str(object_path)], check=True, capture_output=True, text=True
    )
    return renamed
