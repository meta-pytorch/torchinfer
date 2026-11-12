# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# AOT compile of a ``@flyc.jit`` launcher through FlyDSL's public export API:
# ``flyc.compile`` specializes it and ``CompiledFunction.export_to_c`` writes a
# PIC object holding the launcher, its grid math, the embedded code object and
# a private copy of FlyDSL's AOT runtime, so the .so needs only libamdhip64. The
# ABI is read back from the metadata the object embeds, not hand-mirrored, so a
# signature change cannot mislaunch.

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import flydsl.compiler as flyc
import torch
from aot_tensor.compile.flydsl.abi import ExportedLauncher, read_exported_launcher
from aot_tensor.compile.flydsl.args import launcher_kwargs
from aot_tensor.types import FlyAOT


@contextmanager
def _export_env() -> Iterator[None]:
    """The FlyDSL disk cache off, so an AOT compile never writes ``~/.flydsl``,
    and the ROCm toolchain's LLVM tools first on ``PATH`` for the linker and
    objcopy ``export_to_c`` runs, so RE and devservers agree."""
    overrides = {"FLYDSL_RUNTIME_ENABLE_CACHE": "0"}
    rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    llvm_bin = os.path.join(rocm_home, "llvm", "bin") if rocm_home else None
    if llvm_bin is not None and os.path.isdir(llvm_bin):
        overrides["PATH"] = os.pathsep.join([llvm_bin, os.environ.get("PATH", "")])
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


def _placeholder(value: Any) -> Any:
    """A host tensor with ``value``'s dtype and layout. FlyDSL specializes on
    those alone and launches nothing when a tensor is not on the device, so the
    sampled call's tensors (live model state) are never written."""
    if not isinstance(value, torch.Tensor):
        return value
    return torch.empty_strided(value.size(), value.stride(), dtype=value.dtype)


def export_launcher(
    op: FlyAOT, values: dict[str, Any], out_dir: str, file_name: str, symbol: str
) -> ExportedLauncher:
    """Compile ``op``'s launcher for the sampled call ``values`` and export it
    as ``{out_dir}/{file_name}.o`` (and ``.h``). The object defines ``{symbol}``
    and ``{symbol}__*`` only, so several exported launchers link into one .so."""
    kwargs = launcher_kwargs(
        op, {name: _placeholder(value) for name, value in values.items()}
    )
    with _export_env():
        compiled = flyc.compile(op.jit_fn, **kwargs)
        if compiled is None:
            raise RuntimeError(
                "FlyDSL AOT: flyc.compile returned nothing; unset COMPILE_ONLY"
            )
        compiled.export_to_c(out_dir, file_name, symbol)
    return read_exported_launcher(os.path.join(out_dir, f"{file_name}.o"), symbol)
