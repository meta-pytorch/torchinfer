# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from aot_tensor.api.lowering import LoweringResult


_MODEL_FILE = "model.pt"
_MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class ExportOptions:
    pt2_format: bool = False
    validation_inputs: Sequence[Sequence[Any]] = ()


_DEFAULT_EXPORT_OPTIONS = ExportOptions()


def _write_manifest(lowering: LoweringResult) -> None:
    manifest_path = lowering.work_dir / _MANIFEST_FILE
    payload = {
        "shared_libraries": [
            artifact.path.as_posix()
            for artifact in lowering.artifacts
            if artifact.kind == "shared_library"
        ]
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def export_model(
    lowering: LoweringResult,
    *,
    options: ExportOptions = _DEFAULT_EXPORT_OPTIONS,
) -> Path:
    """Export ``lowering.module`` as ``model.pt`` in its work directory."""
    if options.pt2_format:
        raise NotImplementedError("PT2 export is not supported yet")

    model_path = (lowering.work_dir / _MODEL_FILE).resolve()
    manifest_path = lowering.work_dir / _MANIFEST_FILE
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"AOT Tensor export output already exists in {lowering.work_dir}"
        )

    scripted: torch.jit.ScriptModule = torch.jit.script(lowering.module)
    for inputs in options.validation_inputs:
        scripted(*inputs)

    try:
        torch.jit.save(scripted, str(model_path))
        _write_manifest(lowering)
    except Exception:
        model_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise

    return model_path
