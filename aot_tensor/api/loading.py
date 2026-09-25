# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


_MANIFEST_FILE = "manifest.json"


def load_model(
    model_path: os.PathLike[str] | str,
) -> torch.jit.ScriptModule:
    """Load missing custom-op libraries, then load a TorchScript model."""
    path = Path(model_path).resolve()
    payload: object = json.loads(
        (path.parent / _MANIFEST_FILE).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("AOT Tensor manifest must be a JSON object")
    libraries = payload.get("shared_libraries")
    if not isinstance(libraries, list) or not all(
        isinstance(library, str) for library in libraries
    ):
        raise ValueError("AOT Tensor manifest shared_libraries must be a list of paths")

    for relative_path in libraries:
        library = str((path.parent / relative_path).resolve())
        if library not in torch.ops.loaded_libraries:
            torch.ops.load_library(library)

    return torch.jit.load(str(path))
