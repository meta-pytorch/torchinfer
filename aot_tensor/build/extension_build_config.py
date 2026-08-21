# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-strict

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtensionBuildConfig:
    """Optional environment overrides for extension compilation."""

    compiler_path: str | None = None
    gpu_toolkit_path: str | None = None
    extra_torch_include_dirs: tuple[str, ...] = ()
    extra_gpu_library_dirs: tuple[str, ...] = ()
