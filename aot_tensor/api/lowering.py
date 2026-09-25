# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.build.extension_builder_base import TORCH_TARGET_VERSION
from aot_tensor.compile.adapter_base import (
    ArtifactKind,
    CompileContext,
    DslCompileConfig,
)
from aot_tensor.compile.aott_compile import AOTTCompileSession
from aot_tensor.compile.triton.adapter import TritonBuildMetadata, TritonCompileConfig
from aot_tensor.constants import TRITON
from aot_tensor.transform.transform_kernels import (
    assert_weight_signature_matches,
    get_weight_signature,
    transform_kernels,
)
from aot_tensor.types import CuTeAOT
from torch.fx import GraphModule
from triton.backends.compiler import GPUTarget  # @manual=//triton:triton


@dataclass(frozen=True)
class LoweringOptions:
    dsl_configs: tuple[DslCompileConfig, ...] = field(
        default_factory=lambda: (TritonCompileConfig(),)
    )
    extension_build_config: ExtensionBuildConfig | None = None
    validate: bool = True


@dataclass(frozen=True)
class LoweringArtifact:
    """One compiler output, with a path relative to ``work_dir``."""

    dsl_name: str
    kind: ArtifactKind
    path: Path


@dataclass
class LoweringResult:
    module: GraphModule
    work_dir: Path
    artifacts: tuple[LoweringArtifact, ...]
    gpu_target: str
    torch_target_version: str
    op_namespace: str


def _validate_dsl_configs(dsl_configs: tuple[DslCompileConfig, ...]) -> None:
    if len(dsl_configs) != 1 or type(dsl_configs[0]) is not TritonCompileConfig:
        configured = ", ".join(type(config).__name__ for config in dsl_configs)
        raise ValueError(
            "lower_model() v0 requires exactly one TritonCompileConfig; "
            f"got: {configured or 'none'}"
        )


def _prepare_work_dir(work_dir: os.PathLike[str] | str) -> Path:
    path = Path(work_dir).resolve()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"AOT Tensor work_dir is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"AOT Tensor work_dir must be empty: {path}")
    else:
        path.mkdir(parents=True)
    return path


def _reject_cutedsl(
    _marker: CuTeAOT,
    *_args: Any,
    **_kwargs: Any,
) -> None:
    raise RuntimeError(
        "lower_model() v0 supports Triton kernels only; "
        "CuTeDSL support is not yet part of the public API"
    )


def _run_forward_passes(
    module: torch.nn.Module,
    inputs_list: Sequence[Sequence[Any]],
) -> None:
    for inputs in inputs_list:
        module(*inputs)


def _format_gpu_target(target: GPUTarget) -> str:
    architecture = str(target.arch)
    if target.backend == "cuda" and isinstance(target.arch, int):
        architecture = f"sm{target.arch}"
    return f"{target.backend}:{architecture}"


def _relative_artifacts(
    context: CompileContext,
    work_dir: Path,
    dsl_name: str,
) -> tuple[LoweringArtifact, ...]:
    result: list[LoweringArtifact] = []
    for artifact in context.artifacts:
        path = Path(artifact.path).resolve()
        try:
            relative_path = path.relative_to(work_dir)
        except ValueError as error:
            raise RuntimeError(f"Compiled artifact escapes work_dir: {path}") from error
        if not path.is_file():
            raise RuntimeError(f"Compiled artifact does not exist: {path}")
        result.append(
            LoweringArtifact(
                dsl_name=dsl_name,
                kind=artifact.kind,
                path=relative_path,
            )
        )
    return tuple(result)


def lower_model(
    module: GraphModule,
    example_inputs: Sequence[Sequence[Any]],
    *,
    work_dir: os.PathLike[str] | str,
    options: LoweringOptions | None = None,
) -> LoweringResult:
    """Compile Triton kernels and lower their FX call sites in ``module``.

    ``module`` is consumed and mutated in place. ``work_dir`` must not contain
    existing files and remains caller-owned after this function returns.
    Concurrent invocations in one process are unsupported.
    """
    lowering_options = options or LoweringOptions()
    _validate_dsl_configs(lowering_options.dsl_configs)
    resolved_work_dir = _prepare_work_dir(work_dir)
    weight_signature = get_weight_signature(module)

    session = AOTTCompileSession(
        dsl_config=lowering_options.dsl_configs,
        extension_build_config=lowering_options.extension_build_config,
        compile_path=str(resolved_work_dir),
        spec_collector_overrides={CuTeAOT: _reject_cutedsl},
    )
    with session:
        _run_forward_passes(module, example_inputs)

    lowered = transform_kernels(module)
    assert_weight_signature_matches(weight_signature, lowered, "transform_kernels")
    if lowering_options.validate:
        _run_forward_passes(lowered, example_inputs)

    context = session.compile_context
    metadata = context.dsl_metadata.get(TRITON)
    if not isinstance(metadata, TritonBuildMetadata):
        raise RuntimeError("lower_model() collected no Triton kernels")
    artifacts = _relative_artifacts(context, resolved_work_dir, TRITON)
    if not any(artifact.kind == "shared_library" for artifact in artifacts):
        raise RuntimeError(
            f"lower_model() generated no shared libraries in {resolved_work_dir}"
        )

    return LoweringResult(
        module=lowered,
        work_dir=resolved_work_dir,
        artifacts=artifacts,
        gpu_target=_format_gpu_target(metadata.gpu_target),
        torch_target_version=TORCH_TARGET_VERSION,
        op_namespace=metadata.op_namespace,
    )
