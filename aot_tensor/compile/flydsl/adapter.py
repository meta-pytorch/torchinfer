# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import logging
import os
from typing import Any

from aot_tensor.arg_specs.common import sample_spec_and_hash
from aot_tensor.build.flydsl.extension_builder import build_flydsl_aot_extension
from aot_tensor.compile.adapter_base import AOTTAdapter, CompileContext
from aot_tensor.compile.compile_state import add_spec, get_kernel_specs, register_active
from aot_tensor.compile.flydsl.pipeline import compile_flydsl_to_cpp
from aot_tensor.constants import FLYDSL
from aot_tensor.transform.import_utils import drop_imports
from aot_tensor.transform.marker_call_transform import MarkerCallTransform
from aot_tensor.transform.wrapper_codegen_utils import (
    find_sole_marker_in_globals,
    generate_wrapper_files_skeleton,
)

# FlyAOT.__call__ lazily imports flydsl.eager.
# @dep=:eager
from aot_tensor.types import FlyAOT, get_flydsl_aot_dir_name
from torch import package

logger: logging.Logger = logging.getLogger(__name__)


class FlyAOTOperatorTransform(MarkerCallTransform):
    def __init__(self, kernel: FlyAOT, global_names: set[str]) -> None:
        super().__init__(op_name=kernel.name, global_names=global_names)
        self._kernel: FlyAOT = kernel

    def kernel_dir(self) -> str:
        return get_flydsl_aot_dir_name(self._kernel)


class FlyAdapter(AOTTAdapter[tuple[FlyAOT, set[str]]]):
    """AOT-T DSL integration for FlyDSL launchers (``flydsl_aot`` wrapped)."""

    name: str = FLYDSL

    def compile_and_build(self, ctx: CompileContext) -> None:
        kernel_specs = get_kernel_specs(self.name)
        logger.info(f"[AOTT][FlyDSL]: compiling {len(kernel_specs)} kernels")
        for op, kspec in kernel_specs.items():
            fn_dir = os.path.join(ctx.compile_path, get_flydsl_aot_dir_name(op))
            objects = compile_flydsl_to_cpp(
                op=op, specs=kspec.specs, install_dir=fn_dir
            )
            build_flydsl_aot_extension(
                source_dir=fn_dir,
                kernel_name=op.name,
                output_dir=fn_dir,
                objects=objects,
                build_config=ctx.extension_build_config,
            )

    def find_kernel(self, node_target: Any) -> tuple[FlyAOT, set[str]] | None:
        kernel_specs = get_kernel_specs(self.name)
        return find_sole_marker_in_globals(
            node_target,
            FlyAOT,
            in_specs=lambda var: var in kernel_specs,
            missing_spec_error=lambda var: (
                f"Cannot find FlyAOT kernel {var.name} in collected FlyDSL specs"
            ),
        )

    def generate_wrapper_files(
        self,
        node_target: Any,
        match: tuple[FlyAOT, set[str]],
        compile_path: str,
        package_importer: package.PackageImporter | None,
    ) -> None:
        kernel, global_names = match
        generate_wrapper_files_skeleton(
            node_target,
            kernel_dir=get_flydsl_aot_dir_name(kernel),
            transformer=FlyAOTOperatorTransform(
                kernel=kernel, global_names=global_names
            ),
            compile_path=compile_path,
            package_importer=package_importer,
            import_filter=lambda header: drop_imports(header, ("flydsl",)),
        )


def collect(op: FlyAOT, *args: Any, **kwargs: Any) -> None:
    """Spec collector wired to ``FlyAOT`` by ``aott_compile``."""
    register_active(FlyAdapter)
    spec, hashed = sample_spec_and_hash(
        op.arg_specs, op.name, op.module_basename, *args, **kwargs
    )
    add_spec(FLYDSL, op, spec, hashed)
