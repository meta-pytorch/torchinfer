# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# One FlyDSL op ships as one self-contained .so:
#
#   torch.ops.aot_tensor.{op}(Tensor, int, ...)
#        |
#   {op}_torch_op.cpp     generate_torch_op_content(): selects the variant
#        |                matching the call, packs its ABI slots into
#        |                void* args[]
#        |  aott{variant}(void** args), aott{variant}__module_{init,load}
#   {variant}.o           one FlyDSL export_to_c object per variant,
#        |                variant = {op}__{spec_hash[:8]} (grid math +
#        |                launches + embedded code object + AOT runtime)
#        |
#   libamdhip64.so        shared with PyTorch
#
# Unlike CuTeDSL there is no sidecar: the .so links only the process's single
# libamdhip64, so there is no second runtime to interpose.

import logging
import os

from aot_tensor.arg_specs.common import RawSpec, resolve_runtime_args
from aot_tensor.compile.adapter_base import hash_spec
from aot_tensor.compile.flydsl.abi import ExportedLauncher
from aot_tensor.compile.flydsl.aot import export_launcher
from aot_tensor.compile.flydsl.codegen import FlyVariant, generate_torch_op_content
from aot_tensor.compile.spec_selector import (
    distinct_variants,
    expand_autotuned,
    key_exprs,
    variant_routes,
)
from aot_tensor.constants import FLYDSL_SYMBOL_PREFIX
from aot_tensor.types import FlyAOT

logger: logging.Logger = logging.getLogger(__name__)


def compile_flydsl_to_cpp(
    op: FlyAOT,
    specs: list[RawSpec],
    install_dir: str,
) -> list[ExportedLauncher]:
    """Export one launcher object per distinct specialization and generate the
    torch op dispatching between them; returns the objects to link."""
    os.makedirs(install_dir, exist_ok=True)
    if op.tuner is not None:
        specs = expand_autotuned(specs, op.arg_specs, op.name, op.tuner)
    keyed = [(spec, hash_spec(spec.hash_key), spec.hash_key) for spec in specs]
    to_compile = distinct_variants(op.name, keyed)
    variants: list[FlyVariant] = []
    for spec, variant in to_compile:
        values = resolve_runtime_args(
            op.arg_specs, op.name, *spec.runtime_args, **spec.runtime_kwargs
        )
        logger.info("[AOTT][FlyDSL]: compiling %s (%s)", op.name, variant.description)
        exported = export_launcher(
            op=op,
            values=values,
            out_dir=install_dir,
            file_name=variant.symbol,
            symbol=f"{FLYDSL_SYMBOL_PREFIX}{variant.symbol}",
        )
        variants.append(FlyVariant(variant, exported))

    with open(os.path.join(install_dir, f"{op.name}_torch_op.cpp"), "w") as fp:
        routes = variant_routes(
            op.name, [(s.hash_key, v) for s, v in to_compile], op.tuner
        )
        # Autotuned constexprs are baked into the variants, not op params.
        autotuned = frozenset() if op.tuner is None else op.tuner.constexpr_names
        op_specs = [spec for spec in op.arg_specs if spec.name not in autotuned]
        fp.write(
            generate_torch_op_content(
                op.name,
                op_specs,
                variants,
                op.mutated,
                routes,
                key_exprs(op.tuner),
            )
        )
    return [fly.exported for fly in variants]
