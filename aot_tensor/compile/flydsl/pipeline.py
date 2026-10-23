# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# One FlyDSL op ships as one self-contained .so:
#
#   torch.ops.aot_tensor.{op}(Tensor, int, ...)
#        |
#   {op}_torch_op.cpp     generate_torch_op_content(): selects the variant
#        |                matching the call, packs its ABI slots into
#        |                void* args[]; also defines the mgpu* runtime
#        |                functions the launchers call
#        |  {symbol}___mlir_{entry}(void** args)
#   {symbol}.o            one exported FlyDSL host launcher per variant,
#        |                symbol = {op}__{spec_hash[:8]} (grid math +
#        |                launches + embedded gpu.binary)
#        |
#   libamdhip64.so        shared with PyTorch
#
# Unlike CuTeDSL there is no sidecar: the .so links only the process's single
# libamdhip64, so there is no second runtime to interpose.

import logging
import os

from aot_tensor.arg_specs.common import RawSpec, resolve_runtime_args
from aot_tensor.compile.adapter_base import hash_spec
from aot_tensor.compile.flydsl.aot import export_launcher
from aot_tensor.compile.flydsl.args import launcher_kwargs
from aot_tensor.compile.flydsl.codegen import FlyVariant, generate_torch_op_content
from aot_tensor.compile.spec_selector import distinct_variants
from aot_tensor.types import FlyAOT

logger: logging.Logger = logging.getLogger(__name__)


def compile_flydsl_to_cpp(
    op: FlyAOT,
    specs: list[RawSpec],
    install_dir: str,
) -> list[str]:
    """Compile one launcher object per distinct specialization plus the torch
    op dispatching between them; returns the object paths to link."""
    os.makedirs(install_dir, exist_ok=True)
    keyed = [(spec, hash_spec(spec.hash_key), spec.hash_key) for spec in specs]
    variants: list[FlyVariant] = []
    for spec, variant in distinct_variants(op.name, keyed):
        values = resolve_runtime_args(
            op.arg_specs, op.name, *spec.runtime_args, **spec.runtime_kwargs
        )
        logger.info("[AOTT][FlyDSL]: compiling %s (%s)", op.name, variant.description)
        exported = export_launcher(
            op.jit_fn,
            launcher_kwargs(op, values),
            os.path.join(install_dir, f"{variant.symbol}.o"),
            variant.symbol,
        )
        variants.append(FlyVariant(variant, exported))

    with open(os.path.join(install_dir, f"{op.name}_torch_op.cpp"), "w") as fp:
        fp.write(generate_torch_op_content(op.name, op.arg_specs, variants, op.mutated))
    return [fly.exported.object_path for fly in variants]
