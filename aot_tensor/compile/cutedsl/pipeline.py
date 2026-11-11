# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# Compiles one CuTeDSL op: cute.compile + export_to_c per distinct
# specialization, then writes the two C++ shims (see codegen.py). The eager path
# is in eager.py.

import logging
import os
from typing import Any

import torch
from aot_tensor.arg_specs.common import RawSpec, resolve_runtime_args
from aot_tensor.compile.adapter_base import hash_spec
from aot_tensor.compile.cutedsl.codegen import (
    build_cute_call_args,
    CuTeVariant,
    generate_sidecar_entry_content,
    generate_torch_op_content,
    make_codegens,
    make_variant,
    validate_cutedsl_op,
)
from aot_tensor.compile.spec_selector import distinct_variants
from aot_tensor.types import CuTeAOT

logger: logging.Logger = logging.getLogger(__name__)


def compile_cutedsl_to_cpp(
    op: CuTeAOT,
    specs: list[RawSpec],
    install_dir: str,
) -> list[str]:
    """Export one kernel object per distinct specialization, plus the sidecar
    entry and the torch op dispatching between them; returns the object paths
    to link into the sidecar."""
    # Imported only in the CuTe compile path so regular Triton AOT users do not
    # need Cutlass/CuTeDSL at import time.
    import cutlass.cute as cute  # @manual=fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl
    import cutlass.cute.export  # @manual=fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl  # noqa: F401

    if torch.version.hip is not None:
        raise NotImplementedError("CuTeDSL AOT is only supported on NVIDIA CUDA.")
    # TODO: share the torch-op arg codegen (scalar/constexpr tables, schema,
    # C++ params) with FlyDSL.

    codegens = make_codegens(op.arg_specs)
    validate_cutedsl_op(codegens, op.name)
    os.makedirs(install_dir, exist_ok=True)

    keyed = [(spec, hash_spec(spec.hash_key), spec.hash_key) for spec in specs]
    variants: list[CuTeVariant] = []
    for spec, variant in distinct_variants(op.name, keyed):
        # cute.compile args: per-arg values (dlpack tensors with dynamic layout,
        # CuTeDSL scalars, passthrough constexprs) plus the current stream.
        values = resolve_runtime_args(
            op.arg_specs, op.name, *spec.runtime_args, **spec.runtime_kwargs
        )
        cute_args = build_cute_call_args(op.arg_specs, values)

        logger.info("[AOTT][CuTeDSL]: compiling %s (%s)", op.name, variant.description)
        compiled: Any = cute.compile(op.jit_fn, *cute_args)
        compiled.export_to_c(install_dir, variant.symbol)

        with open(os.path.join(install_dir, f"{variant.symbol}.h")) as fp:
            variants.append(make_variant(codegens, variant, fp.read()))

    with open(os.path.join(install_dir, f"{op.name}_entry.cpp"), "w") as fp:
        fp.write(generate_sidecar_entry_content(op.name, codegens, variants))

    sidecar_so_name = f"{op.name.lstrip('_')}_cutedsl_impl.so"
    with open(os.path.join(install_dir, f"{op.name}_torch_op.cpp"), "w") as fp:
        fp.write(
            generate_torch_op_content(op.name, codegens, variants, sidecar_so_name)
        )
    return [os.path.join(install_dir, f"{cute.variant.symbol}.o") for cute in variants]
