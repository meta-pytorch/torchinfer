# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from __future__ import annotations

import logging
from typing import Optional

import torch
from aot_tensor.transform.kernel_wrapper_codegen import kernel_wrapper_codegen
from aot_tensor.transform.replace_kernels import replace_kernels
from torch import package
from torch.fx import GraphModule

logger: logging.Logger = logging.getLogger(__name__)

WeightSignature = dict[str, tuple[tuple[int, ...], torch.dtype]]


def get_weight_signature(
    module: torch.nn.Module,
) -> WeightSignature:
    """Get a signature of all parameters and buffers: {name: (shape, dtype)}."""
    sig = {}
    for name, p in module.named_parameters():
        sig[name] = (tuple(p.shape), p.dtype)
    for name, b in module.named_buffers():
        sig[name] = (tuple(b.shape), b.dtype)
    return sig


def assert_weight_signature_matches(
    before: WeightSignature,
    after: torch.nn.Module,
    context: str,
) -> None:
    """Verify that *after* still has the captured parameter/buffer signature."""
    sig_after = get_weight_signature(after)
    before_names = set(before.keys())
    after_names = set(sig_after.keys())
    if before_names != after_names:
        raise RuntimeError(
            f"[AOTT] {context} changed weight names: "
            f"added={after_names - before_names}, "
            f"removed={before_names - after_names}"
        )
    for name in before_names:
        if before[name] != sig_after[name]:
            raise RuntimeError(
                f"[AOTT] {context} changed weight '{name}': "
                f"before={before[name]}, after={sig_after[name]}"
            )
    logger.info(
        f"[AOTT]: Weight signature check passed for {context}: "
        f"{len(before)} params/buffers unchanged"
    )


def assert_weight_signature_unchanged(
    before: torch.nn.Module,
    after: torch.nn.Module,
    context: str,
) -> None:
    """Verify that parameter/buffer names, shapes, and dtypes are unchanged."""
    assert_weight_signature_matches(get_weight_signature(before), after, context)


def transform_kernels(
    fx_m: GraphModule,
    package_importer: Optional[package.PackageImporter] = None,
) -> GraphModule:
    """Generate AOT wrappers and replace FX graph nodes in one step.

    1. kernel_wrapper_codegen: AST-transforms wrapper functions,
       rewrites kernel[grid](...) -> torch.ops.triton_aot.kernel(...),
       writes {fn}_wrapper.py
    2. replace_kernels: loads wrappers and replaces graph node targets
    """
    kernel_wrapper_codegen(fx_m, package_importer)
    return replace_kernels(fx_m, package_importer=package_importer)
