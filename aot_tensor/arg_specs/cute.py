# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from aot_tensor.arg_specs.common import ConstexprArg, ScalarArg, TensorArg


@dataclass(frozen=True, kw_only=True)
class CuTeTensorArg(TensorArg):
    """Runtime tensor argument for CuTeDSL AOT.

    ``leading_dim`` / ``compact_mode`` / ``stride_order`` / ``divisibility`` are
    layout hints for the ``cute.Tensor`` conversion, like a hand-written
    ``from_dlpack(...).mark_layout_dynamic(...).mark_compact_shape_dynamic(...)``.
    Defaults: auto-detected unit-stride leading dim, no compact/divisibility hint.

    Vectorized kernels with 128-bit ``cp.async`` need ``compact_mode`` +
    ``divisibility``: without them the compiler can't prove 16B stride alignment
    and IR verification rejects the copy.
    """

    alignment: int = 16
    compact_mode: Optional[int] = None
    stride_order: Optional[Tuple[int, ...]] = None
    divisibility: int = 1


CuTeArgSpec = Union[CuTeTensorArg, ScalarArg, ConstexprArg]
