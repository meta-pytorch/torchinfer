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
    ``compact_mode`` needs an explicit ``stride_order``.

    Vectorized kernels with 128-bit ``cp.async`` need ``compact_mode`` +
    ``divisibility``: without them the compiler can't prove 16B stride alignment
    and IR verification rejects the copy.
    """

    compact_mode: Optional[int] = None
    stride_order: Optional[Tuple[int, ...]] = None
    divisibility: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.divisibility < 1:
            raise RuntimeError(
                f"CuTeTensorArg {self.name}: divisibility must be >= 1; "
                f"got {self.divisibility}."
            )
        if self.compact_mode is None:
            return
        # The generated op checks the strides of the dims outside compact_mode,
        # so it must know the order; it can't see what CuTe would deduce from the
        # sample.
        order = self.stride_order
        if (
            order is None
            or sorted(order) != list(range(len(order)))
            or self.compact_mode not in order
        ):
            raise RuntimeError(
                f"CuTeTensorArg {self.name}: compact_mode needs a stride_order "
                f"that is a permutation of the dims and contains it; got {order}."
            )


CuTeArgSpec = Union[CuTeTensorArg, ScalarArg, ConstexprArg]
