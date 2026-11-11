# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from aot_tensor.arg_specs.common import ConstexprArg, ScalarArg, TensorArg


@dataclass(frozen=True, kw_only=True)
class FlyTensorArg(TensorArg):
    """Runtime tensor argument for FlyDSL AOT.

    Bound as ``flyc.from_torch_tensor(t, assumed_align=alignment)
    .mark_layout_dynamic(leading_dim=..., divisibility=...)``, so one compiled
    launcher serves every shape of a given dtype/rank/unit-stride dim.
    """

    divisibility: int = 1
    alignment: int = 16


FlyArgSpec = Union[FlyTensorArg, ScalarArg, ConstexprArg]
