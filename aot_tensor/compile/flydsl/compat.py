# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# FlyDSL has no public AOT API, so AOT-T reads a few private attributes. Every
# read goes through ``private_attr`` so a FlyDSL upgrade that drops one fails
# loudly.

from typing import Any

import flydsl

_VALIDATED_FLYDSL = "0.3.2"


def private_attr(owner: Any, name: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError as e:
        raise RuntimeError(
            f"FlyDSL AOT was validated with FlyDSL {_VALIDATED_FLYDSL} and needs "
            f"private API {name!r}, which FlyDSL {flydsl.__version__} does not provide"
        ) from e
