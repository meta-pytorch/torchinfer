# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-strict


def is_amd() -> bool:
    import torch

    return torch.version.hip is not None
