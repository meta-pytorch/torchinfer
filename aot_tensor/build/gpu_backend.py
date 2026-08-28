# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


def is_amd() -> bool:
    import torch

    return torch.version.hip is not None
