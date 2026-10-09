# Copyright (c) Meta Platforms, Inc. and affiliates.

"""A small DLRM-style ranking model that calls two Triton kernels."""

import torch
import torch.nn.functional as F

from .jagged_sum import jagged_sum
from .linear_relu import linear_relu

# lower_model() replaces each launcher's FX node with its compiled kernel. FX
# records a function as one node only when it is called through a module that
# registered it with torch.fx.wrap, so register the imported launchers here.
torch.fx.wrap("jagged_sum")
torch.fx.wrap("linear_relu")

NUM_ITEMS = 1000
NUM_DENSE_FEATURES = 13


class RecsysRanker(torch.nn.Module):
    """Scores a user's interest from their item history and dense features.

    The item histories of a batch are jagged: one flat ``item_ids`` tensor plus
    ``batch + 1`` ``offsets`` into it. ``dense`` is ``[batch, 13]``.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.item_embedding = torch.nn.Embedding(NUM_ITEMS, embedding_dim)
        self.bottom = torch.nn.Linear(NUM_DENSE_FEATURES, hidden_dim)
        self.top = torch.nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(
        self, item_ids: torch.Tensor, offsets: torch.Tensor, dense: torch.Tensor
    ) -> torch.Tensor:
        history = jagged_sum(self.item_embedding(item_ids), offsets)
        dense = linear_relu(dense, self.bottom.weight, self.bottom.bias)
        hidden = linear_relu(
            torch.cat([history, dense], dim=1), self.top.weight, self.top.bias
        )
        return torch.sigmoid(self.head(hidden))

    def reference(
        self, item_ids: torch.Tensor, offsets: torch.Tensor, dense: torch.Tensor
    ) -> torch.Tensor:
        """The same computation in plain PyTorch."""
        history = F.embedding_bag(
            item_ids,
            self.item_embedding.weight,
            offsets,
            mode="sum",
            include_last_offset=True,
        )
        dense = F.relu(self.bottom(dense))
        hidden = F.relu(self.top(torch.cat([history, dense], dim=1)))
        return torch.sigmoid(self.head(hidden))


def make_inputs(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random inputs: histories of 0 to 19 items, and dense features."""
    lengths = torch.randint(0, 20, (batch_size,), device="cuda")
    offsets = torch.zeros(batch_size + 1, dtype=torch.int64, device="cuda")
    offsets[1:] = torch.cumsum(lengths, dim=0)
    item_ids = torch.randint(0, NUM_ITEMS, (int(offsets[-1]),), device="cuda")
    dense = torch.randn(batch_size, NUM_DENSE_FEATURES, device="cuda")
    return item_ids, offsets, dense
