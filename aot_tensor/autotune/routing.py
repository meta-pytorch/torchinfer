# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

# How an autotuned op picks a config from a call's key values. Eager picks its
# config and the generated C++ selector picks its variant with this one table and
# rule, so both pick the same.
#
# A key value's bucket is the next power of 2, kept as its log2: M=700 -> bucket
# 1024 -> 10. A call takes the table entry nearest to its buckets by squared
# log2 distance summed over the keys (an exact hit is distance 0), and the
# earlier entry on a tie.

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

# What an entry routes to: a config in eager, a variant symbol in the selector.
Choice = TypeVar("Choice")


def log2_bucket(value: int) -> int:
    """log2 of the power-of-2 bucket holding ``value``; values <= 1 share 0."""
    return 0 if value <= 1 else (value - 1).bit_length()


@dataclass(frozen=True)
class RoutingTable(Generic[Choice]):
    # log2 bucket per key -> value, in tie order.
    entries: dict[tuple[int, ...], Choice]

    @staticmethod
    def from_buckets(table: Mapping[tuple[int, ...], Choice]) -> RoutingTable[Choice]:
        """``table`` maps a power-of-2 bucket per key to a value."""
        if not table:
            raise ValueError("autotune routing table is empty")
        num_keys = len(next(iter(table)))
        for buckets in table:
            if len(buckets) != num_keys:
                raise ValueError(
                    f"autotune entry {buckets} has {len(buckets)} buckets, "
                    f"expected {num_keys}"
                )
            for size in buckets:
                if size < 1 or size & (size - 1):
                    raise ValueError(
                        f"autotune bucket {size} in {buckets} is not a power of 2"
                    )
        # Distinct powers of 2 have distinct log2s, so no two entries collide.
        return RoutingTable(
            {tuple(log2_bucket(s) for s in b): v for b, v in table.items()}
        )

    def filter(self, keep: Callable[[Choice], bool]) -> RoutingTable[Choice]:
        """The entries whose value passes ``keep``, in table order."""
        return RoutingTable({p: v for p, v in self.entries.items() if keep(v)})

    def select(self, key_values: Sequence[int]) -> Choice:
        """The value for a call with these key values."""
        if not self.entries:
            raise ValueError("autotune routing table has no entries")
        query = [log2_bucket(v) for v in key_values]
        num_keys = len(next(iter(self.entries)))
        if len(query) != num_keys:
            raise ValueError(
                f"autotune call has {len(query)} key values, expected {num_keys}"
            )

        def distance(point: tuple[int, ...]) -> int:
            return sum((q - p) ** 2 for q, p in zip(query, point))

        # min keeps the first of equal distances, i.e. the earlier entry.
        return self.entries[min(self.entries, key=distance)]
