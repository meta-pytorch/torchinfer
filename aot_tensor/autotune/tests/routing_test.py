# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest

from aot_tensor.autotune.routing import log2_bucket, RoutingTable
from parameterized import parameterized


class Log2BucketTest(unittest.TestCase):
    @parameterized.expand(
        [(0, 0), (1, 0), (2, 1), (3, 2), (128, 7), (129, 8), (700, 10), (2**40, 40)]
    )
    def test_log2_bucket(self, value: int, expected: int) -> None:
        self.assertEqual(log2_bucket(value), expected)


class RoutingTableTest(unittest.TestCase):
    # Buckets 128 (log2 7) and 8192 (log2 13) on M.
    ONE_KEY: RoutingTable[str] = RoutingTable.from_buckets({(128,): "a", (8192,): "b"})

    def test_from_buckets_keeps_log2_in_order(self) -> None:
        self.assertEqual(self.ONE_KEY.entries, {(7,): "a", (13,): "b"})

    @parameterized.expand(
        [
            ("exact_hit", 100, "a"),
            ("exact_hit_second", 8192, "b"),
            ("nearest_below", 1, "a"),
            ("nearest_above", 10**6, "b"),
            # Bucket 10 is 3 from both; the earlier entry wins.
            ("tie_takes_earlier", 700, "a"),
            ("past_the_tie", 1500, "b"),
        ]
    )
    def test_select_one_key(self, _name: str, m: int, expected: str) -> None:
        self.assertEqual(self.ONE_KEY.select([m]), expected)

    def test_select_sums_squared_distance_over_keys(self) -> None:
        table = RoutingTable.from_buckets({(128, 512): "a", (8192, 64): "b"})
        # (log2 10, log2 9): 3^2 + 0^2 = 9 to "a", 3^2 + 3^2 = 18 to "b".
        self.assertEqual(table.select([700, 512]), "a")
        # (13, 6) hits "b" exactly.
        self.assertEqual(table.select([8000, 64]), "b")
        with self.assertRaisesRegex(ValueError, "has 1 key values, expected 2"):
            table.select([700])

    def test_filter_keeps_table_order(self) -> None:
        table = RoutingTable.from_buckets({(64,): "x", (512,): "y", (4096,): "x"})
        kept = table.filter(lambda v: v == "x")
        self.assertEqual(kept.entries, {(6,): "x", (12,): "x"})
        with self.assertRaisesRegex(ValueError, "no entries"):
            table.filter(lambda _: False).select([1])

    @parameterized.expand(
        [
            ("empty", {}, "is empty"),
            ("not_power_of_2", {(100,): "a"}, "100 in \\(100,\\) is not a power of 2"),
            ("zero", {(0,): "a"}, "is not a power of 2"),
            ("ragged", {(64,): "a", (64, 64): "b"}, "has 2 buckets, expected 1"),
        ]
    )
    def test_from_buckets_rejects(
        self, _name: str, table: dict[tuple[int, ...], str], message: str
    ) -> None:
        with self.assertRaisesRegex(ValueError, message):
            RoutingTable.from_buckets(table)
