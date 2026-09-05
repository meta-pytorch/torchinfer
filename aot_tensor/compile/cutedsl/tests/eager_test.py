# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import torch
from aot_tensor.compile.cutedsl.eager import _compile_cache_key
from aot_tensor.cute_specs import (
    CuTeArgSpec,
    CuTeConstexprArg,
    CuTeTensorArg,
    resolve_runtime_args,
)


def _toy_kernel() -> None:
    pass


class CompileCacheKeyTest(unittest.TestCase):
    """Cache-key-level checks for the eager compile cache. Tensor-key details
    (shape-independence, dtype, layout) are covered by
    ``cute_specs_test.TensorHashKeyTest``; here we cover the cache-key aggregation
    -- rank, constexpr, and alignment."""

    SPECS: list[CuTeArgSpec] = [CuTeTensorArg("x"), CuTeConstexprArg("flag", bool)]

    def _key(self, x: torch.Tensor, flag: bool) -> str:
        values = resolve_runtime_args(self.SPECS, "op", x, flag)
        return _compile_cache_key(_toy_kernel, self.SPECS, "op", values)

    def test_rank_distinguishes(self) -> None:
        self.assertNotEqual(
            self._key(torch.empty(4, 8), True),
            self._key(torch.empty(4, 8, 2), True),
        )

    def test_constexpr_distinguishes(self) -> None:
        self.assertNotEqual(
            self._key(torch.empty(4, 8), True),
            self._key(torch.empty(4, 8), False),
        )

    def test_alignment_distinguishes(self) -> None:
        # Drop one float32 (4 bytes) off the base so data_ptr is no longer
        # 16-byte aligned; same dtype/rank but a different alignment bit.
        unaligned = torch.empty(17)[1:]
        self.assertNotEqual(unaligned.data_ptr() % 16, 0)
        self.assertNotEqual(
            self._key(torch.empty(16), True), self._key(unaligned, True)
        )


if __name__ == "__main__":
    unittest.main()
