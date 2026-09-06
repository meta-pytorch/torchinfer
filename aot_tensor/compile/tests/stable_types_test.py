# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""``SCALAR_TYPES`` must be keyed by what Triton's mangler actually emits.

The table is a lookup on ``mangle_type(tensor)`` (``triton/adapter.py``), so a
key Triton never produces is dead: the dtype falls through to
``unsupported tensor type for {arg}``. That is not a loud failure -- it only
fires when some kernel first passes that dtype, and until then the table reads
as though the dtype were supported.

``torch.bool`` sat in exactly that state. The table declared
``"*i1": ScalarType::Bool``, Triton mangles bool to ``*u1``, and no test
exercised bool at all, so every bool pointer was rejected while the table
claimed otherwise. Callers worked around it by copying to ``uint8`` -- a full
pass over the tensor to relabel bytes that were already correct.

So these tests derive the expected keys from ``mangle_type`` rather than
restating them. A test that hardcoded ``"*u1"`` would go stale the same way the
table did if Triton's spelling ever changes.
"""

import unittest

import torch
from aot_tensor.compile.stable_types import SCALAR_TYPES, TORCH_DTYPE_TO_STABLE
from parameterized import parameterized
from triton.runtime.jit import mangle_type  # @manual

# Every dtype AOTT claims to render, as the caller spells it. Keys are derived
# from the mangler below, so this list -- not the table -- is the contract.
_SUPPORTED_DTYPES: list[torch.dtype] = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
]


class StableTypesTest(unittest.TestCase):
    def test_every_supported_dtype_has_a_mangled_key(self) -> None:
        """The regression this file exists for.

        Fails for any dtype whose mangled spelling is missing from the table,
        which is precisely how bool was broken.
        """
        missing = {
            str(dtype): mangle_type(torch.zeros(8, dtype=dtype))
            for dtype in _SUPPORTED_DTYPES
            if mangle_type(torch.zeros(8, dtype=dtype)) not in SCALAR_TYPES
        }
        self.assertEqual(
            missing,
            {},
            "SCALAR_TYPES is keyed on mangle_type output; these dtypes mangle "
            "to keys the table does not have, so AOTT rejects them",
        )

    def test_bool_mangles_to_u1(self) -> None:
        """Pins the specific spelling, since the table carried the wrong one."""
        self.assertEqual(mangle_type(torch.zeros(8, dtype=torch.bool)), "*u1")
        self.assertEqual(SCALAR_TYPES["*u1"], "torch::headeronly::ScalarType::Bool")

    @parameterized.expand([(str(dtype), dtype) for dtype in _SUPPORTED_DTYPES])
    def test_mangled_keys_agree_with_the_torch_dtype_table(
        self, _name: str, dtype: torch.dtype
    ) -> None:
        """The two tables must not disagree about the same dtype.

        ``TORCH_DTYPE_TO_STABLE`` (CuteDSL, keyed on torch's dtype string) and
        ``SCALAR_TYPES`` (Triton, keyed on the mangled string) describe the same
        mapping by different routes. bool was correct in the first and wrong in
        the second, which is what let the gap survive.
        """
        key = mangle_type(torch.zeros(8, dtype=dtype))
        self.assertEqual(
            SCALAR_TYPES[key],
            TORCH_DTYPE_TO_STABLE[str(dtype)],
            f"{dtype} maps to different ScalarTypes in the two tables",
        )

    def test_table_renders_only_reachable_keys(self) -> None:
        """Every key should be something the mangler can emit.

        ``*i1`` is grandfathered: it is unreachable but retained for
        out-of-tree callers. Any *new* unreachable key is a typo of the kind
        that broke bool, so this fails rather than letting it accumulate.
        """
        reachable = {
            mangle_type(torch.zeros(8, dtype=dtype)) for dtype in _SUPPORTED_DTYPES
        }
        # fp8 variants are real Triton dtypes with no plain torch.zeros path
        # here; exempt them rather than assert against dtypes this test cannot
        # construct portably.
        exempt = {"*i1", "*fp8e4nv", "*fp8e4b8"}
        unreachable = sorted(set(SCALAR_TYPES) - reachable - exempt)
        self.assertEqual(
            unreachable,
            [],
            "these keys cannot be produced by mangle_type, so they are dead "
            "entries that make the table look more capable than it is",
        )
