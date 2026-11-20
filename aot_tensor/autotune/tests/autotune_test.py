# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest
from typing import Any

import torch
from aot_tensor.arg_specs.common import ArgSpec, ConstexprArg, ScalarArg, TensorArg
from aot_tensor.autotune.autotune import Tuner
from parameterized import parameterized


_A = {"BM": 1}
_B = {"BM": 2}
_AUTOTUNE_SPECS: list[ArgSpec] = [
    TensorArg("x"),
    ScalarArg("n", "i32"),
    ConstexprArg("NEG", bool),
    ConstexprArg("BM", int),
]


def _tuner(**overrides: Any) -> Tuner:
    # Buckets 16 (log2 4) and 1024 (log2 10) on x's dim 0.
    kwargs: dict[str, Any] = {"key": [("x", 0)], "tuned": {16: _A, 1024: _B}}
    kwargs.update(overrides)
    return Tuner(**kwargs)


def _autotuned(tuner: Tuner, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """The call's values with the config the tuner picks, in route group "g"."""
    values = tuner.bind(_AUTOTUNE_SPECS, "op", *args, **kwargs)
    return values | tuner.best_config("g", _AUTOTUNE_SPECS, values, "op")


class TunerTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("exact_hit", 16, 1),
            ("nearest_above", 4096, 2),
            # Bucket 128 is 3 from both entries; the earlier one wins.
            ("tie_takes_earlier", 100, 1),
        ]
    )
    def test_fills_the_tuned_config(self, _name: str, m: int, bm: int) -> None:
        x = torch.empty(m)
        values = _autotuned(_tuner(), x, m, NEG=False)
        self.assertEqual(values, {"x": x, "n": m, "NEG": False, "BM": bm})

    def test_keys_on_an_int_arg(self) -> None:
        values = _autotuned(_tuner(key=["n"]), torch.empty(1), 4096, NEG=False)
        self.assertEqual(values["BM"], 2)

    def test_skips_configs_invalid_for_the_call(self) -> None:
        tuner = _tuner(valid=lambda cfg, c: not (c["NEG"] and cfg["BM"] == 2))
        x = torch.empty(4096)
        self.assertEqual(_autotuned(tuner, x, 1, True)["BM"], 1)
        self.assertEqual(_autotuned(tuner, x, 1, False)["BM"], 2)

    def test_tuple_and_bare_int_buckets_agree(self) -> None:
        self.assertEqual(
            _tuner(tuned={(16,): _A, (1024,): _B}).routing, _tuner().routing
        )
        self.assertEqual(_tuner().routing.entries, {(4,): _A, (10,): _B})

    @parameterized.expand(
        [
            ("passes_an_autotuned_arg", {}, {"BM": 1}, r"\['BM'\] are autotuned"),
            (
                "key_dim_out_of_rank",
                {"key": [("x", 1)]},
                {},
                r"key \('x', 1\) needs x of rank > 1, got 1",
            ),
            (
                "no_valid_config",
                {"valid": lambda cfg, c: False},
                {},
                r"no autotune config is valid for \{'NEG': False\}",
            ),
        ]
    )
    def test_rejects(
        self,
        _name: str,
        overrides: dict[str, object],
        extra_kwargs: dict[str, Any],
        regex: str,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, regex):
            _autotuned(
                _tuner(**overrides), torch.empty(4), 4, NEG=False, **extra_kwargs
            )

    @parameterized.expand(
        [
            ("no_key", {"key": []}, "must not be empty"),
            ("no_tuned", {"tuned": {}}, "must not be empty"),
            ("negative_dim", {"key": [("x", -1)]}, r"needs a dim >= 0"),
            ("bucket_per_key", {"tuned": {(16, 16): _A}}, "one bucket per key"),
            ("listed_twice", {"tuned": {16: _A, (16,): _B}}, r"lists \(16,\) twice"),
            ("mixed_names", {"tuned": {16: _A, 1024: {"BN": 1}}}, r"expected \['BM'\]"),
        ]
    )
    def test_construction_rejects(
        self, _name: str, overrides: dict[str, object], regex: str
    ) -> None:
        with self.assertRaisesRegex(ValueError, regex):
            _tuner(**overrides)

    def test_check_accepts_the_specs(self) -> None:
        _tuner().check(_AUTOTUNE_SPECS, "op")

    @parameterized.expand(
        [
            (
                "not_a_constexpr",
                {"tuned": {16: {"n": 1}}},
                "autotuned n must be a ConstexprArg",
            ),
            (
                "config_type",
                {"tuned": {16: {"BM": 1.0}}},
                "sets BM to a <class 'float'>",
            ),
            ("dim_of_a_scalar", {"key": [("n", 0)]}, r"key \('n', 0\)"),
            ("tensor_as_int", {"key": ["x"]}, "key 'x'"),
            ("autotuned_as_key", {"key": ["BM"]}, "key 'BM'"),
            ("bool_constexpr_as_key", {"key": ["NEG"]}, "key 'NEG'"),
        ]
    )
    def test_check_rejects(
        self, _name: str, overrides: dict[str, object], regex: str
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, regex):
            _tuner(**overrides).check(_AUTOTUNE_SPECS, "op")
