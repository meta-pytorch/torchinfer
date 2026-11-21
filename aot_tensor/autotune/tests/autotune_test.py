# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest
from collections.abc import Mapping
from typing import Any

import torch
from aot_tensor.arg_specs.common import ArgSpec, ConstexprArg, ScalarArg, TensorArg
from aot_tensor.autotune.autotune import TimeConfigs, Tuner
from parameterized import parameterized


_A = {"BM": 1}
_B = {"BM": 2}
_C = {"BM": 4}
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


def _autotuned(
    tuner: Tuner,
    *args: Any,
    group: str = "g",
    time: TimeConfigs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """The call's values with the config the tuner picks, in route ``group``."""
    values = tuner.bind(_AUTOTUNE_SPECS, "op", *args, **kwargs)
    return values | tuner.best_config(group, _AUTOTUNE_SPECS, values, "op", time)


def _timer(ms_by_bm: dict[int, float], timed_bms: list[int]) -> TimeConfigs:
    """Times each config as ``ms_by_bm[BM]`` and logs its BM to ``timed_bms``."""

    def time_configs(
        values: Mapping[str, Any], configs: list[dict[str, Any]]
    ) -> list[float]:
        timed_bms.extend(config["BM"] for config in configs)
        return [ms_by_bm[config["BM"]] for config in configs]

    return time_configs


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
        self.assertEqual(_tuner(tuned={(16,): _A, (1024,): _B}).tuned, _tuner().tuned)
        tuned = _tuner().tuned
        assert tuned is not None
        self.assertEqual(tuned.entries, {(4,): _A, (10,): _B})

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
            ("neither_tuned_nor_configs", {"tuned": {}}, "give tuned, configs or both"),
            ("negative_dim", {"key": [("x", -1)]}, r"needs a dim >= 0"),
            ("bucket_per_key", {"tuned": {(16, 16): _A}}, "one bucket per key"),
            ("listed_twice", {"tuned": {16: _A, (16,): _B}}, r"lists \(16,\) twice"),
            ("mixed_names", {"tuned": {16: _A, 1024: {"BN": 1}}}, r"expected \['BM'\]"),
            ("configs_name_other", {"configs": [{"BN": 1}]}, r"expected \['BM'\]"),
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
            (
                "configs_type",
                {"tuned": None, "configs": [{"BM": 1.0}]},
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


class TimingTest(unittest.TestCase):
    def test_times_a_new_bucket_once_per_route_group(self) -> None:
        tuner = _tuner(tuned=None, configs=[_A, _B, _C])
        timed_bms: list[int] = []
        time = _timer({1: 3.0, 2: 1.0, 4: 2.0}, timed_bms)
        # 4096 and 3000 share bucket 4096 (log2 12): timed once, BM=2 fastest.
        self.assertEqual(
            _autotuned(tuner, torch.empty(4096), 1, False, time=time)["BM"], 2
        )
        self.assertEqual(
            _autotuned(tuner, torch.empty(3000), 1, False, time=time)["BM"], 2
        )
        self.assertEqual(timed_bms, [1, 2, 4])
        _autotuned(tuner, torch.empty(4096), 1, False, group="h", time=time)
        self.assertEqual(timed_bms, [1, 2, 4] * 2)
        self.assertEqual(tuner.timed, {"g": {(12,): _B}, "h": {(12,): _B}})

    def test_times_only_buckets_tuned_lacks_and_only_valid_configs(self) -> None:
        tuner = _tuner(
            configs=[_B, _C], valid=lambda cfg, c: not (c["NEG"] and cfg["BM"] == 2)
        )
        timed_bms: list[int] = []
        time = _timer({2: 1.0, 4: 2.0}, timed_bms)
        # Bucket 1024 is tuned: BM=2, nothing timed.
        self.assertEqual(
            _autotuned(tuner, torch.empty(1024), 1, False, time=time)["BM"], 2
        )
        self.assertEqual(timed_bms, [])
        # NEG=True: tuned BM=2 is invalid, so bucket 1024 is timed without BM=2.
        self.assertEqual(
            _autotuned(tuner, torch.empty(1024), 1, True, time=time)["BM"], 4
        )
        self.assertEqual(timed_bms, [4])

    def test_no_valid_candidate_falls_back_to_tuned(self) -> None:
        tuner = _tuner(
            configs=[_B], valid=lambda cfg, c: not (c["NEG"] and cfg["BM"] == 2)
        )
        timed_bms: list[int] = []
        # NEG=True: the only candidate is invalid, so the nearest valid tuned entry.
        values = _autotuned(
            tuner, torch.empty(4096), 1, True, time=_timer({2: 1.0}, timed_bms)
        )
        self.assertEqual((values["BM"], timed_bms), (1, []))

    def test_routing_for_appends_the_group_timed_entries(self) -> None:
        tuner = _tuner(configs=[_C])
        _autotuned(
            tuner,
            torch.empty(4096),
            1,
            False,
            time=lambda values, cfgs: [1.0] * len(cfgs),
        )
        self.assertEqual(
            tuner.routing_for("g", {"NEG": False}).entries,
            {(4,): _A, (10,): _B, (12,): _C},
        )
        self.assertEqual(
            tuner.routing_for("h", {"NEG": False}).entries, {(4,): _A, (10,): _B}
        )

    def test_outside_collection_uses_the_routing_table(self) -> None:
        tuner = _tuner(tuned=None, configs=[_A, _B])
        # Nothing timed yet: the first valid config.
        self.assertEqual(_autotuned(tuner, torch.empty(4096), 1, False)["BM"], 1)
        _autotuned(
            tuner, torch.empty(4096), 1, False, time=_timer({1: 2.0, 2: 1.0}, [])
        )
        # Bucket 8192 was not timed: the nearest timed entry, no timing.
        self.assertEqual(_autotuned(tuner, torch.empty(8192), 1, False)["BM"], 2)
