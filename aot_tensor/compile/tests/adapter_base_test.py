# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import importlib
import unittest
from typing import Any
from unittest.mock import MagicMock

from aot_tensor.compile.adapter_base import (
    AOTTAdapter,
    CompileContext,
    DslCompileConfig,
    DslSpecStore,
    hash_spec,
    KernelSpecs,
)
from parameterized import parameterized


class HashSpecTest(unittest.TestCase):
    """hash_spec is deterministic and distinguishes different specs."""

    def test_hash_spec_is_stable_and_distinct(self) -> None:
        spec_a = {"signature": ["*fp32", "i64"]}
        spec_b = {"signature": ["*fp16", "i32"]}

        self.assertEqual(hash_spec(spec_a), hash_spec(dict(spec_a)))
        self.assertNotEqual(hash_spec(spec_a), hash_spec(spec_b))


class KernelSpecsTest(unittest.TestCase):
    """KernelSpecs.add dedups by hash, keeps distinct specs in insertion order."""

    @parameterized.expand(
        [
            # name, (spec, hash) entries to add, expected specs list
            ("same_hash_collapses", [("a", "h1"), ("a", "h1")], ["a"]),
            ("distinct_kept_in_order", [("a", "h1"), ("b", "h2")], ["a", "b"]),
        ]
    )
    def test_add(
        self,
        _name: str,
        entries: list[tuple[str, str]],
        expected_specs: list[str],
    ) -> None:
        ks = KernelSpecs()
        for spec, hashed in entries:
            ks.add(spec, hashed)
        self.assertEqual(ks.specs, expected_specs)
        self.assertEqual(ks.hashes, {h for _, h in entries})


class DslSpecStoreTest(unittest.TestCase):
    """DslSpecStore.add routes each spec into a per-key KernelSpecs."""

    def test_groups_by_key_and_dedups(self) -> None:
        store = DslSpecStore(dsl=MagicMock())
        store.add("k1", "s1", "h1")
        store.add("k1", "s1", "h1")  # duplicate collapses
        store.add("k1", "s2", "h2")
        store.add("k2", "s3", "h3")
        self.assertEqual(store.kernels["k1"].specs, ["s1", "s2"])
        self.assertEqual(store.kernels["k2"].specs, ["s3"])


class _CfgA(DslCompileConfig):
    pass


class _CfgB(DslCompileConfig):
    pass


class CompileContextTest(unittest.TestCase):
    """find_config returns the one config of the queried type, else None."""

    def _ctx(self, configs: list[DslCompileConfig]) -> CompileContext:
        return CompileContext(
            compile_path="/tmp/x",
            import_module=importlib.import_module,
            dsl_config=configs,
        )

    @parameterized.expand(
        [
            ("single_match", [_CfgA], _CfgA, _CfgA),
            ("picks_right_type_among_many", [_CfgA, _CfgB], _CfgB, _CfgB),
            ("absent_returns_none", [_CfgA], _CfgB, None),
        ]
    )
    def test_find_config(
        self,
        _name: str,
        config_types: list[type[DslCompileConfig]],
        query: type[DslCompileConfig],
        expected_type: type[DslCompileConfig] | None,
    ) -> None:
        configs = [make() for make in config_types]
        result = self._ctx(configs).find_config(query)
        if expected_type is None:
            self.assertIsNone(result)
        else:
            self.assertIsInstance(result, expected_type)
            self.assertIn(result, configs)

    def test_rejects_duplicate_config_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate DslCompileConfig"):
            self._ctx([_CfgA(), _CfgA()])


class AOTTAdapterSubclassTest(unittest.TestCase):
    """``AOTTAdapter.__init_subclass__`` enforces a class-level ``name``."""

    def test_subclass_without_name_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "must define a class-level"):

            class _NoName(AOTTAdapter[Any]):
                def compile_and_build(self, ctx: CompileContext) -> None: ...

    def test_subclass_with_name_ok(self) -> None:
        class _WithName(AOTTAdapter[Any]):
            name = "with_name"

            def compile_and_build(self, ctx: CompileContext) -> None: ...

        self.assertEqual(_WithName.name, "with_name")
