# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import os
import unittest
from typing import Callable
from unittest.mock import MagicMock

from aot_tensor.compile.adapter_base import DslSpecStore
from aot_tensor.compile.compile_state import (
    AOTTCompileState,
    assert_aott_compile_session_completed,
    get_aott_compile_path,
    get_aott_compile_state,
    is_aott_compile_enabled,
)
from aot_tensor.types import TritonAOT
from parameterized import parameterized


class _ResetStateTest(unittest.TestCase):
    """Base for tests that mutate the AOTTCompileState singleton: reset it before
    and after each test so cases do not leak state into one another."""

    def setUp(self) -> None:
        get_aott_compile_state().reset()

    def tearDown(self) -> None:
        get_aott_compile_state().reset()


class AOTTCompileStateTest(_ResetStateTest):
    @parameterized.expand(
        [
            ("get_instance", AOTTCompileState.get_instance),
            ("constructor", AOTTCompileState),
            ("module_accessor", get_aott_compile_state),
        ]
    )
    def test_singleton_returns_same_instance(
        self, _name: str, make: Callable[[], AOTTCompileState]
    ) -> None:
        """get_instance() and the constructor both return the one shared singleton."""
        self.assertIs(make(), make())

    def test_reset_clears_kernels_keeps_stores(self) -> None:
        """reset() clears each store's collected kernels and disables collection,
        but keeps the stores (their reused adapter + process-lifetime eager cache).
        """
        state = AOTTCompileState.get_instance()

        # Simulate an active compile: a registered collector + a collected kernel.
        TritonAOT.set_spec_collector(lambda *args, **kwargs: None)
        store = DslSpecStore(dsl=MagicMock())
        store.kernels[MagicMock()] = MagicMock()
        state.dsl_state["triton"] = store

        state.reset()

        # Store kept (adapter + eager cache survive); only its kernels are cleared.
        self.assertIn("triton", state.dsl_state)
        self.assertEqual(store.kernels, {})
        self.assertFalse(is_aott_compile_enabled())

    def test_compile_path_created_lazily(self) -> None:
        """compile_path stays None until get_aott_compile_path() first needs it,
        then it is created once and reused."""
        state = get_aott_compile_state()
        self.assertIsNone(state.compile_path)

        path = get_aott_compile_path()
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(state.compile_path, path)
        self.assertEqual(get_aott_compile_path(), path)


class AssertAottCompileSessionCompletedTest(_ResetStateTest):
    def test_raises_when_no_session_has_run(self) -> None:
        with self.assertRaisesRegex(
            AssertionError, "AOTTCompileSession has not completed"
        ):
            assert_aott_compile_session_completed()

    def test_passes_after_a_completed_session(self) -> None:
        state = get_aott_compile_state()
        get_aott_compile_path()
        state.session_completed = True

        assert_aott_compile_session_completed()

        # The flag alone is the guard's condition; compile_path being set is the
        # invariant that lets callers treat get_aott_compile_path() as a read.
        self.assertIsNotNone(state.compile_path)

    def test_reset_clears_session_completed(self) -> None:
        """A new session must not inherit the previous one's completion."""
        state = get_aott_compile_state()
        get_aott_compile_path()
        state.session_completed = True

        state.reset()

        self.assertFalse(state.session_completed)
        with self.assertRaisesRegex(
            AssertionError, "AOTTCompileSession has not completed"
        ):
            assert_aott_compile_session_completed()
