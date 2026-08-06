# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest
from typing import Any

from aot_tensor.compile.adapter_base import AOTTAdapter, CompileContext, DslSpecStore
from aot_tensor.compile.aott_compile import (
    AOTTCompileSession,
    disable_spec_collection,
    enable_spec_collection,
)
from aot_tensor.compile.compile_state import (
    get_aott_compile_state,
    is_aott_compile_enabled,
)
from aot_tensor.types import AOTTMarker, CuTeAOT, TritonAOT


class SpecCollectionTest(unittest.TestCase):
    """``enable_spec_collection`` / ``disable_spec_collection`` toggle whether AOT
    spec collection is active (collectors registered on the kernel marker types).
    Collection itself is driven by those markers and covered by the per-DSL and
    e2e tests; here we only check the enable/disable lifecycle.
    """

    def setUp(self) -> None:
        get_aott_compile_state().reset()

    def tearDown(self) -> None:
        get_aott_compile_state().reset()
        disable_spec_collection()

    def test_enable_disable_lifecycle(self) -> None:
        self.assertFalse(is_aott_compile_enabled())

        enable_spec_collection()
        self.assertTrue(is_aott_compile_enabled())

        disable_spec_collection()
        self.assertFalse(is_aott_compile_enabled())

    def test_override_remaps_only_the_given_marker(self) -> None:
        def custom_collector(marker: Any, *args: Any, **kwargs: Any) -> None:
            pass

        enable_spec_collection({TritonAOT: custom_collector})

        # The overridden marker got the custom collector; others kept the
        # default wiring.
        self.assertIs(TritonAOT.spec_collector, custom_collector)
        self.assertIsNotNone(CuTeAOT.spec_collector)
        self.assertIsNot(CuTeAOT.spec_collector, custom_collector)

        # disable_spec_collection clears overridden collectors too.
        disable_spec_collection()
        self.assertIsNone(TritonAOT.spec_collector)

    def test_override_rejects_unknown_marker(self) -> None:
        class NotAMarker(AOTTMarker):
            pass

        def custom_collector(marker: Any, *args: Any, **kwargs: Any) -> None:
            pass

        with self.assertRaises(ValueError):
            enable_spec_collection({NotAMarker: custom_collector})


class _RecordingAdapter(AOTTAdapter[Any]):
    """Fake DSL adapter that records the ``CompileContext`` it is handed."""

    name = "recording"

    def __init__(self) -> None:
        self.contexts: list[CompileContext] = []

    def compile_and_build(self, ctx: CompileContext) -> None:
        self.contexts.append(ctx)

    def find_kernel(self, node_target: Any) -> Any | None:
        return None

    def generate_wrapper_files(self, *args: Any, **kwargs: Any) -> None:
        pass


class _BoomAdapter(AOTTAdapter[Any]):
    """Fake DSL adapter whose build raises, to exercise ``__exit__`` cleanup."""

    name = "boom"

    def compile_and_build(self, ctx: CompileContext) -> None:
        raise RuntimeError("boom")

    def find_kernel(self, node_target: Any) -> Any | None:
        return None

    def generate_wrapper_files(self, *args: Any, **kwargs: Any) -> None:
        pass


class CompileSessionTest(unittest.TestCase):
    """``AOTTCompileSession.__exit__`` dispatches ``compile_and_build`` to every DSL
    that collected (via ``dsl_state``) and always clears collection afterwards.
    """

    def setUp(self) -> None:
        get_aott_compile_state().reset()

    def tearDown(self) -> None:
        get_aott_compile_state().reset()
        disable_spec_collection()

    def test_exit_dispatches_compile_and_build_then_disables(self) -> None:
        adapter = _RecordingAdapter()
        with AOTTCompileSession():
            # __enter__ has reset + enabled collection; seed a DSL as if it
            # collected during the body.
            self.assertTrue(is_aott_compile_enabled())
            get_aott_compile_state().dsl_state[adapter.name] = DslSpecStore(dsl=adapter)

        # __exit__ handed the collected DSL exactly one CompileContext...
        self.assertEqual(len(adapter.contexts), 1)
        self.assertIsInstance(adapter.contexts[0], CompileContext)
        # ...and cleared collection.
        self.assertFalse(is_aott_compile_enabled())

    def test_session_forwards_spec_collector_overrides(self) -> None:
        def custom_collector(marker: Any, *args: Any, **kwargs: Any) -> None:
            pass

        with AOTTCompileSession(spec_collector_overrides={TritonAOT: custom_collector}):
            self.assertIs(TritonAOT.spec_collector, custom_collector)
            self.assertIsNotNone(CuTeAOT.spec_collector)

        # __exit__ cleared the overridden collector along with the defaults.
        self.assertIsNone(TritonAOT.spec_collector)
        self.assertFalse(is_aott_compile_enabled())

    def test_exit_skips_compile_when_body_raises(self) -> None:
        adapter = _RecordingAdapter()
        with self.assertRaises(ValueError):
            with AOTTCompileSession():
                get_aott_compile_state().dsl_state[adapter.name] = DslSpecStore(
                    dsl=adapter
                )
                raise ValueError("body failed")

        # __exit__ skipped compile so a build error can't mask the body error...
        self.assertEqual(adapter.contexts, [])
        # ...but still cleared collection.
        self.assertFalse(is_aott_compile_enabled())

    def test_exit_disables_collection_even_if_build_raises(self) -> None:
        session = AOTTCompileSession()
        session.__enter__()
        get_aott_compile_state().dsl_state["boom"] = DslSpecStore(dsl=_BoomAdapter())

        with self.assertRaises(RuntimeError):
            session.__exit__(None, None, None)

        # The ``finally`` in __exit__ must still clear collection.
        self.assertFalse(is_aott_compile_enabled())
