# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch
from aot_tensor.compile.adapter_base import AOTTAdapter, DslSpecStore
from aot_tensor.compile.compile_state import get_aott_compile_state
from aot_tensor.transform.kernel_wrapper_codegen import kernel_wrapper_codegen
from torch.fx import GraphModule


@torch.fx.wrap
def _aott_wrapper(x: torch.Tensor) -> torch.Tensor:
    return x


class _WrapperModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _aott_wrapper(x)


def _mock_dsl(name: str, match: object | None) -> MagicMock:
    dsl = MagicMock()
    dsl.name = name
    dsl.find_kernel.return_value = match
    return dsl


class KernelWrapperCodegenTest(unittest.TestCase):
    def setUp(self) -> None:
        state = get_aott_compile_state()
        self.enterContext(patch.dict(state.dsl_state, clear=True))
        state.reset()
        self.compile_path = "test_compile_path"
        state.compile_path = self.compile_path
        state.session_completed = True
        self.addCleanup(state.reset)

    def _graph_module(self) -> GraphModule:
        return torch.fx.symbolic_trace(_WrapperModule())

    def _seed_dsls(self, *dsls: MagicMock) -> None:
        state = get_aott_compile_state()
        for dsl in dsls:
            state.dsl_state[dsl.name] = DslSpecStore(dsl=cast(AOTTAdapter[Any], dsl))

    def test_raises_without_completed_session(self) -> None:
        get_aott_compile_state().reset()

        with self.assertRaisesRegex(
            AssertionError, "AOTTCompileSession has not completed"
        ):
            kernel_wrapper_codegen(self._graph_module())

    def test_raises_when_multiple_dsls_claim_one_wrapper(self) -> None:
        self._seed_dsls(_mock_dsl("triton", object()), _mock_dsl("cutedsl", object()))

        with self.assertRaisesRegex(RuntimeError, "multiple DSLs"):
            kernel_wrapper_codegen(self._graph_module())

    def test_dispatches_only_to_matching_dsl(self) -> None:
        match = object()
        matching = _mock_dsl("triton", match)
        non_matching = _mock_dsl("cutedsl", None)
        self._seed_dsls(matching, non_matching)
        package_importer = MagicMock()

        kernel_wrapper_codegen(self._graph_module(), package_importer)

        matching.generate_wrapper_files.assert_called_once_with(
            _aott_wrapper,
            match,
            self.compile_path,
            package_importer,
        )
        non_matching.generate_wrapper_files.assert_not_called()

    def test_does_not_generate_files_when_no_dsl_matches(self) -> None:
        triton = _mock_dsl("triton", None)
        cutedsl = _mock_dsl("cutedsl", None)
        self._seed_dsls(triton, cutedsl)

        kernel_wrapper_codegen(self._graph_module())

        triton.generate_wrapper_files.assert_not_called()
        cutedsl.generate_wrapper_files.assert_not_called()
