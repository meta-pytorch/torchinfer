# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Unit tests for aot_tensor.compile.triton.compat module."""

import types
import unittest
from unittest.mock import MagicMock

from aot_tensor.compile.triton.compat import (
    _get_num_ctas,
    get_kernel_name,
    get_scratch_parameters,
    version_gte,
)
from parameterized import parameterized


class VersionGteTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("3.10", "3.5", True),  # string comparison would fail
            ("3.5", "3.5", True),
            ("3.4", "3.5", False),
            ("3.5.0a1", "3.5.0", False),
            ("3.5.0.post1", "3.5.0", True),
            ("3.3.2+fb", "3.5.0+fb", False),
        ]
    )
    def test_version_gte(self, version: str, target: str, expected: bool) -> None:
        self.assertEqual(version_gte(version, target), expected)


class GetKernelNameTest(unittest.TestCase):
    def test_simple_name(self) -> None:
        mock_jit_fn = MagicMock()
        mock_jit_fn._fn_name = "aot_tensor.ops.triton_addmm._addmm_fwd"
        result = get_kernel_name(mock_jit_fn)
        self.assertEqual(result, "_addmm_fwd")


def _mock_kernel(
    global_scratch_size: int | None = 0,
    cluster_dims: tuple[int, int, int] | None = (1, 1, 1),
) -> MagicMock:
    """``=None`` for any arg omits that field. SimpleNamespace (not
    MagicMock) for metadata so ``getattr(..., default)`` defenses fire.
    """
    attrs: dict[str, object] = {"profile_scratch_size": 0}
    if global_scratch_size is not None:
        attrs["global_scratch_size"] = global_scratch_size
    if cluster_dims is not None:
        attrs["cluster_dims"] = cluster_dims
    kernel = MagicMock()
    kernel.metadata = types.SimpleNamespace(**attrs)
    return kernel


class GetNumCtasTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("default_trivial_cluster", (1, 1, 1), 1),
            ("cluster_dims_2x1x1", (2, 1, 1), 2),
            ("cluster_dims_2x2x1", (2, 2, 1), 4),
            # cluster_dims attr missing → getattr fallback to (1,1,1).
            ("cluster_dims_attr_missing", None, 1),
        ]
    )
    def test_num_ctas_derivation(
        self,
        _label: str,
        cluster_dims: tuple[int, int, int] | None,
        expected: int,
    ) -> None:
        kernel = _mock_kernel(global_scratch_size=0, cluster_dims=cluster_dims)
        self.assertEqual(_get_num_ctas(kernel), expected)


class GetScratchParametersTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("cuda_zero_size", "cuda", 0),
            ("hip_nonzero_size", "hip", 128),  # HIP skips TMA alloc.
            ("amd_fork_missing_field", "cuda", None),
        ]
    )
    def test_null_path(
        self, _label: str, backend: str, global_scratch_size: int | None
    ) -> None:
        declarations, arg_pointers = get_scratch_parameters(
            _mock_kernel(global_scratch_size), backend
        )
        self.assertIn("CUdeviceptr global_scratch = 0;", declarations)
        self.assertNotIn("aoti_torch_empty_strided", declarations)
        self.assertNotIn("torch::stable::Tensor _global_scratch_tensor", declarations)
        self.assertNotIn("global_scratch_num_ctas", declarations)
        self.assertEqual(arg_pointers, ["&global_scratch", "&profile_scratch"])

    def test_cuda_with_tma_scratch_emits_empty_strided_block(self) -> None:
        declarations, arg_pointers = get_scratch_parameters(_mock_kernel(128), "cuda")
        for marker in (
            "constexpr int64_t global_scratch_per_program = 128;",
            "constexpr int64_t global_scratch_num_ctas = 1;",
            "global_scratch_per_program * global_scratch_num_ctas",
            "STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided",
            "aoti_torch_dtype_uint8()",
            "aoti_torch_device_type_cuda()",
            "torch::stable::Tensor _global_scratch_tensor",
        ):
            self.assertIn(marker, declarations)
        # Guards against regression to non-stable-ABI surface.
        self.assertNotIn("AOTI_TORCH_ERROR_CODE_CHECK", declarations)
        self.assertNotIn("RAIIAtenTensorHandle", declarations)
        self.assertNotIn("cuMemAllocAsync", declarations)
        self.assertNotIn("TritonAotScratchGuard", declarations)
        self.assertEqual(arg_pointers, ["&global_scratch", "&profile_scratch"])

    @parameterized.expand(
        [
            ("trivial_cluster", (1, 1, 1), 1),
            ("clustered_2x1x1", (2, 1, 1), 2),
            ("clustered_2x2x1", (2, 2, 1), 4),
        ]
    )
    def test_num_ctas_baked_into_sizing_constexpr(
        self,
        _label: str,
        cluster_dims: tuple[int, int, int],
        expected_num_ctas: int,
    ) -> None:
        declarations, _ = get_scratch_parameters(
            _mock_kernel(global_scratch_size=128, cluster_dims=cluster_dims),
            "cuda",
        )
        self.assertIn(
            f"constexpr int64_t global_scratch_num_ctas = {expected_num_ctas};",
            declarations,
        )
