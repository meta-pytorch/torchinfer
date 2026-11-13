# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import re
import unittest

from aot_tensor.arg_specs.common import ConstexprArg, ScalarArg
from aot_tensor.arg_specs.fly import FlyArgSpec, FlyTensorArg
from aot_tensor.autotune.routing import RoutingTable
from aot_tensor.compile.flydsl.abi import AbiSlot, ExportedLauncher
from aot_tensor.compile.flydsl.codegen import FlyVariant, generate_torch_op_content
from aot_tensor.compile.spec_selector import Route, Variant
from aot_tensor.constants import DEFAULT_OP_NAMESPACE_PREFIX
from parameterized import parameterized

_OP = "_flydsl_vec_launch"
_SPECS: list[FlyArgSpec] = [
    FlyTensorArg("x", leading_dim=1, divisibility=4),
    ScalarArg("n", "i32"),
    ConstexprArg("fast", bool),
]
# x is 2-D with dynamic shape[0..1] and stride[0] (64-bit), then n, then stream.
_ABI: tuple[AbiSlot, ...] = (
    AbiSlot("tensor_data", "x", "pointer", 8),
    AbiSlot(
        "tensor_layout",
        "x",
        "bytes",
        16,
        shape_dims=(0, 1),
        stride_dims=(0,),
        stride_bits=64,
    ),
    AbiSlot("scalar", "n", "int32", 4),
    AbiSlot("stream", "stream", "pointer", 8),
)


def _variant(tag: str, condition: str, abi: tuple[AbiSlot, ...] = _ABI) -> FlyVariant:
    symbol = f"{_OP}__{tag}"
    return FlyVariant(
        Variant(symbol, condition, f"fast={condition}"),
        ExportedLauncher(
            object_path=f"/out/{symbol}.o",
            entry_symbol=f"aott{symbol}",
            module_init_symbol=f"aott{symbol}__module_init",
            module_load_symbol=f"aott{symbol}__module_load",
            abi=abi,
        ),
    )


_FAST = _variant("aaaa", "fast == true")
_SLOW = _variant("bbbb", "fast == false")


def _route(fly: FlyVariant) -> Route:
    variant = fly.variant
    return Route(
        variant.condition, variant.description, RoutingTable({(): variant.symbol})
    )


class GenerateTorchOpTest(unittest.TestCase):
    def _render(self, *variants: FlyVariant) -> str:
        variants = variants or (_FAST,)
        routes = [_route(fly) for fly in variants]
        return generate_torch_op_content(
            _OP, _SPECS, list(variants), frozenset({"x"}), routes
        )

    @parameterized.expand(
        [
            ("mutated", frozenset({"x"}), "Tensor(a!)? x"),
            ("read_only", frozenset(), "Tensor? x"),
        ]
    )
    def test_namespace_and_schema(
        self, _name: str, mutated: frozenset[str], tensor: str
    ) -> None:
        out = generate_torch_op_content(_OP, _SPECS, [_FAST], mutated, [_route(_FAST)])
        self.assertIn(f"#define FLYDSL_OP_NAMESPACE {DEFAULT_OP_NAMESPACE_PREFIX}", out)
        self.assertIn(
            f'#define FLYDSL_OP_SCHEMA "{_OP}({tensor}, SymInt n, bool fast) -> ()"',
            out,
        )

    def test_tensors_arrive_optional_and_are_unwrapped_before_dispatch(self) -> None:
        out = self._render()
        self.assertIn(
            "void flydsl_op(std::optional<torch::stable::Tensor> x_opt, "
            "int64_t n, bool fast) {\n"
            '  const torch::stable::Tensor x = flydsl_required(x_opt, "x");\n',
            out,
        )

    def test_each_variant_declares_and_calls_its_own_launcher(self) -> None:
        out = self._render(_FAST, _SLOW)
        for tag in ("aaaa", "bbbb"):
            symbol = f"{_OP}__{tag}"
            exported = f"aott{symbol}"
            for extern in (
                f"int32_t {exported}(void** args);",
                f"int32_t {exported}__module_init(void);",
                f"int32_t {exported}__module_load(int32_t device);",
            ):
                self.assertIn(f'extern "C" {extern}', out)
            self.assertIn(f"void {symbol}_run(", out)
            # Init once, load on the current device, then launch.
            calls = [
                f"std::call_once(initialized, [] {{ flydsl_check_status("
                f'{exported}__module_init(), "{exported}__module_init"); }});',
                f'flydsl_check_status({exported}__module_load(-1), "{exported}__module_load");',
                f'flydsl_check_status({exported}(args), "{exported}");',
            ]
            for call in calls:
                self.assertIn(call, out)
            positions = [out.index(call) for call in calls]
            self.assertEqual(positions, sorted(positions))

    def test_op_dispatches_on_the_variant_conditions(self) -> None:
        out = self._render(_FAST, _SLOW)
        fast = out.index("if (fast == true) {")
        slow = out.index("if (fast == false) {")
        self.assertLess(fast, slow)
        self.assertIn(f"{_OP}__aaaa_run(x, n, fast);", out)
        self.assertIn('flydsl_fail("no compiled specialization matches', out)

    def test_packs_one_pointer_per_abi_slot_in_order(self) -> None:
        out = self._render()
        self.assertIn(
            "void* args[] = {&slot0, static_cast<void*>(slot1), &slot2, &slot3};", out
        )
        self.assertIn("void* slot0 = x.data_ptr();", out)
        self.assertIn('int32_t slot2 = static_cast<int32_t>(flydsl_i32(n, "n"));', out)
        self.assertIn("void* slot3 = flydsl_current_stream(device_index);", out)

    def test_layout_bytes_follow_the_abi_dims(self) -> None:
        out = self._render()
        self.assertIn("alignas(8) uint8_t slot1[16] = {};", out)
        self.assertIn("flydsl_put<int32_t>(slot1, sizeof(slot1), 0, ", out)
        self.assertIn("x.size(0)", out)
        self.assertIn("flydsl_put<int32_t>(slot1, sizeof(slot1), 4, ", out)
        self.assertIn("flydsl_put<int64_t>(slot1, sizeof(slot1), 8, x.stride(0));", out)

    def test_layout_bytes_follow_the_abi_widths(self) -> None:
        wide = AbiSlot(
            "tensor_layout",
            "x",
            "bytes",
            12,
            shape_dims=(0,),
            stride_dims=(0,),
            shape_bits=64,
            stride_bits=32,
        )
        abi = tuple(wide if s.kind == "tensor_layout" else s for s in _ABI)
        out = self._render(_variant("dddd", "true", abi))
        self.assertIn("flydsl_put<int64_t>(slot1, sizeof(slot1), 0, x.size(0));", out)
        self.assertIn(
            'flydsl_put<int32_t>(slot1, sizeof(slot1), 8, flydsl_i32(x.stride(0), "x.stride(0)"));',
            out,
        )

    def test_checks_the_layout_the_launcher_was_compiled_for(self) -> None:
        out = self._render()
        for check in (
            "reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0",
            "x.stride(0) % 4 == 0",
        ):
            self.assertIn(f"if (!({check})) flydsl_fail(", out)
        self.assertNotIn("x.stride(1) == 1", out)

    def test_stays_on_the_stable_abi(self) -> None:
        out = self._render(_FAST, _SLOW)
        for forbidden in (
            "c10::",
            "at::",
            "<torch/library.h>",
            "<ATen/",
            "aoti_runtime",
        ):
            self.assertNotIn(forbidden, out)
        self.assertIsNone(re.search(r"\bmemcpy\b", out))

    @parameterized.expand(
        [
            (
                "scalar_dtype_mismatch",
                tuple(
                    AbiSlot("scalar", "n", "int64", 8) if s.kind == "scalar" else s
                    for s in _ABI
                ),
                "declared i32 but the launcher",
            ),
            (
                "arg_without_spec",
                (*_ABI, AbiSlot("scalar", "m", "int32", 4)),
                "do not match the launcher ABI",
            ),
        ]
    )
    def test_rejects_abi_not_matching_the_specs(
        self, _name: str, abi: tuple[AbiSlot, ...], error: str
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, error):
            self._render(_variant("cccc", "true", abi))
