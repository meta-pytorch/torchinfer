# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""CPU unit tests for CuTeDSL sidecar/torch-op codegen helpers.

These cover the per-arg codegen adapters (tensor / scalar / constexpr), the
header-driven tensor ABI (contiguous vs non-contiguous structs), the torch-op
surface (params / schema / variant selector), and validation — without
a GPU or a real CuTeDSL compile.
"""

import os
import sys
import tempfile
import types as pytypes
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from aot_tensor.arg_specs.common import ConstexprArg, RawSpec, ScalarArg
from aot_tensor.arg_specs.cute import CuTeTensorArg
from aot_tensor.compile.cutedsl import pipeline as cutedsl_mod
from aot_tensor.compile.cutedsl.codegen import (
    _SCALAR_DTYPES,
    ArgCodegen,
    ConstexprCodegen,
    cute_scalar,
    CuTeVariant,
    entry_args,
    entry_struct_builds,
    generate_sidecar_entry_content,
    generate_torch_op_content,
    make_codegen,
    make_variant,
    parse_tensor_struct_fields,
    ScalarCodegen,
    TensorCodegen,
    validate_cutedsl_op,
)
from aot_tensor.compile.cutedsl.pipeline import compile_cutedsl_to_cpp
from aot_tensor.compile.spec_selector import Variant
from aot_tensor.constants import DEFAULT_OP_NAMESPACE_PREFIX
from parameterized import parameterized

_PREFIX = "pfx"

# A contiguous tensor (dynamic shapes, derived strides) and a non-contiguous one
# (dynamic shapes and strides), as emitted by ``export_to_c``.
_HEADER = """
typedef struct { void* data; int32_t dynamic_shapes[2]; } pfx_Tensor_x_t;
typedef struct { void* data; int32_t dynamic_shapes[2]; int64_t dynamic_strides[2]; } pfx_Tensor_y_t;
"""

# Heterogeneous dynamic-dim counts: a rank-3 tensor, a rank-2 tensor, and a
# rank-1 contiguous tensor (shapes only). The per-struct body must be read in
# isolation -- a regex that spans from the first struct into a later one makes
# every later tensor mis-read the first tensor's array sizes.
_HETERO_HEADER = """
typedef struct { void* data; int32_t dynamic_shapes[3]; int64_t dynamic_strides[2]; } pfx_Tensor_a_t;
typedef struct { void* data; int32_t dynamic_shapes[2]; int64_t dynamic_strides[1]; } pfx_Tensor_bias_t;
typedef struct { void* data; int32_t dynamic_shapes[1]; } pfx_Tensor_seq_t;
"""


class ParseTensorStructFieldsTest(unittest.TestCase):
    def test_contiguous_has_shapes_only(self) -> None:
        self.assertEqual(parse_tensor_struct_fields(_HEADER, _PREFIX, "x"), (2, 0))

    def test_non_contiguous_has_shapes_and_strides(self) -> None:
        self.assertEqual(parse_tensor_struct_fields(_HEADER, _PREFIX, "y"), (2, 2))

    def test_missing_struct_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_tensor_struct_fields(_HEADER, _PREFIX, "missing")

    def test_heterogeneous_ranks_read_per_struct(self) -> None:
        # Each struct's counts are read independently of neighbors, even when a
        # lower-rank tensor follows a higher-rank one in the header.
        self.assertEqual(
            parse_tensor_struct_fields(_HETERO_HEADER, _PREFIX, "a"), (3, 2)
        )
        self.assertEqual(
            parse_tensor_struct_fields(_HETERO_HEADER, _PREFIX, "bias"), (2, 1)
        )
        self.assertEqual(
            parse_tensor_struct_fields(_HETERO_HEADER, _PREFIX, "seq"), (1, 0)
        )


class EntryAbiConsistencyTest(unittest.TestCase):
    """Each entry arg pairs a decl, a fn-ptr type, and a call expr in lockstep."""

    def test_contiguous_tensor_omits_strides(self) -> None:
        args = entry_args(
            [make_codegen(CuTeTensorArg("x")), make_codegen(ScalarArg("n", "i32"))],
            {"x": (2, 0)},
        )
        self.assertEqual(
            [a.decl for a in args],
            ["void* x_data", "const int64_t* x_sizes", "int32_t n"],
        )
        self.assertEqual(
            [a.ctype for a in args], ["void*", "const int64_t*", "int32_t"]
        )
        self.assertEqual(
            [a.call for a in args],
            ["x.data_ptr()", "x.sizes().data()", 'cutedsl_i32(n, "n")'],
        )

    def test_non_contiguous_tensor_includes_strides(self) -> None:
        args = entry_args([make_codegen(CuTeTensorArg("y"))], {"y": (2, 2)})
        self.assertEqual(
            [a.decl for a in args],
            ["void* y_data", "const int64_t* y_sizes", "const int64_t* y_strides"],
        )
        self.assertEqual(
            [a.call for a in args],
            ["y.data_ptr()", "y.sizes().data()", "y.strides().data()"],
        )

    @parameterized.expand(
        [
            ("i32", "int32_t"),
            ("i64", "int64_t"),
            ("fp32", "float"),
            ("bool", "bool"),
        ]
    )
    def test_scalar_entry_param_per_dtype(self, dtype: str, entry: str) -> None:
        args = entry_args([make_codegen(ScalarArg("v", dtype))], {})
        self.assertEqual(args[0].decl, f"{entry} v")
        self.assertEqual(args[0].ctype, entry)


class EntryStructBuildsTest(unittest.TestCase):
    def test_contiguous_fills_shapes_only(self) -> None:
        out = entry_struct_builds(
            [make_codegen(CuTeTensorArg("x"))], _PREFIX, {"x": (2, 0)}
        )
        self.assertIn("pfx_Tensor_x_t x_t;", out)
        self.assertIn("x_t.dynamic_shapes[0] = static_cast<int32_t>(x_sizes[0]);", out)
        self.assertIn("x_t.dynamic_shapes[1] = static_cast<int32_t>(x_sizes[1]);", out)
        self.assertNotIn("dynamic_strides", out)

    def test_non_contiguous_fills_strides(self) -> None:
        out = entry_struct_builds(
            [make_codegen(CuTeTensorArg("y"))], _PREFIX, {"y": (2, 2)}
        )
        self.assertIn("y_t.dynamic_strides[0] = y_strides[0];", out)
        self.assertIn("y_t.dynamic_strides[1] = y_strides[1];", out)

    def test_leading_dim_strides_skip_static_unit_stride_dim(self) -> None:
        # A tensor whose static unit-stride dim is not dim 0 (e.g. an N-major
        # [N, K, G] operand, leading_dim=0). export_to_c omits that dim from
        # dynamic_strides, so the stride slots must map to the remaining torch
        # dims in order (1, 2), not identity (0, 1) which would feed the static
        # unit stride into the kernel layout.
        out = entry_struct_builds(
            [make_codegen(CuTeTensorArg("b", leading_dim=0))], _PREFIX, {"b": (3, 2)}
        )
        # dynamic_shapes stay identity (every dim has a dynamic shape).
        self.assertIn("b_t.dynamic_shapes[0] = static_cast<int32_t>(b_sizes[0]);", out)
        self.assertIn("b_t.dynamic_shapes[2] = static_cast<int32_t>(b_sizes[2]);", out)
        # dynamic_strides skip the static leading dim 0 -> torch dims 1, 2.
        self.assertIn("b_t.dynamic_strides[0] = b_strides[1];", out)
        self.assertIn("b_t.dynamic_strides[1] = b_strides[2];", out)


class MakeCodegenTest(unittest.TestCase):
    def test_dispatch_per_kind(self) -> None:
        self.assertIsInstance(make_codegen(CuTeTensorArg("x")), TensorCodegen)
        self.assertIsInstance(make_codegen(ScalarArg("n", "i32")), ScalarCodegen)
        self.assertIsInstance(make_codegen(ConstexprArg("c", int)), ConstexprCodegen)

    def test_unsupported_spec_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            make_codegen(object())  # pyre-ignore[6]: intentionally bad spec


class TorchParamTest(unittest.TestCase):
    def test_tensor_cpp_and_schema(self) -> None:
        cg = make_codegen(CuTeTensorArg("x"))
        self.assertEqual(cg.torch_cpp_param(), "torch::stable::Tensor x")
        # Schema alias tracks the arg's position in the op signature.
        self.assertEqual(cg.torch_schema_param(0), "Tensor(a!) x")
        self.assertEqual(cg.torch_schema_param(1), "Tensor(b!) x")

    def test_scalar_cpp_widens_but_schema_stays_int(self) -> None:
        cg = make_codegen(ScalarArg("n", "i32"))
        self.assertEqual(cg.torch_cpp_param(), "int64_t n")
        self.assertEqual(cg.torch_schema_param(0), "int n")

    @parameterized.expand(
        [
            ("bool", bool, "bool", "bool"),
            ("int", int, "int64_t", "int"),
            ("float", float, "double", "float"),
        ]
    )
    def test_constexpr_cpp_and_schema(
        self, _name: str, py_type: type, cpp: str, schema: str
    ) -> None:
        cg = make_codegen(ConstexprArg("c", py_type))
        self.assertEqual(cg.torch_cpp_param(), f"{cpp} c")
        self.assertEqual(cg.torch_schema_param(0), f"{schema} c")

    def test_constexpr_unsupported_type_raises(self) -> None:
        cg = make_codegen(ConstexprArg("c", str))
        with self.assertRaises(RuntimeError):
            cg.torch_cpp_param()
        with self.assertRaises(RuntimeError):
            cg.torch_schema_param(0)


class ConstexprCodegenTest(unittest.TestCase):
    def test_not_runtime_contributes_nothing_to_entry(self) -> None:
        cg = make_codegen(ConstexprArg("flag", bool))
        self.assertFalse(cg.is_runtime)
        self.assertEqual(cg.entry_args({}), [])
        self.assertIsNone(cg.wrapper_arg())

    def test_cute_arg_passthrough(self) -> None:
        self.assertIs(make_codegen(ConstexprArg("flag", bool)).cute_arg(True), True)
        self.assertEqual(make_codegen(ConstexprArg("n", int)).cute_arg(7), 7)


class ValidateTest(unittest.TestCase):
    def test_clean_op_passes(self) -> None:
        # A well-formed op validates without raising (returns None).
        self.assertIsNone(
            validate_cutedsl_op(
                [
                    make_codegen(CuTeTensorArg("x")),
                    make_codegen(ScalarArg("n", "i32")),
                ],
                "op",
            )
        )

    def test_bad_scalar_dtype_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_cutedsl_op([make_codegen(ScalarArg("n", "f16"))], "op")

    def test_bad_identifier_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_cutedsl_op([], "bad-op")


class ScalarDtypeTableTest(unittest.TestCase):
    def test_entry_cast_formats_for_every_dtype(self) -> None:
        for dtype, info in _SCALAR_DTYPES.items():
            cast = info.entry_cast.format(name="v")
            self.assertIn("v", cast, f"{dtype} cast must reference the arg")

    def test_scalar_dtype_table_attrs(self) -> None:
        # cute_ctor ("" means no cutlass type) and py_cast, per dtype.
        expected = {
            "i32": ("Int32", int),
            "i64": ("Int64", int),
            "fp32": ("Float32", float),
            "bool": ("", bool),
        }
        for dtype, (ctor, py_cast) in expected.items():
            with self.subTest(dtype=dtype):
                info = _SCALAR_DTYPES[dtype]
                self.assertEqual(info.cute_ctor, ctor)
                self.assertIs(info.py_cast, py_cast)

    def test_cute_scalar_bool_needs_no_cutlass(self) -> None:
        # Dtypes with an empty cute_ctor (bool) go through py_cast, not cutlass.
        self.assertIs(cute_scalar(1, "bool"), True)
        self.assertIs(cute_scalar(0, "bool"), False)

    def test_cute_scalar_unknown_dtype_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            cute_scalar(1, "f16")


_CODEGENS: list[ArgCodegen[Any]] = [
    make_codegen(CuTeTensorArg("x")),
    make_codegen(ScalarArg("n", "i32")),
]


def _variant(symbol: str, condition: str, x_struct_body: str) -> CuTeVariant:
    header = f"typedef struct {{ void* data; {x_struct_body} }} {symbol}_Tensor_x_t;"
    return make_variant(
        _CODEGENS, Variant(symbol, condition, f"{symbol} signature"), header
    )


_F32 = _variant("op__aaaa", "x.scalar_type() == F32", "int32_t dynamic_shapes[2];")
# Non-contiguous: its entry takes strides too.
_F16 = _variant(
    "op__bbbb",
    "x.scalar_type() == F16",
    "int32_t dynamic_shapes[2]; int64_t dynamic_strides[2];",
)


class MakeVariantTest(unittest.TestCase):
    def test_reads_fields_from_the_header_and_derives_the_entry_abi(self) -> None:
        self.assertEqual(_F16.fields, {"x": (2, 2)})
        self.assertEqual(
            [a.decl for a in _F16.entry_args],
            [
                "void* x_data",
                "const int64_t* x_sizes",
                "const int64_t* x_strides",
                "int32_t n",
            ],
        )


class GenerateContentTest(unittest.TestCase):
    """Render both shims for two variants (CPU, no compile)."""

    def test_sidecar_entry(self) -> None:
        out = generate_sidecar_entry_content("op", _CODEGENS, [_F32, _F16])
        self.assertIn('#define CUTEDSL_OP_NAME "op"', out)
        self.assertIn('#include "op__aaaa.h"\n#include "op__bbbb.h"', out)
        for symbol in ("op__aaaa", "op__bbbb"):
            self.assertIn(f"int32_t {symbol}_entry(void* x_data", out)
            self.assertIn(f"{symbol}_Tensor_x_t x_t;", out)
            self.assertIn(f"cute_dsl_{symbol}_wrapper(&module,", out)
        # Only the non-contiguous variant's entry takes strides.
        self.assertEqual(out.count("const int64_t* x_strides"), 1)

    def test_torch_op(self) -> None:
        out = generate_torch_op_content("op", _CODEGENS, [_F32, _F16], "impl.so")
        self.assertIn(
            f"#define CUTEDSL_OP_NAMESPACE {DEFAULT_OP_NAMESPACE_PREFIX}\n", out
        )
        self.assertIn('#define CUTEDSL_OP_SCHEMA "op(Tensor(a!) x, int n) -> ()"', out)
        self.assertIn('#define CUTEDSL_SIDECAR_NAME "impl.so"', out)
        for symbol in ("op__aaaa", "op__bbbb"):
            self.assertIn(f"void {symbol}_run(torch::stable::Tensor x, int64_t n)", out)
            self.assertIn(f'cutedsl_load_entry<EntryFn>("{symbol}_entry")', out)
        # Each run function calls its own variant's entry ABI.
        self.assertIn(
            "using EntryFn = int32_t (*)(void*, const int64_t*, int32_t, CUstream);",
            out,
        )
        self.assertIn(
            "using EntryFn = int32_t (*)(void*, const int64_t*, const int64_t*, "
            "int32_t, CUstream);",
            out,
        )
        self.assertIn('cutedsl_i32(n, "n")', out)
        self.assertIn("if (x.scalar_type() == F32) {\n    op__aaaa_run(x, n);", out)
        self.assertIn("if (x.scalar_type() == F16) {\n    op__bbbb_run(x, n);", out)
        self.assertIn(
            'cutedsl_fail("no compiled specialization matches the arguments; '
            'compiled: op__aaaa signature; op__bbbb signature");',
            out,
        )


class ArgCodegenBaseTest(unittest.TestCase):
    """The base class supplies constexpr-friendly defaults; the torch/cute hooks
    are abstract and every concrete kind overrides them."""

    def test_defaults(self) -> None:
        cg: ArgCodegen[CuTeTensorArg] = ArgCodegen(CuTeTensorArg("x"))
        self.assertEqual(cg.entry_args({}), [])
        self.assertEqual(cg.struct_build_lines("pfx", {}), [])
        self.assertIsNone(cg.wrapper_arg())
        self.assertIsNone(cg.validate("op"))

    def test_abstract_hooks_raise(self) -> None:
        cg: ArgCodegen[CuTeTensorArg] = ArgCodegen(CuTeTensorArg("x"))
        with self.assertRaises(NotImplementedError):
            cg.torch_cpp_param()
        with self.assertRaises(NotImplementedError):
            cg.torch_schema_param(0)
        with self.assertRaises(NotImplementedError):
            cg.cute_arg(1)


def _vector_add_spec(x: str, dtype: str, aligned16: bool = True) -> RawSpec:
    return RawSpec(
        runtime_args=(x, 4),
        runtime_kwargs={},
        hash_key={
            "name": "_cutedsl_vector_add",
            "jit_module": "toy",
            "args": [
                {
                    "name": "x",
                    "tensor": {
                        "dtype": dtype,
                        "dim": 1,
                        "leading_dim": 0,
                        "aligned16": aligned16,
                    },
                },
                {"name": "n", "scalar": {"dtype": "i32"}},
            ],
        },
    )


class _FakeCompiled:
    """Stands in for ``cute.compile``'s result: ``export_to_c`` writes the
    ``{symbol}.h`` / ``{symbol}.o`` pair the pipeline reads and links."""

    def export_to_c(self, install_dir: str, symbol: str) -> None:
        with open(os.path.join(install_dir, f"{symbol}.h"), "w") as fp:
            fp.write(
                "typedef struct { void* data; int32_t dynamic_shapes[1]; } "
                f"{symbol}_Tensor_x_t;\n"
            )
        with open(os.path.join(install_dir, f"{symbol}.o"), "w") as fp:
            fp.write("")


class CompileCuTeDSLToCppTest(unittest.TestCase):
    """Orchestrator path with cutlass + stream glue stubbed.

    Exercises resolve_runtime_args + build_cute_call_args wiring and the
    per-variant export on CPU; the real cute.compile/export_to_c are GPU-only
    and covered by the E2E test.
    """

    def test_exports_one_object_per_distinct_specialization(self) -> None:
        op = SimpleNamespace(
            name="_cutedsl_vector_add",
            arg_specs=[CuTeTensorArg("x"), ScalarArg("n", "i32")],
            jit_fn=lambda *a: None,
        )
        specs = [
            _vector_add_spec("X32", "torch.float32"),
            _vector_add_spec("X16", "torch.float16"),
            # Differs only in the sample's alignment: same compiled code.
            _vector_add_spec("X32u", "torch.float32", aligned16=False),
        ]

        fake_cute = pytypes.ModuleType("cutlass.cute")
        # pyre-ignore[16]: attaching a stub attribute to a fake module.
        fake_cute.compile = MagicMock(return_value=_FakeCompiled())
        fake_cutlass = pytypes.ModuleType("cutlass")
        # pyre-ignore[16]
        fake_cutlass.cute = fake_cute
        fake_export = pytypes.ModuleType("cutlass.cute.export")

        with (
            tempfile.TemporaryDirectory() as install_dir,
            patch.dict(
                sys.modules,
                {
                    "cutlass": fake_cutlass,
                    "cutlass.cute": fake_cute,
                    "cutlass.cute.export": fake_export,
                },
            ),
            patch.object(
                cutedsl_mod,
                "build_cute_call_args",
                side_effect=lambda arg_specs, values: (values["x"],),
            ),
        ):
            # pyre-ignore[6]: op is a minimal stub for this codegen path.
            kernel_obj_files = compile_cutedsl_to_cpp(op, specs, install_dir)

            symbols = [
                os.path.splitext(os.path.basename(o))[0] for o in kernel_obj_files
            ]
            self.assertEqual(len(symbols), 2)
            for obj in kernel_obj_files:
                self.assertTrue(os.path.exists(obj))
            with open(os.path.join(install_dir, f"{op.name}_entry.cpp")) as fp:
                entry = fp.read()
            with open(os.path.join(install_dir, f"{op.name}_torch_op.cpp")) as fp:
                torch_op = fp.read()

        self.assertEqual(
            [c.args for c in fake_cute.compile.call_args_list],
            [(op.jit_fn, "X32"), (op.jit_fn, "X16")],
        )
        for symbol in symbols:
            self.assertTrue(symbol.startswith("_cutedsl_vector_add__"))
            self.assertIn(f"int32_t {symbol}_entry(", entry)
            self.assertIn(f"{symbol}_run(x, n);", torch_op)
        self.assertIn(
            '#define CUTEDSL_SIDECAR_NAME "cutedsl_vector_add_cutedsl_impl.so"',
            torch_op,
        )
        self.assertIn(
            "x.scalar_type() == torch::headeronly::ScalarType::Half", torch_op
        )
