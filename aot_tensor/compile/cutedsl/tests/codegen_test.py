# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""CPU unit tests for CuTeDSL sidecar/torch-op codegen helpers.

These cover the per-arg codegen adapters (tensor / scalar / constexpr), the
header-driven tensor ABI (contiguous vs non-contiguous structs), the torch-op
surface (params / schema / guards), and validation — without a GPU or a real
CuTeDSL compile.
"""

import sys
import tempfile
import types as pytypes
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aot_tensor.compile.cutedsl import pipeline as cutedsl_mod
from aot_tensor.compile.cutedsl.codegen import (
    _entry_args,
    _entry_struct_builds,
    _format_constexpr_value,
    _SCALAR_DTYPES,
    ArgCodegen,
    ConstexprCodegen,
    cute_scalar,
    make_codegen,
    parse_tensor_struct_fields,
    ScalarCodegen,
    TensorCodegen,
    validate_cutedsl_op,
)
from aot_tensor.compile.cutedsl.pipeline import (
    compile_cutedsl_to_cpp,
    generate_sidecar_entry_content,
    generate_torch_op_content,
)
from aot_tensor.cute_specs import CuTeConstexprArg, CuTeScalarArg, CuTeTensorArg
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
        args = _entry_args(
            [make_codegen(CuTeTensorArg("x")), make_codegen(CuTeScalarArg("n", "i32"))],
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
            ["x.data_ptr()", "x.sizes().data()", "static_cast<int32_t>(n)"],
        )

    def test_non_contiguous_tensor_includes_strides(self) -> None:
        args = _entry_args([make_codegen(CuTeTensorArg("y"))], {"y": (2, 2)})
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
        args = _entry_args([make_codegen(CuTeScalarArg("v", dtype))], {})
        self.assertEqual(args[0].decl, f"{entry} v")
        self.assertEqual(args[0].ctype, entry)


class EntryStructBuildsTest(unittest.TestCase):
    def test_contiguous_fills_shapes_only(self) -> None:
        out = _entry_struct_builds(
            [make_codegen(CuTeTensorArg("x"))], _PREFIX, {"x": (2, 0)}
        )
        self.assertIn("pfx_Tensor_x_t x_t;", out)
        self.assertIn("x_t.dynamic_shapes[0] = static_cast<int32_t>(x_sizes[0]);", out)
        self.assertIn("x_t.dynamic_shapes[1] = static_cast<int32_t>(x_sizes[1]);", out)
        self.assertNotIn("dynamic_strides", out)

    def test_non_contiguous_fills_strides(self) -> None:
        out = _entry_struct_builds(
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
        out = _entry_struct_builds(
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
        self.assertIsInstance(make_codegen(CuTeScalarArg("n", "i32")), ScalarCodegen)
        self.assertIsInstance(
            make_codegen(CuTeConstexprArg("c", int)), ConstexprCodegen
        )

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
        cg = make_codegen(CuTeScalarArg("n", "i32"))
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
        cg = make_codegen(CuTeConstexprArg("c", py_type))
        self.assertEqual(cg.torch_cpp_param(), f"{cpp} c")
        self.assertEqual(cg.torch_schema_param(0), f"{schema} c")

    def test_constexpr_unsupported_type_raises(self) -> None:
        cg = make_codegen(CuTeConstexprArg("c", str))
        with self.assertRaises(RuntimeError):
            cg.torch_cpp_param()
        with self.assertRaises(RuntimeError):
            cg.torch_schema_param(0)


class GuardTest(unittest.TestCase):
    def test_tensor_dtype_guard(self) -> None:
        cg = make_codegen(CuTeTensorArg("x"))
        out = cg.guard("op", SimpleNamespace(dtype="torch.float32"))
        self.assertIn("x.scalar_type() != torch::headeronly::ScalarType::Float", out)

    def test_tensor_unsupported_dtype_raises(self) -> None:
        cg = make_codegen(CuTeTensorArg("x"))
        with self.assertRaises(RuntimeError):
            cg.guard("op", SimpleNamespace(dtype="torch.complex64"))

    def test_scalar_i32_range_guard(self) -> None:
        cg = make_codegen(CuTeScalarArg("n", "i32"))
        self.assertIn("std::numeric_limits<int32_t>", cg.guard("op", 0))

    def test_scalar_non_i32_has_no_guard(self) -> None:
        self.assertEqual(make_codegen(CuTeScalarArg("n", "i64")).guard("op", 0), "")

    def test_constexpr_value_guard(self) -> None:
        out = make_codegen(CuTeConstexprArg("flag", bool)).guard("op", True)
        self.assertIn("flag != true", out)


class ConstexprCodegenTest(unittest.TestCase):
    def test_not_runtime_contributes_nothing_to_entry(self) -> None:
        cg = make_codegen(CuTeConstexprArg("flag", bool))
        self.assertFalse(cg.is_runtime)
        self.assertEqual(cg.entry_args({}), [])
        self.assertIsNone(cg.wrapper_arg())

    def test_cute_arg_passthrough(self) -> None:
        self.assertIs(make_codegen(CuTeConstexprArg("flag", bool)).cute_arg(True), True)
        self.assertEqual(make_codegen(CuTeConstexprArg("n", int)).cute_arg(7), 7)


class ValidateTest(unittest.TestCase):
    def test_clean_op_passes(self) -> None:
        # A well-formed op validates without raising (returns None).
        self.assertIsNone(
            validate_cutedsl_op(
                [
                    make_codegen(CuTeTensorArg("x")),
                    make_codegen(CuTeScalarArg("n", "i32")),
                ],
                "op",
                _PREFIX,
            )
        )

    def test_bad_scalar_dtype_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_cutedsl_op(
                [make_codegen(CuTeScalarArg("n", "f16"))], "op", _PREFIX
            )

    def test_bad_identifier_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_cutedsl_op([], "op", "bad-prefix")


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


class GenerateContentTest(unittest.TestCase):
    """Exercise the full sidecar/torch-op string generators (CPU, no compile)."""

    def test_sidecar_entry_content(self) -> None:
        codegens = [
            make_codegen(CuTeTensorArg("x")),
            make_codegen(CuTeScalarArg("n", "i32")),
        ]
        out = generate_sidecar_entry_content("op", codegens, "pfx", {"x": (2, 0)})
        self.assertIn("int32_t pfx_entry(", out)
        self.assertIn("void* x_data", out)
        self.assertIn("int32_t n", out)
        self.assertIn("cute_dsl_pfx_wrapper(&module,", out)
        self.assertIn('#include "pfx.h"', out)
        # The op name is rendered into the OP_NAME region; the static cudart
        # shims come verbatim from the template into the rendered output.
        self.assertIn('#define CUTEDSL_OP_NAME "op"', out)
        self.assertIn("cudaLibraryLoadData", out)
        self.assertIn("cutedsl_cudart_symbol", out)

    def test_torch_op_content(self) -> None:
        codegens = [
            make_codegen(CuTeTensorArg("x")),
            make_codegen(CuTeScalarArg("n", "i32")),
        ]
        values = {"x": SimpleNamespace(dtype="torch.float32"), "n": 0}
        out = generate_torch_op_content(
            "op", codegens, values, "pfx", "impl.so", {"x": (2, 0)}
        )
        self.assertIn("m.def(CUTEDSL_OP_SCHEMA)", out)
        self.assertIn('#define CUTEDSL_OP_SCHEMA "op(', out)
        self.assertIn("torch::stable::Tensor x", out)
        self.assertIn("STABLE_TORCH_LIBRARY_IMPL(triton_aot, CUDA, m)", out)
        self.assertIn("TORCH_BOX(&cutedsl_op)", out)
        # Per-op name/symbol/sidecar are rendered into the OP_NAME macro region;
        # the static dladdr loader comes verbatim from the template. Only the
        # basename is baked in (no absolute build path).
        self.assertIn('#define CUTEDSL_OP_NAME "op"', out)
        self.assertIn('#define CUTEDSL_ENTRY_SYMBOL "pfx_entry"', out)
        self.assertIn('#define CUTEDSL_SIDECAR_NAME "impl.so"', out)
        self.assertIn("dladdr", out)
        self.assertNotIn("/tmp", out)
        self.assertIn("unexpected dtype for x", out)

    def test_torch_op_content_with_constexpr(self) -> None:
        codegens = [
            make_codegen(CuTeTensorArg("x")),
            make_codegen(CuTeConstexprArg("flag", bool)),
        ]
        values = {"x": SimpleNamespace(dtype="torch.float32"), "flag": True}
        out = generate_torch_op_content(
            "op", codegens, values, "pfx", "impl.so", {"x": (2, 0)}
        )
        self.assertIn("bool flag", out)
        self.assertIn("flag != true", out)


class FormatConstexprValueTest(unittest.TestCase):
    def test_bool(self) -> None:
        self.assertEqual(_format_constexpr_value(True), "true")
        self.assertEqual(_format_constexpr_value(False), "false")

    def test_str_is_quoted(self) -> None:
        self.assertEqual(_format_constexpr_value("abc"), '"abc"')

    def test_number(self) -> None:
        self.assertEqual(_format_constexpr_value(7), "7")


class ArgCodegenBaseTest(unittest.TestCase):
    """The base class supplies constexpr-friendly defaults; the torch/cute hooks
    are abstract and every concrete kind overrides them."""

    def test_defaults(self) -> None:
        cg: ArgCodegen[CuTeTensorArg] = ArgCodegen(CuTeTensorArg("x"))
        self.assertEqual(cg.entry_args({}), [])
        self.assertEqual(cg.struct_build_lines("pfx", {}), [])
        self.assertIsNone(cg.wrapper_arg())
        self.assertEqual(cg.guard("op", None), "")
        self.assertIsNone(cg.validate("op"))

    def test_abstract_hooks_raise(self) -> None:
        cg: ArgCodegen[CuTeTensorArg] = ArgCodegen(CuTeTensorArg("x"))
        with self.assertRaises(NotImplementedError):
            cg.torch_cpp_param()
        with self.assertRaises(NotImplementedError):
            cg.torch_schema_param(0)
        with self.assertRaises(NotImplementedError):
            cg.cute_arg(1)


class CompileCuTeDSLToCppTest(unittest.TestCase):
    """Orchestrator path up to cute.compile, with cutlass + stream glue stubbed.

    Exercises resolve_runtime_args + build_cute_call_args wiring on CPU; the real
    cute.compile/export_to_c are GPU-only and covered by the E2E test.
    """

    def test_builds_cute_args_then_calls_compile(self) -> None:
        class _StopAtCompile(Exception):
            pass

        op = SimpleNamespace(
            name="_cutedsl_vector_add",
            arg_specs=[CuTeTensorArg("x"), CuTeScalarArg("n", "i32")],
            jit_fn=lambda *a: None,
        )
        spec = SimpleNamespace(runtime_args=("X", 4), runtime_kwargs={})

        fake_cute = pytypes.ModuleType("cutlass.cute")
        # pyre-ignore[16]: attaching a stub attribute to a fake module.
        fake_cute.compile = MagicMock(side_effect=_StopAtCompile)
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
                cutedsl_mod, "build_cute_call_args", return_value=("cute_arg",)
            ) as build_args,
        ):
            with self.assertRaises(_StopAtCompile):
                # pyre-ignore[6]: op/specs are minimal stubs for this codegen path.
                compile_cutedsl_to_cpp(op, [spec], install_dir, "_cutedsl_vector_add")

        build_args.assert_called_once_with(op.arg_specs, {"x": "X", "n": 4})
        fake_cute.compile.assert_called_once_with(op.jit_fn, "cute_arg")
