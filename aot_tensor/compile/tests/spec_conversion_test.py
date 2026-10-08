# Copyright (c) Meta Platforms, Inc. and affiliates.


"""Unit tests for aot_tensor.compile.spec_conversion module."""

import unittest

from aot_tensor.compile.spec_conversion import (
    collect_constraints,
    ConstantValue,
    constexpr,
    extract_constants,
    get_fp8_replacement_signature_for_amd,
    get_fp8_replacement_signature_for_sm80,
    make_instance_descriptor,
    signature_list_to_dict,
    SignatureConstraints,
    SignatureElement,
)
from parameterized import parameterized


class ConstexprTest(unittest.TestCase):
    """Tests for constexpr()."""

    @parameterized.expand(
        [
            # compile-time constants -> the constant itself
            ("int_string", "123", 123),
            ("bare_int", 128, 128),
            ("float_string", "1.5", 1.5),
            ("bool_true", True, True),
            ("bool_false", False, False),
            ("non_dtype_string", "leaky_relu", "leaky_relu"),
            ("none_value", None, None),
            # tuple -> its first element is evaluated
            ("tuple_first_element", ("not_a_dtype", 16), "not_a_dtype"),
            # dtypes / pointers are not constants -> None
            ("dtype_i32", "i32", None),
            ("dtype_fp32", "fp32", None),
            ("dtype_bf16", "bf16", None),
            ("dtype_fp16", "fp16", None),
            ("pointer_fp32", "*fp32", None),
            ("pointer_bf16", "*bf16", None),
        ]
    )
    def test_constexpr(
        self, _name: str, input_val: SignatureElement, expected: ConstantValue
    ) -> None:
        self.assertEqual(constexpr(input_val), expected)


class CollectConstraintsTest(unittest.TestCase):
    """Tests for collect_constraints()."""

    def test_empty_signature_returns_empty_sets(self) -> None:
        """Empty signature produces empty constraint sets."""
        result = collect_constraints([])
        self.assertEqual(result.divisible_by_16, set())
        self.assertEqual(result.divisible_by_8, set())
        self.assertEqual(result.equal_to_1, set())
        self.assertEqual(result.none_args, set())
        self.assertEqual(result.optional_args, set())
        self.assertFalse(result.has_fp8)

    @parameterized.expand(
        [
            # (name, element, in_div16, in_div8, in_equal_1, in_none)
            ("aligned_16", ("*fp32", 16), True, True, False, False),
            ("aligned_32", ("*fp32", 32), True, True, False, False),
            ("aligned_8", ("*fp32", 8), False, True, False, False),
            ("value_4", ("*fp32", 4), False, False, False, False),
            ("value_1", ("*fp32", 1), False, False, True, False),
            ("value_none", ("i32", None), False, False, False, True),
        ]
    )
    def test_two_tuple_constraints(
        self,
        _name: str,
        element: SignatureElement,
        in_div16: bool,
        in_div8: bool,
        in_eq1: bool,
        in_none: bool,
    ) -> None:
        """2-tuple (dtype, value) signature elements."""
        result = collect_constraints([element])
        self.assertEqual(0 in result.divisible_by_16, in_div16)
        self.assertEqual(0 in result.divisible_by_8, in_div8)
        self.assertEqual(0 in result.equal_to_1, in_eq1)
        self.assertEqual(0 in result.none_args, in_none)

    @parameterized.expand(
        [
            # (name, element, in_optional, in_none, in_div16)
            ("has_value_true", ("*fp32", 16, True), True, False, True),
            ("has_value_false", ("*fp32", 16, False), True, True, False),
        ]
    )
    def test_three_tuple_optional_args(
        self,
        _name: str,
        element: SignatureElement,
        in_optional: bool,
        in_none: bool,
        in_div16: bool,
    ) -> None:
        """3-tuple (optional arg) signature elements."""
        result = collect_constraints([element])
        self.assertEqual(0 in result.optional_args, in_optional)
        self.assertEqual(0 in result.none_args, in_none)
        self.assertEqual(0 in result.divisible_by_16, in_div16)

    @parameterized.expand(
        [
            ("fp8e4nv_tuple", [("*fp8e4nv", 16)], True),
            ("fp8e4b8_tuple", [("*fp8e4b8", 16)], True),
            ("fp8e4nv_string", ["*fp8e4nv"], True),
            ("no_fp8", [("*fp32", 16), ("*bf16", 16)], False),
        ]
    )
    def test_has_fp8(
        self, _name: str, signature: list[SignatureElement], expected: bool
    ) -> None:
        """has_fp8 flags FP8 dtypes anywhere in the signature."""
        self.assertEqual(collect_constraints(signature).has_fp8, expected)

    def test_string_element(self) -> None:
        """Plain string elements have no divisibility constraints."""
        signature: list[SignatureElement] = ["*fp32", "i32"]
        result = collect_constraints(signature)
        self.assertEqual(result.divisible_by_16, set())
        self.assertEqual(result.divisible_by_8, set())

    def test_bare_integer_element(self) -> None:
        """Bare integer element uses value for divisibility check."""
        signature: list[SignatureElement] = [128]
        result = collect_constraints(signature)
        self.assertIn(0, result.divisible_by_8)


class MakeInstanceDescriptorTest(unittest.TestCase):
    """Tests for make_instance_descriptor()."""

    def test_ids_of_folded_args_is_union(self) -> None:
        """ids_of_folded_args is the union of equal_to_1 and none_args."""
        constraints = SignatureConstraints(
            divisible_by_16=set(),
            divisible_by_8=set(),
            equal_to_1={0, 1},
            none_args={1, 2},
            optional_args=set(),
            has_fp8=False,
        )
        result = make_instance_descriptor(constraints)
        # overlap (1 in both) proves it's a union, not concat/intersection
        self.assertEqual(result[0].ids_of_folded_args, {0, 1, 2})


class ExtractConstantsTest(unittest.TestCase):
    """Tests for extract_constants() (its own logic, not constexpr)."""

    def test_equal_to_1_args_get_value_1(self) -> None:
        signature: list[SignatureElement] = [("*fp32", 1)]
        constraints = SignatureConstraints(
            divisible_by_16=set(),
            divisible_by_8=set(),
            equal_to_1={0},
            none_args=set(),
            optional_args=set(),
            has_fp8=False,
        )
        result = extract_constants(signature, constraints)
        self.assertEqual(result[0], 1)

    def test_none_args_get_value_none(self) -> None:
        signature: list[SignatureElement] = [("*fp32", None)]
        constraints = SignatureConstraints(
            divisible_by_16=set(),
            divisible_by_8=set(),
            equal_to_1=set(),
            none_args={0},
            optional_args=set(),
            has_fp8=False,
        )
        result = extract_constants(signature, constraints)
        self.assertIn(0, result)
        self.assertIsNone(result[0])

    def test_constexpr_results_included(self) -> None:
        signature: list[SignatureElement] = [("*fp32", 16), 128]
        constraints = SignatureConstraints(
            divisible_by_16={0},
            divisible_by_8={0},
            equal_to_1=set(),
            none_args=set(),
            optional_args=set(),
            has_fp8=False,
        )
        result = extract_constants(signature, constraints)
        self.assertIn(1, result)
        self.assertEqual(result[1], 128)


class SignatureListToDictTest(unittest.TestCase):
    """Tests for signature_list_to_dict()."""

    def test_excludes_constant_indices(self) -> None:
        signature: list[SignatureElement] = [("*fp32", 16), "const", ("i32", None)]
        constants: dict[int, ConstantValue] = {1: "const", 2: None}
        result = signature_list_to_dict(signature, constants)
        self.assertEqual(result, {0: "*fp32"})

    @parameterized.expand(
        [
            ("two_tuple", [("*fp32", 16)], {0: "*fp32"}),
            ("three_tuple", [("*bf16", 8, True)], {0: "*bf16"}),
            ("string", ["*fp32"], {0: "*fp32"}),
        ]
    )
    def test_dtype_extraction(
        self, _name: str, signature: list[SignatureElement], expected: dict[int, str]
    ) -> None:
        self.assertEqual(signature_list_to_dict(signature, {}), expected)

    @parameterized.expand(
        [
            ("empty_signature", [], {}),
            ("all_constants", ["const1", 128], {0: "const1", 1: 128}),
        ]
    )
    def test_empty_output(
        self,
        _name: str,
        signature: list[SignatureElement],
        constants: dict[int, ConstantValue],
    ) -> None:
        self.assertEqual(signature_list_to_dict(signature, constants), {})


class GetFP8ReplacementTest(unittest.TestCase):
    """Tests for get_fp8_replacement_signature_for_amd/sm80."""

    @parameterized.expand(
        [
            # AMD: fp8e4nv -> fp8e4b8 on MI300X (gfx942)
            ("amd_mi300x_cc_format", {"94"}, "*fp8e4nv", "*fp8e4b8"),
            ("amd_mi300x_gfx_format", {"gfx942"}, "*fp8e4nv", "*fp8e4b8"),
            # AMD: fp8e4nv kept on MI350X (gfx950)
            ("amd_mi350x_cc_format", {"95"}, "*fp8e4nv", "*fp8e4nv"),
            ("amd_mi350x_gfx_format", {"gfx950"}, "*fp8e4nv", "*fp8e4nv"),
            # AMD: fp8e4b8 -> fp8e4nv on MI350X
            ("amd_mi350x_reverse", {"gfx950"}, "*fp8e4b8", "*fp8e4nv"),
            # non-FP8 unchanged
            ("amd_non_fp8", {"gfx942"}, "*fp32", "*fp32"),
        ]
    )
    def test_amd_replacement(
        self, _name: str, cc: set[str], input_dtype: str, expected: str
    ) -> None:
        spec = {"signature": {0: input_dtype}}
        result = get_fp8_replacement_signature_for_amd(spec, cc)
        self.assertEqual(result[0], expected)

    @parameterized.expand(
        [
            ("fp8e4nv_to_bf16", "*fp8e4nv", "*bf16"),
            ("non_pointer_fp8", "fp8e4nv", "bf16"),
            ("non_fp8_unchanged", "*fp32", "*fp32"),
        ]
    )
    def test_sm80_replacement(
        self, _name: str, input_dtype: str, expected: str
    ) -> None:
        spec = {"signature": {0: input_dtype}}
        result = get_fp8_replacement_signature_for_sm80(spec)
        self.assertEqual(result[0], expected)


if __name__ == "__main__":
    unittest.main()
