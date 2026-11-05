# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Pure TritonCC spec metadata, independent of runtime Torch and autotuning."""

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import triton

SpecType = Union[Tuple[str, int], Tuple[str, int, bool], int, str]
NamedSpecType = Dict[str, SpecType]
SpecListType = List[SpecType]
_Function = TypeVar("_Function")


@dataclass
class VersionedSpec:
    """
    spec: a dict that maps each argument name to a spec
    version: the version of the spec
    num_warps: optional Triton compile-time num_warps override (None = default 4)
    num_stages: optional Triton compile-time num_stages override (None = default 3)
    """

    spec: NamedSpecType = dataclasses.field(default_factory=dict)
    version: str = ""
    default_values: Dict[str, Any] = dataclasses.field(default_factory=dict)
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None


@dataclass
class VersionedUnnamedSpec:
    """
    spec: a list of unnamed specs
    version: the version of the spec
    num_warps / num_stages: optional Triton compile-time overrides per spec.
    """

    spec: SpecListType = dataclasses.field(default_factory=list)
    version: str = ""
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"signature": self.spec}
        if self.num_warps is not None:
            out["num_warps"] = self.num_warps
        if self.num_stages is not None:
            out["num_stages"] = self.num_stages
        return out


def tritoncc_specs(
    versioned_specs: List[VersionedSpec],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    This is a decorator to register TritonCC specs for a Triton kernel
    """

    def decorator(fn: Any) -> Any:
        fn_arg_names = inspect.getfullargspec(fn).args
        specs: List[VersionedUnnamedSpec] = []

        for versioned_spec in versioned_specs:
            fn_arg_set = set(fn_arg_names)
            spec_key_set = set(versioned_spec.spec.keys())
            missing_in_spec = fn_arg_set - spec_key_set
            extra_in_spec = spec_key_set - fn_arg_set

            assert fn_arg_set == spec_key_set, (
                f"Spec keys mismatch. Expected: {sorted(fn_arg_names)}, Got: {sorted(versioned_spec.spec.keys())}. "
                + (f"Missing: {sorted(missing_in_spec)}. " if missing_in_spec else "")
                + (f"Extra: {sorted(extra_in_spec)}." if extra_in_spec else "")
            )
            spec: SpecListType = [versioned_spec.spec[n] for n in fn_arg_names]
            specs.append(
                VersionedUnnamedSpec(
                    spec=spec,
                    version=versioned_spec.version,
                    num_warps=versioned_spec.num_warps,
                    num_stages=versioned_spec.num_stages,
                )
            )
        fn.tritoncc_specs = specs
        fn.default_values = versioned_spec.default_values
        return fn

    return decorator


def unwrap_heuristic(func: Any) -> triton.runtime.jit.JITFunction:
    while not isinstance(func, triton.runtime.jit.JITFunction):
        func = func.fn
    return func


def register_tritoncc_specs(
    func: _Function, versioned_specs: List[VersionedSpec]
) -> _Function:
    decorator = tritoncc_specs(versioned_specs=versioned_specs)
    jit_func = unwrap_heuristic(func)
    assert isinstance(jit_func, triton.runtime.jit.JITFunction)
    jit_func.fn = decorator(jit_func.fn)
    return func
