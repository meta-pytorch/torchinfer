# Copyright (c) Meta Platforms, Inc. and affiliates.


import hashlib
from typing import Any, Callable, cast, Optional

# @dep=fbsource//third-party/pypi/torch:torch
# @manual=//triton:triton
from triton.runtime.autotuner import Autotuner, Config
from triton.runtime.jit import JITFunction, KernelInterface

# One Triton ``Autotuner.cache``: autotune key tuple -> best ``Config``.
# AOT-T overrides are these caches keyed by kernel name.
AutotuneCache = dict[tuple[Any, ...], Config]


def _is_class_alias(obj: Any, *, qualname: str, module_suffix: str) -> bool:
    """True iff *obj* is an instance of a class whose ``__name__`` equals
    *qualname* AND whose ``__module__`` ends with *module_suffix*, walking
    the full MRO so subclasses pass.
    """
    return any(
        cls.__name__ == qualname and cls.__module__.endswith(module_suffix)
        for cls in type(obj).__mro__
    )


def is_autotuner(obj: Any) -> bool:
    """Whether *obj* is an ``Autotuner`` (any module-path alias of it).

    PPO autotuner is 'intern' class, can't rely on isinstance.
    """
    return _is_class_alias(
        obj, qualname="Autotuner", module_suffix="triton.runtime.autotuner"
    )


def unwrap_to_jit(fn: Any) -> JITFunction[Callable[..., Any]]:
    """Peel ``KernelInterface`` layers (Heuristics / Autotuner / ...) until
    landing on the underlying ``JITFunction``. Raises if not reachable."""
    while isinstance(fn, KernelInterface) and not isinstance(fn, JITFunction):
        # pyre-ignore[16]: KernelInterface subclasses all define ``fn``.
        fn = fn.fn
    if not isinstance(fn, JITFunction):
        raise TypeError(
            f"Cannot unwrap to JITFunction; got {type(fn).__name__}: {fn!r}"
        )
    return fn


def try_get_autotuner(fn: Any) -> Optional[Autotuner]:
    """Return the outermost ``Autotuner`` wrapping *fn*, or ``None`` if the
    kernel is not autotuned.
    """
    while isinstance(fn, KernelInterface):
        if is_autotuner(fn):
            return cast(Autotuner, fn)
        # pyre-ignore[16]: KernelInterface subclasses all define ``fn``.
        fn = fn.fn
    return None


def kernel_param_names(fn: Any) -> set[str]:
    """Kernel JIT signature parameter names (unwraps autotune/heuristics).

    Anything in ``cfg.kwargs`` not in this set is a backend opt.
    """
    return set(unwrap_to_jit(fn).arg_names)


def cfg_constexpr_keys(fn: Any, cfg: Any) -> list[str]:
    """Subset of ``cfg.kwargs`` keys that are kernel constexprs of *fn*
    (i.e. appear in its signature), preserving original ``cfg.kwargs`` order.

    The complement are backend opts (e.g. AMD ``matrix_instr_nonkdim``) and
    flow through ``AutotuneAttrs`` instead — see ``AutotuneAttrs.from_cfg``.
    """
    params = kernel_param_names(fn)
    return [k for k in cfg.kwargs if k in params]


def hash_kernel_name(kernel_name: str) -> str:
    """Hash kernel name to create shorter, filesystem-safe names.

    Args:
        kernel_name: Full kernel name (can be very long with specialization suffixes).
            e.g., "_addmm_fwd_sm80_pfp32_pfp32_pfp32_pfp32_i32_..."

    Returns:
        Hashed name in format "kernel_<hex>".
            e.g., "kernel_a1b2c3d4e5f6..."

    """
    # Non-crypto: just shortens a long kernel name into a filesystem-safe token.
    digest = hashlib.blake2b(kernel_name.encode("utf-8"), digest_size=20).hexdigest()
    return "kernel_" + digest
