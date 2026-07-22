# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
"""Locate the shared launch core header (``launch.h``) for the AOT-T build.

Single source of truth for the candidate lookup so the codegen-time predicate
(``codegen._launch_header_available``) and the build-time vendoring action
(``NvidiaExtensionBuilder._vendor_launch_header``) always search the same paths
and therefore agree on whether the Level-1 launcher can be emitted.
"""

import os
from typing import Optional


def find_launch_header() -> Optional[str]:
    """Return the path to ``launch.h`` if it can be found, else ``None``.

    The Level-1 ``launcher_src`` calls ``triton_launch_<name>()`` declared in
    ``launch.h``; when the header is not shippable (e.g. AOT-T compiled against a
    triton build that does not ship it), callers fall back to the legacy
    launcher rather than emit an unresolvable ``#include``.
    """
    candidates = []
    try:
        import triton.backends.nvidia as _nv_backend

        candidates.append(
            os.path.join(os.path.dirname(_nv_backend.__file__), "launch.h")
        )
    except ImportError:
        pass
    try:
        import triton

        candidates.append(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(triton.__file__))),
                "triton",
                "runtime",
                "launch.h",
            )
        )
    except ImportError:
        pass
    return next((p for p in candidates if os.path.exists(p)), None)
