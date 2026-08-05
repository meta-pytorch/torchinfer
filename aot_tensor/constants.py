# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

TRITON: str = "triton"
CUTEDSL: str = "cutedsl"

# Filename of the per-kernel torch-op schema JSON.
# Details: https://fburl.com/code/v38eznnc
AOTT_OP_SCHEMAS_FILE_NAME: str = "aott_op_schemas.json"

# Marks compiler output so tooling skips it -- linters cannot resolve
# compilation flags for generated C++ (no buck target owns it) and would
# otherwise report that on every diff. Assembled at runtime so the modules
# emitting it are not themselves treated as generated.
GENERATED_TOKEN: str = "@" + "generated"


def generated_header(comment: str) -> str:
    """Header marking a file as compiler output, using *comment* as the prefix."""
    return (
        f"{comment} {GENERATED_TOKEN} by AOT-T codegen\n"
        f"{comment} Do not edit; regenerate by recompiling the kernel.\n"
    )
