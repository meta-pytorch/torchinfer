# Copyright (c) Meta Platforms, Inc. and affiliates.


TRITON: str = "triton"
CUTEDSL: str = "cutedsl"

# Filename of the per-kernel torch-op schema JSON.
AOTT_OP_SCHEMAS_FILE_NAME: str = "aott_op_schemas.json"

# --- AOT-T operator namespace ------------------------------------------------
#
# The default operator namespace (`TritonCompileConfig.op_namespace`): with
# default options, `lower_model()` registers `torch.ops.aot_tensor.<kernel>`.
# Kernel libraries that define the SAME operator cannot both be loaded into one
# process (Dispatcher::registerDef hard-fails the second), so independently
# compiled artifacts need distinct namespaces.
#
# Internal model publishing never uses the prefix alone: it composes
# `<prefix>_f_<forward_method>_m_<merge>` (`_aott_op_namespace`), so each
# forward method of a multi-forward publish gets its own namespace.
#
# Renaming this prefix is safe because the namespace is baked into the `.so`
# and into the graph that calls it by the SAME lowering, and both travel in one
# artifact -- published models keep calling their own namespace against their
# own `.so`. Two constants deliberately do NOT follow it, because they record
# what LEGACY artifacts registered under: `_LEGACY_AOTT_DEFAULT_OP_NAMESPACE`
# and `_AOTT_DEFAULT_OP_NAMESPACE`. Nor do the `aot_triton_kernel_wrapper_*` FX
# leaf names, which are function names.
DEFAULT_OP_NAMESPACE_PREFIX: str = "aot_tensor"

# Delimit the two axes inside a composed namespace, and label which axis each
# component is so the name stays mechanically splittable.
AOTT_NS_FORWARD_TAG: str = "_f_"
AOTT_NS_MERGE_TAG: str = "_m_"

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
