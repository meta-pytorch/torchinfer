# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""Load and render the per-DSL C++ code templates.

A ``TemplateSet`` bundles a DSL's resource package with its generate-marker
tag; rendering swaps each marked region for generated code.
"""

import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources

from aot_tensor.constants import generated_header


def render_template(
    template: str,
    replacements: dict[str, str],
    tag: str = "TRITON_AOT",
) -> str:
    """Fill a template's ``// __<tag>_GENERATE_{BEGIN,END}__ NAME`` regions.

    Each NAME must have exactly one BEGIN/END pair, and the keys must match
    ``replacements`` exactly. Markers are left in place to ease debugging.
    Raises AssertionError on duplicate, mismatched, or unknown markers.
    """
    BEGIN_PREFIX = f"// __{tag}_GENERATE_BEGIN__ "
    END_PREFIX = f"// __{tag}_GENERATE_END__ "

    begin_keys = re.findall(rf"// __{re.escape(tag)}_GENERATE_BEGIN__ (\w+)", template)
    end_keys = re.findall(rf"// __{re.escape(tag)}_GENERATE_END__ (\w+)", template)

    # Check for duplicate keys
    begin_key_counts = Counter(begin_keys)
    end_key_counts = Counter(end_keys)
    for key, count in begin_key_counts.items():
        assert count == 1, f"Duplicate BEGIN marker for key: {key}"
    for key, count in end_key_counts.items():
        assert count == 1, f"Duplicate END marker for key: {key}"

    # Check BEGIN and END keys match
    template_keys = set(begin_keys)
    assert template_keys == set(end_keys), (
        f"Mismatched BEGIN/END markers: BEGIN={template_keys}, END={set(end_keys)}"
    )

    # Validate keys match between template and replacements
    replacement_keys = set(replacements.keys())
    assert template_keys == replacement_keys, (
        f"Keys mismatch: in template but not in replacements: {template_keys - replacement_keys}, "
        f"in replacements but not in template: {replacement_keys - template_keys}"
    )

    # Do the replacements
    result = template
    for key, value in replacements.items():
        begin_marker = f"{BEGIN_PREFIX}{key}"
        end_marker = f"{END_PREFIX}{key}"

        begin_idx = result.find(begin_marker)
        newline_idx = result.find("\n", begin_idx)
        assert newline_idx != -1, (
            f"BEGIN marker for key '{key}' must be followed by newline"
        )
        content_start = newline_idx + 1
        end_idx = result.find(end_marker, begin_idx)

        result = result[:content_start] + value + result[end_idx:]

    return result


@dataclass(frozen=True)
class TemplateSet:
    """A DSL's templates: resource ``package`` + generate-marker ``tag``.

    The two always go together, so callers pass one object. Use ``render`` to
    load and fill a template, or ``load`` for the raw text.
    """

    package: str
    tag: str

    def load(self, name: str) -> str:
        """Raw text of template ``name`` (Buck picks CUDA vs hipified HIP)."""
        return resources.files(self.package).joinpath(name).read_text()

    def render(self, name: str, replacements: dict[str, str]) -> str:
        """Load ``name`` and fill its ``// __<tag>_GENERATE_*__`` regions.

        The generated-file header is added here rather than in the templates,
        which are hand-written and should stay linted. Output is therefore a
        whole file, not a fragment to embed in one.
        """
        return generated_header("//") + render_template(
            self.load(name), replacements, self.tag
        )


TRITON_TEMPLATES: TemplateSet = TemplateSet(
    package="aot_tensor.compile.triton.templates", tag="TRITON_AOT"
)
CUTEDSL_TEMPLATES: TemplateSet = TemplateSet(
    package="triton_aot.compile.cutedsl.templates", tag="CUTEDSL_AOT"
)
