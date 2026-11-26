# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from aot_tensor.compile.triton.gen_hip_templates import generate, TEMPLATE_NAMES


def check_templates(
    templates_dir: Path, filenames: Iterable[str]
) -> list[dict[str, str | int]]:
    templates_dir = templates_dir.resolve()
    inputs = {templates_dir.parent / "gen_hip_templates.py"}
    inputs.update(templates_dir / name for name in TEMPLATE_NAMES)
    inputs.update(templates_dir / "hip" / name for name in TEMPLATE_NAMES)
    if inputs.isdisjoint(Path(filename).resolve() for filename in filenames):
        return []

    generated = generate(templates_dir)
    messages = []
    for name in TEMPLATE_NAMES:
        path = templates_dir / "hip" / name
        original = path.read_bytes().decode("utf-8") if path.exists() else ""
        replacement = generated[name]
        if original != replacement:
            messages.append(
                {
                    "path": str(path),
                    "line": 1,
                    "char": 1,
                    "code": "HIPTEMPLATES",
                    "severity": "warning",
                    "name": "format",
                    "original": original,
                    "replacement": replacement,
                    "description": "Regenerate HIP templates from the CUDA sources.",
                }
            )
    return messages


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--templates-dir", type=Path, required=True)
    parser.add_argument(
        "--severity",
        choices=("error", "warning", "advice", "disabled"),
        default="warning",
    )
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args(argv)
    for message in check_templates(args.templates_dir, args.filenames):
        message["severity"] = args.severity
        print(json.dumps(message), flush=True)


if __name__ == "__main__":
    main()
