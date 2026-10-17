# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import os
from collections.abc import Sequence

from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from aot_tensor.build.extension_builder_base import SoBuildExtension
from aot_tensor.build.gpu_backend import is_amd
from aot_tensor.build.triton.amd_extension_builder import AmdExtensionBuilder
from setuptools import Extension, setup

# The .so exports nothing: the torch op registers from a static initializer, and
# hiding the launcher and mgpu* symbols keeps two FlyDSL .so files -- or the
# eager FlyDSL runtime -- in one process from binding to each other's.
_VERSION_SCRIPT = "{ local: *; };\n"


class FlydslExtensionBuilder(AmdExtensionBuilder):
    """Links the generated torch op with the exported FlyDSL launcher objects
    (one per specialization).

    Reuses the AMD builder's ROCm/torch include and flag resolution; ``build``
    is overridden because the inherited one embeds Triton hsaco files, which a
    FlyDSL launcher object already carries.
    """

    def __init__(
        self,
        source_dir: str,
        kernel_name: str,
        output_dir: str,
        objects: Sequence[str],
        build_config: ExtensionBuildConfig | None = None,
    ) -> None:
        super().__init__(source_dir, kernel_name, output_dir, build_config)
        self.objects: list[str] = list(objects)

    def get_extra_compile_args(self) -> list[str]:
        # SoBuildExtension drops sysconfig CFLAGS, so without this the op body
        # (the per-call host path) builds at -O0.
        return [*super().get_extra_compile_args(), "-O3"]

    def build(self) -> str:
        torch_op_file = os.path.join(
            self.source_dir, f"{self.kernel_name}_torch_op.cpp"
        )
        if not self.objects:
            raise ValueError("FlyDSL AOT build needs at least one launcher object")
        for path in (torch_op_file, *self.objects):
            if not os.path.exists(path):
                raise FileNotFoundError(f"FlyDSL AOT build input missing: {path}")
        build_temp = os.path.join(self.output_dir, "build_temp")
        os.makedirs(build_temp, exist_ok=True)
        version_script = os.path.join(build_temp, f"{self.ext_name}.map")
        with open(version_script, "w") as fp:
            fp.write(_VERSION_SCRIPT)

        extension = Extension(
            name=self.ext_name,
            sources=[torch_op_file],
            include_dirs=self.get_gpu_include_dirs() + self.get_torch_include_dirs(),
            library_dirs=self.get_gpu_library_dirs(),
            libraries=self.get_libraries(),
            extra_compile_args=self.get_extra_compile_args(),
            extra_objects=self.objects,
            extra_link_args=[f"-Wl,--version-script={version_script}"],
            language="c++",
        )
        script_args = [
            "build_ext",
            f"--build-lib={self.output_dir}",
            f"--build-temp={build_temp}",
        ]
        if self.build_config.compiler_path is not None:
            script_args.append(f"--compiler-path={self.build_config.compiler_path}")

        setup(
            name=self.ext_name,
            ext_modules=[extension],
            script_args=script_args,
            cmdclass={"build_ext": SoBuildExtension},
        )
        output = os.path.join(self.output_dir, f"{self.ext_name}.so")
        if not os.path.exists(output):
            raise RuntimeError(f"Expected built FlyDSL extension: {output}")
        return output


def build_flydsl_aot_extension(
    source_dir: str,
    kernel_name: str,
    output_dir: str,
    objects: Sequence[str],
    build_config: ExtensionBuildConfig | None = None,
) -> str:
    if not is_amd():
        raise NotImplementedError("FlyDSL AOT is only supported on AMD ROCm.")
    return FlydslExtensionBuilder(
        source_dir, kernel_name, output_dir, objects, build_config=build_config
    ).build()
