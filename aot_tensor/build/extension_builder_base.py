# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
DSL- and hardware-agnostic base for AOT-T extension builders.

`ExtensionBuilder` holds the machinery shared by every builder (torch include
resolution, common compile args, the `.so`-naming `SoBuildExtension`); the GPU
toolkit specifics are abstract hooks. `CudaExtensionBuilder` fills those hooks in
for NVIDIA CUDA and is the parent for both the Triton-NVIDIA builder and the
CuTeDSL builder (both are CUDA-only). AMD/HIP and the Triton cubin-embedding
`build()` live in the per-DSL/per-HW subclasses.
"""

import abc
import logging
import os

from aot_tensor.build.extension_build_config import ExtensionBuildConfig
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDA_HOME


logger: logging.Logger = logging.getLogger(__name__)

# Path components for torch API headers (torch/csrc/api/include/torch/types.h)
# Used to locate bundled headers regardless of header_namespace setting
TORCH_API_INCLUDE_PATH_PARTS: tuple[str, ...] = ("torch", "csrc", "api", "include")

CPP_STANDARD: str = "c++20"

# Fallback CUDA toolkit root when neither CUDA_HOME nor CUDA_PATH is set.
DEFAULT_SYSTEM_CUDA_HOME: str = "/usr/local/cuda"

# Pin the torch stable-ABI feature version one below the newest so the .so loads
# on older runtimes.
# TODO make configurable and when changing sync StableAbiCheckTest,
# KNOWN_POST_BASELINE_BOXING_SHIMS etc
TORCH_TARGET_VERSION: str = "0x020C000000000000"  # 2.12.0, newest is 2.13


class SoBuildExtension(build_ext):
    """
    By default, setuptools generates .so files with platform-specific suffixes like:
        addmm_fwd.cpython-39-x86_64-linux-gnu.so
    This class overrides that behavior to produce simpler names:
        addmm_fwd.so

    We inherit from setuptools' build_ext directly (instead of PyTorch's BuildExtension),
    as we only compile C++ wrapper code, not CUDA kernels (cubins are pre-compiled by Triton)

    Uses setuptools' selected compiler unless an explicit override is supplied.
    """

    # Setuptools rejects script arguments not declared here. Callers omit this
    # optional flag to keep the compiler selected by setuptools.
    user_options = build_ext.user_options + [
        ("compiler-path=", None, "Explicit compiler executable"),
    ]

    compiler_path: str | None

    def initialize_options(self) -> None:
        super().initialize_options()
        self.compiler_path = None

    def get_ext_filename(self, fullname: str) -> str:
        return f"{fullname}.so"

    def build_extensions(self) -> None:
        """Apply an explicitly configured compiler before building extensions."""
        if self.compiler_path is not None:
            self.compiler.set_executables(
                compiler=self.compiler_path,
                compiler_so=self.compiler_path,
                compiler_cxx=self.compiler_path,
                linker_so=f"{self.compiler_path} -shared",
                linker_exe=self.compiler_path,
            )
            logger.info(f"Using compiler: {self.compiler_path}")
        super().build_extensions()


class ExtensionBuilder(abc.ABC):
    """Shared compile machinery (see module docstring). GPU-toolkit specifics are
    the abstract hooks below, filled in by each hardware subclass."""

    source_dir: str
    kernel_name: str
    output_dir: str
    ext_name: str
    gpu_toolkit_path: str
    build_config: ExtensionBuildConfig

    def __init__(
        self,
        source_dir: str,
        kernel_name: str,
        output_dir: str = "/tmp",
        build_config: ExtensionBuildConfig | None = None,
    ) -> None:
        """
        Initialize the extension builder.

        Args:
            source_dir: Directory containing the generated C++ sources and cubin/hsaco files.
            kernel_name: Name of the kernel (e.g., "_addmm_fwd").
            output_dir: Directory to place the built .so file.
            build_config: Optional extension-build overrides for the compiler
                executable, GPU toolkit root, Torch include roots, and GPU
                library directories.
        """
        self.source_dir = source_dir
        self.kernel_name = kernel_name
        self.output_dir = output_dir
        self.ext_name = kernel_name.lstrip("_")
        self.build_config = build_config or ExtensionBuildConfig()
        if self.build_config.gpu_toolkit_path is None:
            self.gpu_toolkit_path = self._default_gpu_toolkit_path()
        else:
            self.gpu_toolkit_path = self.build_config.gpu_toolkit_path

        if not os.path.exists(self.gpu_toolkit_path):
            raise RuntimeError(f"GPU toolkit not found at {self.gpu_toolkit_path}. ")

    # ---- abstract hardware hooks ----------------------------------------

    @abc.abstractmethod
    def _default_gpu_toolkit_path(self) -> str:
        """Return the default GPU toolkit path for this backend."""
        ...

    @abc.abstractmethod
    def get_gpu_include_dirs(self) -> list[str]:
        """Return GPU toolkit include directory paths, validated."""
        ...

    @abc.abstractmethod
    def get_gpu_library_dirs(self) -> list[str]:
        """Return directories containing the GPU runtime/driver libraries."""
        ...

    @abc.abstractmethod
    def get_libraries(self) -> list[str]:
        """Return GPU libraries to link against."""
        ...

    @abc.abstractmethod
    def get_torch_device_type(self) -> str:
        """Return the torch device type for include path lookup (e.g. cuda/hip)."""
        ...

    # ---- shared machinery -----------------------------------------------

    def get_extra_compile_args(self) -> list[str]:
        """Compiler args common to all backends; toolkit defines added by subclasses."""
        args = [
            f"-std={CPP_STANDARD}",
            "-fPIC",  # Position Independent Code, required for shared libraries
            # See TORCH_TARGET_VERSION.
            f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}",
            "-v",  # Verbose compiler output to help debug compilation issues
        ]
        return args

    def get_torch_include_dirs(self) -> list[str]:
        """
        Return torch include directories for C++ extension compilation.

        Includes standard torch headers and any explicitly supplied roots.
        """
        device_type = self.get_torch_device_type()
        candidates = list(cpp_extension.include_paths(device_type))

        for include_root in self.build_config.extra_torch_include_dirs:
            candidates.append(include_root)

            api_include = self._find_torch_api_include(include_root)
            if api_include is None:
                continue
            candidates.append(api_include)

            prefix = api_include
            for _ in range(len(TORCH_API_INCLUDE_PATH_PARTS)):
                prefix = os.path.dirname(prefix)
            if prefix != include_root:
                candidates.append(prefix)

        return [directory for directory in candidates if os.path.isdir(directory)]

    def _find_torch_api_include(self, include_root: str) -> str | None:
        api_include_suffix = os.path.join(*TORCH_API_INCLUDE_PATH_PARTS)
        for root, _, _ in os.walk(include_root):
            if root.endswith(api_include_suffix):
                return root
        return None


class CudaExtensionBuilder(ExtensionBuilder):
    """NVIDIA CUDA hardware backend hooks, shared by Triton-NVIDIA and CuTeDSL."""

    def _default_gpu_toolkit_path(self) -> str:
        if CUDA_HOME is None:
            raise RuntimeError(
                "CUDA_HOME is not set. Install CUDA toolkit or set CUDA_HOME environment variable."
            )
        return CUDA_HOME

    def _candidate_cuda_toolkits(self) -> list[str]:
        """Ordered CUDA toolkit roots to try: $CUDA_HOME, $CUDA_PATH, then the
        system default. Used to find a toolkit exposing a required API when the
        selected one does not."""
        candidates = [
            os.environ.get("CUDA_HOME"),
            os.environ.get("CUDA_PATH"),
            DEFAULT_SYSTEM_CUDA_HOME,
        ]
        return [c for c in candidates if c]

    def get_gpu_include_dirs(self) -> list[str]:
        """Return CUDA include directory path, validated."""
        include_dir = os.path.join(self.gpu_toolkit_path, "include")
        cuda_header = os.path.join(include_dir, "cuda.h")
        if not os.path.exists(cuda_header):
            raise RuntimeError(
                f"CUDA header not found at {cuda_header}. "
                f"CUDA Toolkit (not just Runtime) must be installed."
            )
        return [include_dir]

    def get_gpu_library_dirs(self) -> list[str]:
        """
        Return directories containing CUDA libraries (libcuda.so or stubs).

        Includes standard CUDA installation paths and any explicitly supplied
        library directories.
        """
        candidates = [
            *self.build_config.extra_gpu_library_dirs,
            os.path.join(self.gpu_toolkit_path, "lib64/stubs"),
            os.path.join(self.gpu_toolkit_path, "lib/stubs"),
            os.path.join(self.gpu_toolkit_path, "lib64"),
            os.path.join(self.gpu_toolkit_path, "lib"),
        ]

        result = [d for d in candidates if os.path.isdir(d)]
        if not result:
            raise RuntimeError(
                f"No CUDA library directories found in {self.gpu_toolkit_path}. Searched: {candidates}"
            )
        return result

    def get_libraries(self) -> list[str]:
        """Return CUDA libraries to link against."""
        return ["cuda"]

    def get_torch_device_type(self) -> str:
        """Return the torch device type for include path lookup."""
        return "cuda"

    def get_extra_compile_args(self) -> list[str]:
        """Add the CUDA define on top of the shared compile args."""
        args = super().get_extra_compile_args()
        args.append("-DUSE_CUDA")  # Makes shim.h CUDA declarations visible
        return args
