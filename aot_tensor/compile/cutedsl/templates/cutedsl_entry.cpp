// __CUTEDSL_AOT_GENERATE_BEGIN__ HEADER_INCLUDE
#include "kernel.h"
// __CUTEDSL_AOT_GENERATE_END__ HEADER_INCLUDE

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

// __CUTEDSL_AOT_GENERATE_BEGIN__ OP_NAME
#define CUTEDSL_OP_NAME "placeholder"
// __CUTEDSL_AOT_GENERATE_END__ OP_NAME

namespace {

void* cutedsl_try_cudart(const std::string& path) {
  if (path.empty()) {
    return nullptr;
  }
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    return nullptr;
  }
  // Clear stale dynamic loader state before checking dlsym.
  dlerror();
  void* symbol = dlsym(handle, "cudaLibraryLoadData");
  const char* err = dlerror();
  if (err != nullptr || symbol == nullptr) {
    dlclose(handle);
    return nullptr;
  }
  return handle;
}

void* cutedsl_try_cudart_home(const char* cuda_home) {
  if (cuda_home == nullptr) {
    return nullptr;
  }
  return cutedsl_try_cudart(std::string(cuda_home) + "/lib64/libcudart.so.12");
}

void* cutedsl_cudart_handle() {
  static std::once_flag load_once;
  static void* handle = nullptr;
  static std::string load_error;
  std::call_once(load_once, []() {
    handle = cutedsl_try_cudart_home(std::getenv("CUDA_HOME"));
    if (handle == nullptr) {
      handle = cutedsl_try_cudart_home(std::getenv("CUDA_PATH"));
    }
    if (handle == nullptr) {
      handle = cutedsl_try_cudart("/usr/local/cuda/lib64/libcudart.so.12");
    }
    if (handle == nullptr) {
      handle = cutedsl_try_cudart("libcudart.so.12");
    }
    if (handle == nullptr) {
      load_error = "CuTeAOT " CUTEDSL_OP_NAME
                   ": failed to load libcudart with cudaLibrary* support";
    }
  });
  if (handle == nullptr) {
    throw std::runtime_error(load_error);
  }
  return handle;
}

template <typename Fn>
Fn cutedsl_cudart_symbol(const char* name) {
  // Clear stale dynamic loader state before checking dlsym.
  dlerror();
  void* symbol = dlsym(cutedsl_cudart_handle(), name);
  const char* err = dlerror();
  if (err != nullptr || symbol == nullptr) {
    throw std::runtime_error(
        std::string(
            "CuTeAOT " CUTEDSL_OP_NAME
            ": failed to load CUDA runtime symbol ") +
        name + ": " + (err == nullptr ? "unknown" : err));
  }
  return reinterpret_cast<Fn>(symbol);
}

} // namespace

extern "C" cudaError_t
_cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
  return cudaDeviceGetAttribute(value, attr, device);
}

extern "C" cudaError_t
_cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) {
  return cudaFuncSetAttribute(func, attr, value);
}

extern "C" cudaError_t _cudaGetDevice(int* device) {
  return cudaGetDevice(device);
}

extern "C" cudaError_t _cudaKernelSetAttributeForDevice(
    cudaKernel_t kernel,
    cudaFuncAttribute attr,
    int value,
    int device) {
  using Fn = cudaError_t (*)(cudaKernel_t, cudaFuncAttribute, int, int);
  static Fn fn = cutedsl_cudart_symbol<Fn>("cudaKernelSetAttributeForDevice");
  return fn(kernel, attr, value, device);
}

extern "C" cudaError_t _cudaLaunchKernelEx(
    const cudaLaunchConfig_t* config,
    const void* func,
    void** args) {
  using Fn = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
  static Fn fn = cutedsl_cudart_symbol<Fn>("cudaLaunchKernelExC");
  return fn(config, func, args);
}

extern "C" cudaError_t _cudaLibraryGetKernel(
    cudaKernel_t* kernel,
    cudaLibrary_t library,
    const char* name) {
  using Fn = cudaError_t (*)(cudaKernel_t*, cudaLibrary_t, const char*);
  static Fn fn = cutedsl_cudart_symbol<Fn>("cudaLibraryGetKernel");
  return fn(kernel, library, name);
}

extern "C" cudaError_t _cudaLibraryLoadData(
    cudaLibrary_t* library,
    const void* code,
    cudaJitOption* jit_options,
    void** jit_options_values,
    unsigned int num_jit_options,
    cudaLibraryOption* library_options,
    void** library_options_values,
    unsigned int num_library_options) {
  using Fn = cudaError_t (*)(
      cudaLibrary_t*,
      const void*,
      cudaJitOption*,
      void**,
      unsigned int,
      cudaLibraryOption*,
      void**,
      unsigned int);
  static Fn fn = cutedsl_cudart_symbol<Fn>("cudaLibraryLoadData");
  return fn(
      library,
      code,
      jit_options,
      jit_options_values,
      num_jit_options,
      library_options,
      library_options_values,
      num_library_options);
}

extern "C" CUresult _cuKernelGetAttribute(
    int* value,
    CUfunction_attribute attr,
    CUkernel kernel,
    CUdevice device) {
  return cuKernelGetAttribute(value, attr, kernel, device);
}

// __CUTEDSL_AOT_GENERATE_BEGIN__ ENTRY_FN
extern "C" int32_t placeholder_entry() {}
// __CUTEDSL_AOT_GENERATE_END__ ENTRY_FN
