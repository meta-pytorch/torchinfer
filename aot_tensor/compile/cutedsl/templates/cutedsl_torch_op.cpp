// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda.h>
#include <dlfcn.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

// __CUTEDSL_AOT_GENERATE_BEGIN__ OP_NAME
#define CUTEDSL_OP_NAME "placeholder"
#define CUTEDSL_ENTRY_SYMBOL "placeholder_entry"
#define CUTEDSL_SIDECAR_NAME "placeholder_cutedsl_impl.so"
#define CUTEDSL_OP_SCHEMA "placeholder(Tensor x) -> ()"
// __CUTEDSL_AOT_GENERATE_END__ OP_NAME

namespace {

// __CUTEDSL_AOT_GENERATE_BEGIN__ ENTRY_TYPE
using EntryFn = int32_t (*)(void*);
// __CUTEDSL_AOT_GENERATE_END__ ENTRY_TYPE

CUstream cutedsl_aot_get_current_stream() {
  auto device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  void* stream_ptr = nullptr;
  if (aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr) != 0) {
    throw std::runtime_error(
        "CuTeAOT " CUTEDSL_OP_NAME ": failed to get current CUDA stream");
  }
  return reinterpret_cast<CUstream>(stream_ptr);
}

// Resolve the sidecar next to this .so at load time via dladdr, instead of
// baking in the absolute build-time path, so the package stays relocatable.
std::string cutedsl_sidecar_path() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&cutedsl_sidecar_path), &info) == 0 ||
      info.dli_fname == nullptr) {
    throw std::runtime_error(
        "CuTeAOT " CUTEDSL_OP_NAME
        ": dladdr could not locate the torch op .so");
  }
  std::string self_path(info.dli_fname);
  std::string::size_type slash = self_path.find_last_of('/');
  std::string dir = slash == std::string::npos ? std::string(".")
                                               : self_path.substr(0, slash);
  return dir + "/" + CUTEDSL_SIDECAR_NAME;
}

EntryFn cutedsl_load_entry() {
  static EntryFn fn = []() -> EntryFn {
    std::string sidecar_path = cutedsl_sidecar_path();
    void* handle = dlopen(sidecar_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char* err = dlerror();
      throw std::runtime_error(
          std::string("CuTeAOT " CUTEDSL_OP_NAME ": dlopen(") + sidecar_path +
          ") failed: " + (err == nullptr ? "unknown" : err));
    }
    dlerror();
    void* sym = dlsym(handle, CUTEDSL_ENTRY_SYMBOL);
    const char* err = dlerror();
    if (err != nullptr || sym == nullptr) {
      throw std::runtime_error(
          std::string("CuTeAOT " CUTEDSL_OP_NAME ": dlsym failed: ") +
          (err == nullptr ? "unknown" : err));
    }
    return reinterpret_cast<EntryFn>(sym);
  }();
  return fn;
}

// __CUTEDSL_AOT_GENERATE_BEGIN__ OP_FN
void cutedsl_op() {}
void cutedsl_dummy_op() {}
// __CUTEDSL_AOT_GENERATE_END__ OP_FN

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(triton_aot, m) {
  m.def(CUTEDSL_OP_SCHEMA);
}

STABLE_TORCH_LIBRARY_IMPL(triton_aot, CUDA, m) {
  m.impl(CUTEDSL_OP_NAME, TORCH_BOX(&cutedsl_op));
}

STABLE_TORCH_LIBRARY_IMPL(triton_aot, CPU, m) {
  m.impl(CUTEDSL_OP_NAME, TORCH_BOX(&cutedsl_dummy_op));
}

STABLE_TORCH_LIBRARY_IMPL(triton_aot, Meta, m) {
  m.impl(CUTEDSL_OP_NAME, TORCH_BOX(&cutedsl_dummy_op));
}
