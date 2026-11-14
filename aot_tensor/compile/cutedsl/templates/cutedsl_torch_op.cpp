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
#include <optional>
#include <stdexcept>
#include <string>

// __CUTEDSL_AOT_GENERATE_BEGIN__ OP_NAME
#define CUTEDSL_OP_NAMESPACE placeholder
#define CUTEDSL_OP_NAME "placeholder"
#define CUTEDSL_SIDECAR_NAME "placeholder_cutedsl_impl.so"
#define CUTEDSL_OP_SCHEMA "placeholder(Tensor x) -> ()"
// __CUTEDSL_AOT_GENERATE_END__ OP_NAME

namespace {

[[noreturn]] void cutedsl_fail(const std::string& message) {
  throw std::runtime_error("CuTeAOT " CUTEDSL_OP_NAME ": " + message);
}

// Tensors arrive as Tensor? (TorchScript compatibility); the kernel needs one.
torch::stable::Tensor cutedsl_required(
    const std::optional<torch::stable::Tensor>& tensor,
    const char* name) {
  if (!tensor.has_value()) {
    cutedsl_fail(std::string(name) + " must not be None");
  }
  return *tensor;
}

// An i32 scalar arrives as int64_t; narrowing must not silently wrap.
int32_t cutedsl_i32(int64_t value, const char* name) {
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    cutedsl_fail(std::string("scalar ") + name + " does not fit i32");
  }
  return static_cast<int32_t>(value);
}

// Autotune routing (autotune/routing.py): the index of the entry nearest to
// the keys' log2 buckets, the earlier entry on a tie.
template <size_t N, size_t K>
size_t aott_nearest_entry(
    const int64_t (&entries)[N][K],
    const int64_t (&keys)[K]) {
  size_t nearest = 0;
  int64_t nearest_distance = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < N; ++i) {
    int64_t distance = 0;
    for (size_t k = 0; k < K; ++k) {
      // The power-of-2 bucket holding the value, as its log2; <= 1 shares 0.
      const int64_t bucket = keys[k] <= 1
          ? 0
          : 64 - __builtin_clzll(static_cast<uint64_t>(keys[k] - 1));
      distance += (bucket - entries[i][k]) * (bucket - entries[i][k]);
    }
    if (distance < nearest_distance) {
      nearest = i;
      nearest_distance = distance;
    }
  }
  return nearest;
}

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

void* cutedsl_sidecar_handle() {
  static void* handle = []() -> void* {
    std::string sidecar_path = cutedsl_sidecar_path();
    void* loaded = dlopen(sidecar_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (loaded == nullptr) {
      const char* err = dlerror();
      throw std::runtime_error(
          std::string("CuTeAOT " CUTEDSL_OP_NAME ": dlopen(") + sidecar_path +
          ") failed: " + (err == nullptr ? "unknown" : err));
    }
    return loaded;
  }();
  return handle;
}

// One sidecar entry per variant; each run function caches its own lookup.
template <typename Fn>
Fn cutedsl_load_entry(const char* symbol) {
  void* handle = cutedsl_sidecar_handle();
  dlerror();
  void* sym = dlsym(handle, symbol);
  const char* err = dlerror();
  if (err != nullptr || sym == nullptr) {
    throw std::runtime_error(
        std::string("CuTeAOT " CUTEDSL_OP_NAME ": dlsym(") + symbol +
        ") failed: " + (err == nullptr ? "unknown" : err));
  }
  return reinterpret_cast<Fn>(sym);
}

// __CUTEDSL_AOT_GENERATE_BEGIN__ OP_FN
void cutedsl_op() {}
void cutedsl_dummy_op() {}
// __CUTEDSL_AOT_GENERATE_END__ OP_FN

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(CUTEDSL_OP_NAMESPACE, m) {
  m.def(CUTEDSL_OP_SCHEMA);
}

STABLE_TORCH_LIBRARY_IMPL(CUTEDSL_OP_NAMESPACE, CUDA, m) {
  m.impl(CUTEDSL_OP_NAME, TORCH_BOX(&cutedsl_op));
}

STABLE_TORCH_LIBRARY_IMPL(CUTEDSL_OP_NAMESPACE, CPU, m) {
  m.impl(CUTEDSL_OP_NAME, TORCH_BOX(&cutedsl_dummy_op));
}

STABLE_TORCH_LIBRARY_IMPL(CUTEDSL_OP_NAMESPACE, Meta, m) {
  m.impl(CUTEDSL_OP_NAME, TORCH_BOX(&cutedsl_dummy_op));
}
