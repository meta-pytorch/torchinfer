// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <hip/hip_runtime_api.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

// __FLYDSL_AOT_GENERATE_BEGIN__ OP_NAME
#define FLYDSL_OP_NAMESPACE placeholder
#define FLYDSL_OP_NAME "placeholder"
#define FLYDSL_OP_SCHEMA "placeholder(Tensor x) -> ()"
// __FLYDSL_AOT_GENERATE_END__ OP_NAME

// Exported by the FlyDSL launcher objects linked into this .so, one set per
// specialization.
// __FLYDSL_AOT_GENERATE_BEGIN__ EXTERNS
// __FLYDSL_AOT_GENERATE_END__ EXTERNS

namespace {

[[noreturn]] void flydsl_fail(const std::string& what) {
  throw std::runtime_error("FlyAOT " FLYDSL_OP_NAME ": " + what);
}

void* flydsl_current_stream(int32_t device_index) {
  void* stream = nullptr;
  if (aoti_torch_get_current_cuda_stream(device_index, &stream) != 0) {
    flydsl_fail("failed to get current stream");
  }
  return stream;
}

// Runs one variant's GPU module init + load once. They only record the image
// (mgpuModuleLoad below); each device loads it on its first launch, in
// mgpuModuleGetFunction.
using FlydslModuleLoader = void (*)(void** module, int32_t* err);

void flydsl_load_module(FlydslModuleLoader init, FlydslModuleLoader load) {
  void* module = nullptr;
  int32_t err = 0;
  init(&module, &err);
  if (err != 0) {
    flydsl_fail("GPU module init failed with error " + std::to_string(err));
  }
  load(&module, &err);
  if (err != 0 || module == nullptr) {
    flydsl_fail("GPU module load failed with error " + std::to_string(err));
  }
}

torch::stable::Tensor flydsl_required(
    const std::optional<torch::stable::Tensor>& tensor,
    const char* name) {
  if (!tensor.has_value()) {
    flydsl_fail(std::string(name) + " must not be None");
  }
  return *tensor;
}

int32_t flydsl_i32(int64_t value, const char* what) {
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    flydsl_fail(std::string(what) + " does not fit int32");
  }
  return static_cast<int32_t>(value);
}

template <typename T>
void flydsl_put(uint8_t* buffer, size_t buffer_size, size_t offset, T value) {
  if (offset > buffer_size || buffer_size - offset < sizeof(T)) {
    flydsl_fail("layout slot write out of bounds");
  }
  const auto bytes = std::bit_cast<std::array<uint8_t, sizeof(T)>>(value);
  std::copy(bytes.begin(), bytes.end(), buffer + offset);
}

void flydsl_check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    flydsl_fail(std::string(what) + " failed: " + hipGetErrorName(err));
  }
}

// A HIP module only runs on the device it was loaded on, so the embedded image
// is loaded once per device, on that device's first launch.
struct FlydslModule {
  const void* image;
  std::mutex mutex;
  std::unordered_map<int, hipModule_t> loaded;
};

hipModule_t flydsl_module_on(FlydslModule& module, int device) {
  std::lock_guard<std::mutex> lock(module.mutex);
  if (auto it = module.loaded.find(device); it != module.loaded.end()) {
    return it->second;
  }
  hipModule_t handle = nullptr;
  flydsl_check(hipModuleLoadData(&handle, module.image), "hipModuleLoadData");
  module.loaded.emplace(device, handle);
  return handle;
}

// __FLYDSL_AOT_GENERATE_BEGIN__ OP_FN
void flydsl_op() {}
void flydsl_dummy_op() {}
// __FLYDSL_AOT_GENERATE_END__ OP_FN

} // namespace

// The MLIR GPU runtime functions the launcher object calls (signatures from
// mlir/lib/ExecutionEngine/RocmRuntimeWrappers.cpp). Defined here rather than
// linked from FlyDSL's runtime libraries, so the .so is self-contained.

// Called once per variant, from its module load function; lives for the
// process.
extern "C" void* mgpuModuleLoad(void* image, size_t /*size*/) {
  return new FlydslModule{image, {}, {}};
}

// Called on every launch. ``name`` points at a constant in the launcher object.
extern "C" hipFunction_t mgpuModuleGetFunction(void* module, const char* name) {
  int device = 0;
  flydsl_check(hipGetDevice(&device), "hipGetDevice");
  using Key = std::tuple<void*, int, const char*>;
  thread_local std::map<Key, hipFunction_t> functions;
  const Key key{module, device, name};
  if (auto it = functions.find(key); it != functions.end()) {
    return it->second;
  }
  hipFunction_t function = nullptr;
  flydsl_check(
      hipModuleGetFunction(
          &function,
          flydsl_module_on(*static_cast<FlydslModule*>(module), device),
          name),
      "hipModuleGetFunction");
  functions.emplace(key, function);
  return function;
}

extern "C" void mgpuLaunchKernel(
    hipFunction_t function,
    intptr_t grid_x,
    intptr_t grid_y,
    intptr_t grid_z,
    intptr_t block_x,
    intptr_t block_y,
    intptr_t block_z,
    int32_t shared_mem,
    hipStream_t stream,
    void** params,
    void** extra,
    size_t /*params_count*/) {
  flydsl_check(
      hipModuleLaunchKernel(
          function,
          grid_x,
          grid_y,
          grid_z,
          block_x,
          block_y,
          block_z,
          shared_mem,
          stream,
          params,
          extra),
      "hipModuleLaunchKernel");
}

STABLE_TORCH_LIBRARY_FRAGMENT(FLYDSL_OP_NAMESPACE, m) {
  m.def(FLYDSL_OP_SCHEMA);
}

// ROCm torch dispatches HIP tensors through the CUDA key.
STABLE_TORCH_LIBRARY_IMPL(FLYDSL_OP_NAMESPACE, CUDA, m) {
  m.impl(FLYDSL_OP_NAME, TORCH_BOX(&flydsl_op));
}

STABLE_TORCH_LIBRARY_IMPL(FLYDSL_OP_NAMESPACE, CPU, m) {
  m.impl(FLYDSL_OP_NAME, TORCH_BOX(&flydsl_dummy_op));
}

STABLE_TORCH_LIBRARY_IMPL(FLYDSL_OP_NAMESPACE, Meta, m) {
  m.impl(FLYDSL_OP_NAME, TORCH_BOX(&flydsl_dummy_op));
}
