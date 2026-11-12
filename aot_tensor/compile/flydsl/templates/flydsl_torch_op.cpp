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
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

// __FLYDSL_AOT_GENERATE_BEGIN__ OP_NAME
#define FLYDSL_OP_NAMESPACE placeholder
#define FLYDSL_OP_NAME "placeholder"
#define FLYDSL_OP_SCHEMA "placeholder(Tensor x) -> ()"
// __FLYDSL_AOT_GENERATE_END__ OP_NAME

// Exported by the FlyDSL export_to_c objects linked into this .so, one set per
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

// The status an exported entry or lifecycle function returns: 0, -1 (module
// not initialized), -2 (not loaded on the device) or a hipError_t.
void flydsl_check_status(int32_t status, const char* what) {
  if (status == 0) {
    return;
  }
  // The FlyDSL runtime leaves a failed load's or launch's error pending, and
  // the process's next HIP call would fail with it.
  (void)hipGetLastError();
  std::string reason;
  if (status == -1) {
    reason = "GPU module not initialized";
  } else if (status == -2) {
    reason = "GPU module not loaded on the current device";
  } else {
    reason = hipGetErrorName(static_cast<hipError_t>(status));
  }
  flydsl_fail(std::string(what) + " failed: " + reason);
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

// __FLYDSL_AOT_GENERATE_BEGIN__ OP_FN
void flydsl_op() {}
void flydsl_dummy_op() {}
// __FLYDSL_AOT_GENERATE_END__ OP_FN

} // namespace

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
