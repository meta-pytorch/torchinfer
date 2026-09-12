#pragma once

#include <torch/csrc/stable/tensor.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace triton {
namespace aot {

#ifndef GRID_DIM_DEFINED_MACRO
struct gridDims {
  int x = 1;
  int y = 1;
  int z = 1;
  cudaStream_t stream = 0;
  gridDims(int _x = 1, int _y = 1, int _z = 1, cudaStream_t _stream = 0)
      : x(_x), y(_y), z(_z), stream(_stream) {}
};
#define GRID_DIM_DEFINED_MACRO
#endif

#ifndef FITS_I32_DEFINED_MACRO
constexpr bool fits_i32(int64_t v) {
  return v >= INT32_MIN && v <= INT32_MAX;
}
#define FITS_I32_DEFINED_MACRO
#endif

// Both generated regions below are named after the kernel alone, so every .so
// generated for that kernel defines the same `triton::aot::<kernel>` and
// `triton::aot::<kernel>_meta` symbols. Multi-forward emits one .so per forward
// method and the predictor dlopens them all into a single process; at default
// visibility the first library loaded wins the binding and serves every
// method's calls -- silently running another method's guard chain, whose
// compiled variants are a different set. Nothing outside the owning .so ever
// calls these, so hide them.
#pragma GCC visibility push(hidden)
// __TRITON_AOT_GENERATE_BEGIN__ TUNER_META_CPP
// __TRITON_AOT_GENERATE_END__ TUNER_META_CPP

// __TRITON_AOT_GENERATE_BEGIN__ SELECTOR_PROTO
// __TRITON_AOT_GENERATE_END__ SELECTOR_PROTO
#pragma GCC visibility pop

} // namespace aot
} // namespace triton
