#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) \
  {                                    \
    if ((err) != cudaSuccess) {        \
      printf(                          \
          "Got Cuda Error %s: %s\n",   \
          cudaGetErrorName(err),       \
          cudaGetErrorString(err));    \
    }                                  \
  }

#endif

typedef struct {
  cudaLibrary_t module;
} _cutedsl_VectorAddKernel_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir__cutedsl_VectorAddKernel_cuda_init(void**);
void _mlir__cutedsl_VectorAddKernel_cuda_load_to_device(void**);
static inline void _cutedsl_VectorAddKernel_Kernel_Module_Load(
    _cutedsl_VectorAddKernel_Kernel_Module_t* module) {
  cudaLibrary_t* libraryPtr = &(module->module);
  cudaError_t ret;
  struct {
    cudaLibrary_t** libraryPtr;
    cudaError_t* ret;
  } initArgs = {&libraryPtr, &ret};
  _mlir__cutedsl_VectorAddKernel_cuda_init((void**)(&initArgs));
  CUTE_DSL_CUDA_ERROR_CHECK(ret);
  int32_t device_id = 0;
  struct {
    cudaLibrary_t** library;
    int32_t* device_id;
    cudaError_t* ret;
  } loadArgs = {&libraryPtr, &device_id, &ret};
  int32_t device_count;
  CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
  for (int32_t i = 0; i < device_count; i++) {
    device_id = i;
    _mlir__cutedsl_VectorAddKernel_cuda_load_to_device((void**)(&loadArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
  }
}

static inline void _cutedsl_VectorAddKernel_Kernel_Module_Unload(
    _cutedsl_VectorAddKernel_Kernel_Module_t* module) {
  CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
  void* data;
  int32_t dynamic_shapes[1];
} _cutedsl_VectorAddKernel_Tensor_x_t;

typedef struct {
  void* data;
  int32_t dynamic_shapes[1];
} _cutedsl_VectorAddKernel_Tensor_y_t;

typedef struct {
  void* data;
  int32_t dynamic_shapes[1];
} _cutedsl_VectorAddKernel_Tensor_out_t;

#ifdef __cplusplus
extern "C"
#endif
    void
    _mlir__cutedsl_VectorAddKernel__mlir_ciface_cutlass___call___triton_aotexamplemodelstoy_cutedslVectorAddKernel_object_at__Tensorgmemo1_Tensorgmemo1_Tensorgmemo1__CUstream0x0(
        void** args,
        int32_t num_args);

static inline int32_t cute_dsl__cutedsl_VectorAddKernel_wrapper(
    _cutedsl_VectorAddKernel_Kernel_Module_t* module,
    _cutedsl_VectorAddKernel_Tensor_x_t* x,
    _cutedsl_VectorAddKernel_Tensor_y_t* y,
    _cutedsl_VectorAddKernel_Tensor_out_t* out,
    int32_t n,
    cudaStream_t stream) {
  int32_t ret;
  void* args[6] = {x, y, out, &n, &stream, &ret};
  _mlir__cutedsl_VectorAddKernel__mlir_ciface_cutlass___call___triton_aotexamplemodelstoy_cutedslVectorAddKernel_object_at__Tensorgmemo1_Tensorgmemo1_Tensorgmemo1__CUstream0x0(
      args, 6);
  return ret;
}
