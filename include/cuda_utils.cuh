#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Error checking macro for CUDA runtime API calls
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                               \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

namespace flash_engine {
namespace cuda {

// -----------------------------------------------------------------------------
// Warp-Level Shuffle Primitives
// -----------------------------------------------------------------------------

// Warp-wide reduction for sum (FP32)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Warp-wide reduction for max (FP32)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

// Warp-wide broadcast from lane 0
__device__ __forceinline__ float warp_broadcast(float val, int src_lane = 0) {
    return __shfl_sync(FULL_MASK, val, src_lane);
}

// Block-wide reduction for sum (FP32)
template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared_mem[wid] = val;
    }
    __syncthreads();

    float sum = (threadIdx.x < NUM_WARPS) ? shared_mem[threadIdx.x] : 0.0f;
    if (wid == 0) {
        sum = warp_reduce_sum(sum);
    }
    return sum;
}

// Block-wide reduction for max (FP32)
template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared_mem[wid] = val;
    }
    __syncthreads();

    float max_val = (threadIdx.x < NUM_WARPS) ? shared_mem[threadIdx.x] : -1e20f;
    if (wid == 0) {
        max_val = warp_reduce_max(max_val);
    }
    return max_val;
}

// -----------------------------------------------------------------------------
// Vectorized Memory Access Helpers (128-bit memory transactions)
// -----------------------------------------------------------------------------

// Vectorized load float4 (128 bits = 4x float32)
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// Vectorized store float4
__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

// Vectorized load half2 (32 bits = 2x float16)
__device__ __forceinline__ half2 load_half2(const half* ptr) {
    return *reinterpret_cast<const half2*>(ptr);
}

// Vectorized store half2
__device__ __forceinline__ void store_half2(half* ptr, half2 val) {
    *reinterpret_cast<half2*>(ptr) = val;
}

// Fast exponential approximation
__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

// Mathematical constants
constexpr float NEG_INFINITY = -1e30f;

} // namespace cuda
} // namespace flash_engine
