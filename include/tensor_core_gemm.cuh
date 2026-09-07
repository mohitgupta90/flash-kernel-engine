#pragma once

#include "cuda_utils.cuh"
#include <mma.h>
#include <cuda_fp16.h>

namespace flash_engine {
namespace tensor_core {

using namespace nvcuda;

// Dimensions for WMMA sub-matrices
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block dimensions for Tensor Core GEMM
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;

// Warp tile counts
constexpr int WARPS_M = TILE_M / WMMA_M; // 4
constexpr int WARPS_N = TILE_N / WMMA_N; // 4

// Launch configuration for WMMA-accelerated GEMM: C = alpha * (A * B) + beta * C
cudaError_t launch_wmma_gemm_fp16(
    const half* d_A,
    const half* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
);

// WMMA-accelerated Attention projection
cudaError_t launch_wmma_attention_qk_pv(
    const half* d_Q,
    const half* d_K,
    const half* d_V,
    half* d_O,
    int B, int H, int seq_len, int head_dim,
    float scale,
    cudaStream_t stream = nullptr
);

} // namespace tensor_core
} // namespace flash_engine
