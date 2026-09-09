#pragma once

#include "cuda_utils.cuh"
#include <cstdint>

namespace flash_engine {
namespace quantized {

// Configuration for Quantized GEMM: C = (A_int8 * B_int8) * (scale_a * scale_b)
struct QuantGEMMConfig {
    int M;                // Rows of A and C (e.g. batch_size * tokens)
    int N;                // Columns of B and C (e.g. output projection dim)
    int K;                // Columns of A, rows of B (e.g. hidden dim)

    QuantGEMMConfig(int m, int n, int k) : M(m), N(n), K(k) {}
};

// Host dispatch for INT8 DP4A Matrix Multiplication
cudaError_t launch_int8_gemm_dp4a(
    const int8_t* d_A,                  // [M, K] row-major
    const int8_t* d_B,                  // [K, N] column-major or packed
    float* d_C,                         // [M, N] output (float32 dequantized)
    float scale_a,
    float scale_b,
    const QuantGEMMConfig& config,
    cudaStream_t stream = nullptr
);

// Host dispatch for FP8 emulation / Tensor Core dequantization
cudaError_t launch_fp8_gemm_scaled(
    const uint8_t* d_A,                 // FP8 E4M3 / E5M2 raw bytes
    const uint8_t* d_B,
    float* d_C,
    float scale_a,
    float scale_b,
    const QuantGEMMConfig& config,
    cudaStream_t stream = nullptr
);

} // namespace quantized
} // namespace flash_engine
