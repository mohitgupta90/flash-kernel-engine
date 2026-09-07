#pragma once

#include "cuda_utils.cuh"
#include <cuda_fp16.h>

namespace flash_engine {
namespace norm {

// Configuration for RMSNorm
struct RMSNormConfig {
    int batch_size;       // Total tokens (B * seq_len)
    int hidden_dim;       // Hidden dimension (e.g., 2048, 4096, 8192)
    float epsilon;        // Epsilon numerical stability constant (e.g. 1e-5 or 1e-6)

    RMSNormConfig(int b, int d, float eps = 1e-5f)
        : batch_size(b), hidden_dim(d), epsilon(eps) {}
};

// Host dispatch for Fused Single-Pass RMSNorm (FP32)
cudaError_t launch_fused_rmsnorm_fp32(
    const float* d_input,
    const float* d_gamma,
    float* d_output,
    const RMSNormConfig& config,
    cudaStream_t stream = nullptr
);

// Host dispatch for Fused Single-Pass RMSNorm (FP16)
cudaError_t launch_fused_rmsnorm_fp16(
    const half* d_input,
    const half* d_gamma,
    half* d_output,
    const RMSNormConfig& config,
    cudaStream_t stream = nullptr
);

} // namespace norm
} // namespace flash_engine
