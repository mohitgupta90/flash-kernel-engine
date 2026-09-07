#pragma once

#include "cuda_utils.cuh"

namespace flash_engine {
namespace rope {

struct RoPEConfig {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float theta_base;

    RoPEConfig(int b, int h, int s, int d, float base = 10000.0f)
        : batch_size(b), num_heads(h), seq_len(s), head_dim(d), theta_base(base) {}
};

// Host dispatch for Fused RoPE Kernel (FP32)
cudaError_t launch_fused_rope_fp32(
    const float* d_in,
    const float* d_cos,
    const float* d_sin,
    float* d_out,
    const RoPEConfig& config,
    cudaStream_t stream = nullptr
);

} // namespace rope
} // namespace flash_engine
