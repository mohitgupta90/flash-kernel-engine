#pragma once

#include "cuda_utils.cuh"
#include <cuda_fp16.h>

namespace flash_engine {
namespace attention {

// Configuration struct for Multi-Head Attention kernel execution
struct AttentionConfig {
    int batch_size;       // Batch size (B)
    int num_heads;        // Number of attention heads (H)
    int seq_len_q;        // Query sequence length (N)
    int seq_len_kv;       // Key/Value sequence length (M)
    int head_dim;         // Head dimension (d) - typically 32, 64, or 128
    bool is_causal;       // Whether to apply autoregressive causal lower-triangular mask
    float softmax_scale;  // Scaling factor 1.0f / sqrt(head_dim)

    AttentionConfig(int b, int h, int sq, int skv, int d, bool causal = false)
        : batch_size(b), num_heads(h), seq_len_q(sq), seq_len_kv(skv),
          head_dim(d), is_causal(causal), softmax_scale(1.0f / sqrtf(static_cast<float>(d))) {}
};

// Tile dimensions tuned for modern NVIDIA GPU SRAM (L1/Shared Memory)
// B_r = Query block size, B_c = Key/Value block size
template <int HEAD_DIM>
struct TilingConfig;

// Specialization for Head Dim = 32
template <>
struct TilingConfig<32> {
    static constexpr int BLOCK_M = 64;  // Br
    static constexpr int BLOCK_N = 64;  // Bc
    static constexpr int HEAD_DIM = 32;
    static constexpr int WARPS_PER_BLOCK = 4;
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE; // 128
};

// Specialization for Head Dim = 64
template <>
struct TilingConfig<64> {
    static constexpr int BLOCK_M = 64;  // Br
    static constexpr int BLOCK_N = 64;  // Bc
    static constexpr int HEAD_DIM = 64;
    static constexpr int WARPS_PER_BLOCK = 4;
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE; // 128
};

// Specialization for Head Dim = 128
template <>
struct TilingConfig<128> {
    static constexpr int BLOCK_M = 32;  // Br
    static constexpr int BLOCK_N = 64;  // Bc
    static constexpr int HEAD_DIM = 128;
    static constexpr int WARPS_PER_BLOCK = 4;
    static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE; // 128
};

// Host dispatch prototypes for forward FlashAttention-2
cudaError_t launch_flash_attention_forward_fp32(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    const AttentionConfig& config,
    cudaStream_t stream = nullptr
);

cudaError_t launch_flash_attention_forward_fp16(
    const half* d_Q,
    const half* d_K,
    const half* d_V,
    half* d_O,
    const AttentionConfig& config,
    cudaStream_t stream = nullptr
);

// Naive attention baseline for comparative speedup evaluation
cudaError_t launch_naive_attention_forward_fp32(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    const AttentionConfig& config,
    cudaStream_t stream = nullptr
);

} // namespace attention
} // namespace flash_engine
