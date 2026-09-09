#pragma once

#include "cuda_utils.cuh"

namespace flash_engine {
namespace decoding {

struct FlashDecodingConfig {
    int batch_size;       // Batch size (B)
    int num_heads;        // Number of attention heads (H)
    int seq_len_kv;       // Long context sequence length (M)
    int head_dim;         // Head dimension (d)
    int num_splits;       // Number of splits along the sequence dimension (e.g. 4, 8, 16)
    float scale;          // Softmax scaling factor 1.0f / sqrt(head_dim)

    FlashDecodingConfig(int b, int h, int m, int d, int splits = 8)
        : batch_size(b), num_heads(h), seq_len_kv(m), head_dim(d),
          num_splits(splits), scale(1.0f / sqrtf(static_cast<float>(d))) {}
};

// Host dispatch for Split-KV FlashDecoding
cudaError_t launch_flash_decoding_forward_fp32(
    const float* d_query,                // [B, H, 1, d]
    const float* d_key,                  // [B, H, M, d]
    const float* d_value,                // [B, H, M, d]
    float* d_output,                     // [B, H, 1, d]
    float* d_scratch_out,                // Scratchpad: [B, H, num_splits, d]
    float* d_scratch_stats,              // Scratchpad: [B, H, num_splits, 2] (stores m and l)
    const FlashDecodingConfig& config,
    cudaStream_t stream = nullptr
);

} // namespace decoding
} // namespace flash_engine
