#pragma once

#include "cuda_utils.cuh"
#include <cuda_fp16.h>

namespace flash_engine {
namespace paged_attention {

// Configuration for PagedAttention (vLLM-style virtual block table management)
struct PagedAttentionConfig {
    int num_seqs;                // Number of sequences in the batch
    int num_heads;               // Number of query attention heads
    int num_kv_heads;            // Number of KV heads (supports Multi-Query & Grouped-Query Attention)
    int head_dim;                // Dimension per head (e.g. 64, 128)
    int block_size;              // Number of tokens per physical cache block (typically 16 or 32)
    int max_context_len;         // Maximum sequence length in batch
    int max_num_blocks_per_seq;  // Maximum physical blocks allocated per sequence
    float scale;                 // Softmax scaling factor 1.0f / sqrt(head_dim)

    PagedAttentionConfig(int seqs, int heads, int kv_heads, int dim, int blk_size, int max_len, int max_blks)
        : num_seqs(seqs), num_heads(heads), num_kv_heads(kv_heads), head_dim(dim),
          block_size(blk_size), max_context_len(max_len), max_num_blocks_per_seq(max_blks),
          scale(1.0f / sqrtf(static_cast<float>(dim))) {}
};

// Host dispatch for PagedAttention v1 forward (FP16 KV Cache)
cudaError_t launch_paged_attention_v1_fp16(
    const half* d_query,                 // [num_seqs, num_heads, head_dim]
    const half* d_key_cache,             // [num_blocks, num_kv_heads, head_dim / x, block_size, x]
    const half* d_value_cache,           // [num_blocks, num_kv_heads, head_dim, block_size]
    half* d_output,                      // [num_seqs, num_heads, head_dim]
    const int* d_block_tables,           // [num_seqs, max_num_blocks_per_seq]
    const int* d_context_lens,           // [num_seqs]
    const PagedAttentionConfig& config,
    cudaStream_t stream = nullptr
);

// Host dispatch for PagedAttention v1 forward (FP32 precision)
cudaError_t launch_paged_attention_v1_fp32(
    const float* d_query,                // [num_seqs, num_heads, head_dim]
    const float* d_key_cache,            // [num_blocks, num_kv_heads, block_size, head_dim]
    const float* d_value_cache,          // [num_blocks, num_kv_heads, block_size, head_dim]
    float* d_output,                     // [num_seqs, num_heads, head_dim]
    const int* d_block_tables,           // [num_seqs, max_num_blocks_per_seq]
    const int* d_context_lens,           // [num_seqs]
    const PagedAttentionConfig& config,
    cudaStream_t stream = nullptr
);

} // namespace paged_attention
} // namespace flash_engine
