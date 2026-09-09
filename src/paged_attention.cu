#include "paged_attention.cuh"
#include <cuda_fp16.h>
#include <cfloat>

namespace flash_engine {
namespace paged_attention {

// =============================================================================
// PAGED ATTENTION V1 FORWARD KERNEL (FP32)
// Memory-efficient KV Cache traversal via virtual block table mappings
// Grid: (num_seqs, num_heads)
// Block: 128 threads (4 warps)
// =============================================================================
template <int BLOCK_SIZE, int HEAD_DIM>
__global__ void __launch_bounds__(128) paged_attention_v1_kernel_fp32(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ output,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int max_num_blocks_per_seq,
    int num_kv_heads,
    int num_heads,
    float scale
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    // GQA/MQA mapping: multiple query heads map to the same KV head
    int heads_per_kv_head = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / heads_per_kv_head;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    constexpr int NUM_WARPS = 128 / WARP_SIZE; // 4

    // Query pointer: [num_seqs, num_heads, head_dim]
    size_t q_offset = (static_cast<size_t>(seq_idx) * num_heads + head_idx) * HEAD_DIM;
    const float* q_ptr = query + q_offset;

    // Load query into shared memory
    __shared__ float s_query[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();

    // Block table for current sequence: [num_seqs, max_num_blocks_per_seq]
    const int* seq_block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Thread-local online softmax statistics
    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc_o[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc_o[d] = 0.0f;
    }

    // Iterate across logical blocks in the sequence
    for (int logical_blk = 0; logical_blk < num_blocks; ++logical_blk) {
        int physical_blk = seq_block_table[logical_blk];
        int start_token_idx = logical_blk * BLOCK_SIZE;

        // Each physical block stores BLOCK_SIZE tokens
        // Key/Value Cache shape: [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
        size_t block_base_offset = (static_cast<size_t>(physical_blk) * num_kv_heads + kv_head_idx) * BLOCK_SIZE * HEAD_DIM;
        const float* k_blk_ptr = key_cache + block_base_offset;
        const float* v_blk_ptr = value_cache + block_base_offset;

        // Number of valid tokens in this block
        int tokens_in_block = (logical_blk == num_blocks - 1) 
            ? (context_len - start_token_idx) 
            : BLOCK_SIZE;

        // Threads in the block cooperatively evaluate dot products for tokens
        for (int t = tid; t < tokens_in_block; t += blockDim.x) {
            const float* k_token = k_blk_ptr + t * HEAD_DIM;
            const float* v_token = v_blk_ptr + t * HEAD_DIM;

            // Dot-product Q . K
            float score = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += s_query[d] * k_token[d];
            }
            score *= scale;

            // Online Softmax update
            float m_prev = m_i;
            m_i = fmaxf(m_prev, score);
            float correction = __expf(m_prev - m_i);
            float p = __expf(score - m_i);

            l_i = l_i * correction + p;

            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc_o[d] = acc_o[d] * correction + p * v_token[d];
            }
        }
    }

    // -------------------------------------------------------------------------
    // Inter-thread Block Reduction across 128 threads
    // Combine thread-local m_i, l_i, and acc_o[d]
    // -------------------------------------------------------------------------
    __shared__ float s_warp_m[NUM_WARPS];
    __shared__ float s_warp_l[NUM_WARPS];
    __shared__ float s_warp_o[NUM_WARPS][HEAD_DIM];

    // Intra-warp max reduction
    float warp_m = flash_engine::cuda::warp_reduce_max(m_i);
    float warp_m_broadcast = flash_engine::cuda::warp_broadcast(warp_m, 0);

    // Rescale local l_i and acc_o to warp max
    float local_rescale = __expf(m_i - warp_m_broadcast);
    l_i *= local_rescale;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc_o[d] *= local_rescale;
    }

    float warp_l = flash_engine::cuda::warp_reduce_sum(l_i);

    if (lane == 0) {
        s_warp_m[wid] = warp_m_broadcast;
        s_warp_l[wid] = warp_l;
    }
    __syncthreads();

    // Reduce across warps using warp 0
    if (wid == 0 && lane < NUM_WARPS) {
        // Find global block maximum
        float global_m = -1e30f;
        for (int w = 0; w < NUM_WARPS; ++w) {
            global_m = fmaxf(global_m, s_warp_m[w]);
        }
        s_warp_m[0] = global_m;

        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; ++w) {
            global_l += s_warp_l[w] * __expf(s_warp_m[w] - global_m);
        }
        s_warp_l[0] = global_l;
    }
    __syncthreads();

    float block_global_m = s_warp_m[0];
    float block_global_l = s_warp_l[0];

    // Write final normalized output to global memory
    float inv_l = 1.0f / (block_global_l + 1e-6f);
    float final_correction = __expf(m_i - block_global_m);

    // Accumulate each dimension across all threads using atomicAdd to shared memory
    __shared__ float s_final_acc[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        s_final_acc[d] = 0.0f;
    }
    __syncthreads();

    for (int d = 0; d < HEAD_DIM; ++d) {
        atomicAdd(&s_final_acc[d], acc_o[d] * final_correction);
    }
    __syncthreads();

    float* out_ptr = output + q_offset;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        out_ptr[d] = s_final_acc[d] * inv_l;
    }
}

// Host dispatch FP32
cudaError_t launch_paged_attention_v1_fp32(
    const float* d_query,
    const float* d_key_cache,
    const float* d_value_cache,
    float* d_output,
    const int* d_block_tables,
    const int* d_context_lens,
    const PagedAttentionConfig& config,
    cudaStream_t stream
) {
    dim3 grid(config.num_seqs, config.num_heads);
    dim3 block(128);

    if (config.head_dim == 64) {
        paged_attention_v1_kernel_fp32<16, 64><<<grid, block, 0, stream>>>(
            d_query, d_key_cache, d_value_cache, d_output,
            d_block_tables, d_context_lens,
            config.max_num_blocks_per_seq, config.num_kv_heads, config.num_heads, config.scale
        );
    } else if (config.head_dim == 128) {
        paged_attention_v1_kernel_fp32<16, 128><<<grid, block, 0, stream>>>(
            d_query, d_key_cache, d_value_cache, d_output,
            d_block_tables, d_context_lens,
            config.max_num_blocks_per_seq, config.num_kv_heads, config.num_heads, config.scale
        );
    } else {
        paged_attention_v1_kernel_fp32<16, 32><<<grid, block, 0, stream>>>(
            d_query, d_key_cache, d_value_cache, d_output,
            d_block_tables, d_context_lens,
            config.max_num_blocks_per_seq, config.num_kv_heads, config.num_heads, config.scale
        );
    }

    return cudaGetLastError();
}

// Host dispatch FP16
cudaError_t launch_paged_attention_v1_fp16(
    const half* d_query,
    const half* d_key_cache,
    const half* d_value_cache,
    half* d_output,
    const int* d_block_tables,
    const int* d_context_lens,
    const PagedAttentionConfig& config,
    cudaStream_t stream
) {
    // For FP16 dispatch
    return cudaGetLastError();
}

} // namespace paged_attention
} // namespace flash_engine
