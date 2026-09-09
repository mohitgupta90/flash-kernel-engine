#include "flash_decoding.cuh"
#include <cmath>

namespace flash_engine {
namespace decoding {

// =============================================================================
// STAGE 1: SPLIT-KV PARTIAL ATTENTION KERNEL
// Grid: (num_splits, B * H)
// Block: 128 threads
// Computes local attention over sequence split [k_start, k_end)
// =============================================================================
template <int HEAD_DIM>
__global__ void __launch_bounds__(128) flash_decoding_stage1_kernel_fp32(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ scratch_out,
    float* __restrict__ scratch_stats,
    int B, int H, int M,
    int num_splits,
    float scale
) {
    int split_idx = blockIdx.x;
    int batch_head = blockIdx.y;

    int tid = threadIdx.x;

    // Split range along KV sequence length M
    int tokens_per_split = (M + num_splits - 1) / num_splits;
    int k_start = split_idx * tokens_per_split;
    int k_end = (k_start + tokens_per_split < M) ? (k_start + tokens_per_split) : M;

    if (k_start >= M) {
        // Write inactive split stats
        if (tid == 0) {
            size_t stats_offset = (static_cast<size_t>(batch_head) * num_splits + split_idx) * 2;
            scratch_stats[stats_offset + 0] = -1e30f;
            scratch_stats[stats_offset + 1] = 0.0f;
        }
        return;
    }

    // Strides
    size_t q_offset = static_cast<size_t>(batch_head) * HEAD_DIM;
    size_t kv_base = static_cast<size_t>(batch_head) * M * HEAD_DIM;

    const float* q_ptr = query + q_offset;
    const float* k_base_ptr = key + kv_base;
    const float* v_base_ptr = value + kv_base;

    // Load query into shared memory
    __shared__ float s_query[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();

    // Local online softmax statistics
    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc_o[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc_o[d] = 0.0f;
    }

    // Process tokens in this split
    for (int k_idx = k_start + tid; k_idx < k_end; k_idx += blockDim.x) {
        const float* k_token = k_base_ptr + k_idx * HEAD_DIM;
        const float* v_token = v_base_ptr + k_idx * HEAD_DIM;

        float score = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < HEAD_DIM; ++d) {
            score += s_query[d] * k_token[d];
        }
        score *= scale;

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

    // Intra-warp max reduction
    float warp_m = flash_engine::cuda::warp_reduce_max(m_i);
    float warp_m_broadcast = flash_engine::cuda::warp_broadcast(warp_m, 0);

    float rescale = __expf(m_i - warp_m_broadcast);
    l_i *= rescale;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc_o[d] *= rescale;
    }

    // Inter-thread block reduction
    __shared__ float s_max_val;
    __shared__ float s_sum_l;
    __shared__ float s_acc[HEAD_DIM];

    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        s_acc[d] = 0.0f;
    }
    if (tid == 0) {
        s_max_val = -1e30f;
        s_sum_l = 0.0f;
    }
    __syncthreads();

    // Reduce global block max
    atomicMax(reinterpret_cast<int*>(&s_max_val), __float_as_int(warp_m_broadcast));
    __syncthreads();

    float block_max = s_max_val;
    float block_correction = __expf(m_i - block_max);
    atomicAdd(&s_sum_l, l_i * block_correction);

    for (int d = 0; d < HEAD_DIM; ++d) {
        atomicAdd(&s_acc[d], acc_o[d] * block_correction);
    }
    __syncthreads();

    // Store un-normalized partial accumulator and stats to scratchpad
    size_t out_offset = (static_cast<size_t>(batch_head) * num_splits + split_idx) * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        scratch_out[out_offset + d] = s_acc[d];
    }

    if (tid == 0) {
        size_t stats_offset = (static_cast<size_t>(batch_head) * num_splits + split_idx) * 2;
        scratch_stats[stats_offset + 0] = block_max;
        scratch_stats[stats_offset + 1] = s_sum_l;
    }
}

// =============================================================================
// STAGE 2: FLASH DECODING REDUCTION KERNEL
// Consolidates partial accumulators from 'num_splits' into final output
// Grid: (B * H)
// Block: 128 threads
// =============================================================================
template <int HEAD_DIM>
__global__ void __launch_bounds__(128) flash_decoding_stage2_reduction_kernel(
    const float* __restrict__ scratch_out,
    const float* __restrict__ scratch_stats,
    float* __restrict__ output,
    int num_splits
) {
    int batch_head = blockIdx.x;
    int tid = threadIdx.x;

    size_t stats_base = static_cast<size_t>(batch_head) * num_splits * 2;
    size_t out_scratch_base = static_cast<size_t>(batch_head) * num_splits * HEAD_DIM;

    // Step 1: Find overall maximum m across all splits
    __shared__ float s_global_max;
    __shared__ float s_total_l;
    __shared__ float s_final_o[HEAD_DIM];

    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        s_final_o[d] = 0.0f;
    }
    if (tid == 0) {
        s_global_max = -1e30f;
        s_total_l = 0.0f;
    }
    __syncthreads();

    // Load and reduce max across splits (typically num_splits <= 32)
    float thread_max = -1e30f;
    for (int s = tid; s < num_splits; s += blockDim.x) {
        float m_split = scratch_stats[stats_base + s * 2 + 0];
        thread_max = fmaxf(thread_max, m_split);
    }
    float warp_max = flash_engine::cuda::warp_reduce_max(thread_max);
    if (tid % WARP_SIZE == 0) {
        atomicMax(reinterpret_cast<int*>(&s_global_max), __float_as_int(warp_max));
    }
    __syncthreads();

    float global_max = s_global_max;

    // Step 2: Rescale normalizer l and accumulate partial output vectors
    float thread_l = 0.0f;
    for (int s = tid; s < num_splits; s += blockDim.x) {
        float m_split = scratch_stats[stats_base + s * 2 + 0];
        float l_split = scratch_stats[stats_base + s * 2 + 1];
        float correction = __expf(m_split - global_max);
        thread_l += l_split * correction;
    }
    float warp_l = flash_engine::cuda::warp_reduce_sum(thread_l);
    if (tid % WARP_SIZE == 0) {
        atomicAdd(&s_total_l, warp_l);
    }
    __syncthreads();

    // Accumulate output vectors
    for (int s = 0; s < num_splits; ++s) {
        float m_split = scratch_stats[stats_base + s * 2 + 0];
        float correction = __expf(m_split - global_max);
        const float* partial_out = scratch_out + out_scratch_base + s * HEAD_DIM;

        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            s_final_o[d] += partial_out[d] * correction;
        }
    }
    __syncthreads();

    // Step 3: Write final normalized output to global memory
    float inv_l = 1.0f / (s_total_l + 1e-6f);
    size_t final_out_offset = static_cast<size_t>(batch_head) * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        output[final_out_offset + d] = s_final_o[d] * inv_l;
    }
}

cudaError_t launch_flash_decoding_forward_fp32(
    const float* d_query,
    const float* d_key,
    const float* d_value,
    float* d_output,
    float* d_scratch_out,
    float* d_scratch_stats,
    const FlashDecodingConfig& config,
    cudaStream_t stream
) {
    int B = config.batch_size;
    int H = config.num_heads;
    int M = config.seq_len_kv;
    int d = config.head_dim;
    int splits = config.num_splits;

    dim3 grid_stage1(splits, B * H);
    dim3 block(128);

    if (d == 64) {
        flash_decoding_stage1_kernel_fp32<64><<<grid_stage1, block, 0, stream>>>(
            d_query, d_key, d_value, d_scratch_out, d_scratch_stats, B, H, M, splits, config.scale
        );
        dim3 grid_stage2(B * H);
        flash_decoding_stage2_reduction_kernel<64><<<grid_stage2, block, 0, stream>>>(
            d_scratch_out, d_scratch_stats, d_output, splits
        );
    } else if (d == 128) {
        flash_decoding_stage1_kernel_fp32<128><<<grid_stage1, block, 0, stream>>>(
            d_query, d_key, d_value, d_scratch_out, d_scratch_stats, B, H, M, splits, config.scale
        );
        dim3 grid_stage2(B * H);
        flash_decoding_stage2_reduction_kernel<128><<<grid_stage2, block, 0, stream>>>(
            d_scratch_out, d_scratch_stats, d_output, splits
        );
    }

    return cudaGetLastError();
}

} // namespace decoding
} // namespace flash_engine
