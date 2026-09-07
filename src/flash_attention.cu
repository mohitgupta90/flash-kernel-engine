#include "flash_attention.cuh"
#include <algorithm>
#include <cfloat>

namespace flash_engine {
namespace attention {

// =============================================================================
// NAIVE ATTENTION BASELINE (Global Memory Roundtrips - O(N^2) HBM traffic)
// =============================================================================
// Used for rigorous speedup benchmarks against standard un-fused implementations.
__global__ void naive_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int d,
    float scale,
    bool is_causal
) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || h >= H || q_idx >= N) return;

    // Strides for [B, H, N, d]
    size_t batch_head_offset = (static_cast<size_t>(b) * H + h) * N * d;
    const float* q_row = Q + batch_head_offset + q_idx * d;
    const float* k_base = K + batch_head_offset;
    const float* v_base = V + batch_head_offset;
    float* o_row = O + batch_head_offset + q_idx * d;

    // Step 1: Compute Q * K^T row-wise and find max for numerical stability
    float max_score = -1e30f;
    extern __shared__ float s_scores[]; // Dynamic shared memory per thread if needed

    // Allocate thread-local scores dynamically via register buffer or local array
    // Here we compute online softmax directly to fit within local registers
    float l_i = 0.0f;
    float m_i = -1e30f;

    // Temporary accumulator for Output vector of size d
    float acc_o[128];
    #pragma unroll
    for (int dim = 0; dim < d; ++dim) {
        acc_o[dim] = 0.0f;
    }

    int max_k = is_causal ? (q_idx + 1) : N;

    for (int k_idx = 0; k_idx < max_k; ++k_idx) {
        const float* k_row = k_base + k_idx * d;
        const float* v_row = v_base + k_idx * d;

        // Compute dot-product S = (q . k) * scale
        float score = 0.0f;
        #pragma unroll 4
        for (int dim = 0; dim < d; ++dim) {
            score += q_row[dim] * k_row[dim];
        }
        score *= scale;

        // Online softmax update
        float m_prev = m_i;
        m_i = fmaxf(m_i, score);
        float exp_diff = __expf(m_prev - m_i);
        float p = __expf(score - m_i);

        l_i = l_i * exp_diff + p;

        #pragma unroll 4
        for (int dim = 0; dim < d; ++dim) {
            acc_o[dim] = acc_o[dim] * exp_diff + p * v_row[dim];
        }
    }

    // Final normalize by sum of exponentials l_i
    float inv_l = 1.0f / (l_i + 1e-6f);
    #pragma unroll 4
    for (int dim = 0; dim < d; ++dim) {
        o_row[dim] = acc_o[dim] * inv_l;
    }
}

cudaError_t launch_naive_attention_forward_fp32(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    const AttentionConfig& config,
    cudaStream_t stream
) {
    dim3 grid((config.seq_len_q + 127) / 128, 1, config.batch_size * config.num_heads);
    dim3 block(128);

    naive_attention_forward_kernel<<<grid, block, 0, stream>>>(
        d_Q, d_K, d_V, d_O,
        config.batch_size, config.num_heads, config.seq_len_q, config.head_dim,
        config.softmax_scale, config.is_causal
    );

    return cudaGetLastError();
}


// =============================================================================
// FLASHATTENTION-2 FORWARD KERNEL (FP32)
// SRAM Tiling + Online Softmax + Bank Conflict Avoidance + Warp Intrinsics
// =============================================================================
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128, 2) flash_attention_2_forward_kernel_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int M,
    float scale,
    bool is_causal
) {
    // Grid: (T_r, 1, B * H) where T_r = ceil(N / BLOCK_M)
    int block_m_idx = blockIdx.x; // Block of queries
    int batch_head  = blockIdx.z;
    int b = batch_head / H;
    int h = batch_head % H;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Global memory offset for current batch and head: [B, H, N, HEAD_DIM]
    size_t q_stride_bh = static_cast<size_t>(b * H + h) * N * HEAD_DIM;
    size_t kv_stride_bh = static_cast<size_t>(b * H + h) * M * HEAD_DIM;

    const float* Q_bh = Q + q_stride_bh;
    const float* K_bh = K + kv_stride_bh;
    const float* V_bh = V + kv_stride_bh;
    float* O_bh       = O + q_stride_bh;

    // -------------------------------------------------------------------------
    // Shared Memory Allocation with Bank Conflict Padding
    // HEAD_DIM + 1 pads the stride to prevent 32-way bank conflicts when loading columns
    // -------------------------------------------------------------------------
    constexpr int SMEM_PAD = 1;
    constexpr int STRIDE_Q = HEAD_DIM + SMEM_PAD;
    constexpr int STRIDE_KV = HEAD_DIM + SMEM_PAD;

    __shared__ float s_Q[BLOCK_M][STRIDE_Q];
    __shared__ float s_K[BLOCK_N][STRIDE_KV];
    __shared__ float s_V[BLOCK_N][STRIDE_KV];

    // Thread-local statistics for Online Softmax: running max (m_i) and sum of exp (l_i)
    // Each thread in the block is responsible for rows: tid, tid + blockDim.x, ...
    // With BLOCK_M = 64, and blockDim.x = 128, each thread handles at most 1 row if BLOCK_M <= blockDim.x
    // Or each thread computes a subset of rows:
    constexpr int ROWS_PER_THREAD = (BLOCK_M + 127) / 128; // typically 1 or 2
    int local_row = tid % BLOCK_M;
    int global_row = block_m_idx * BLOCK_M + local_row;

    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc_o[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc_o[d] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // 1. Cooperative Load Q tile into Shared Memory
    // -------------------------------------------------------------------------
    int total_q_elements = BLOCK_M * HEAD_DIM;
    for (int i = tid; i < total_q_elements; i += blockDim.x) {
        int r = i / HEAD_DIM;
        int c = i % HEAD_DIM;
        int g_r = block_m_idx * BLOCK_M + r;
        if (g_r < N) {
            s_Q[r][c] = Q_bh[g_r * HEAD_DIM + c];
        } else {
            s_Q[r][c] = 0.0f;
        }
    }
    __syncthreads();

    // Determine loop boundaries for Key/Value tiles (Tc = ceil(M / BLOCK_N))
    int num_kv_blocks = (M + BLOCK_N - 1) / BLOCK_N;
    int max_kv_block = num_kv_blocks;
    if (is_causal) {
        // Causal masking: key block index j cannot exceed query block index i
        max_kv_block = std::min(num_kv_blocks, ((block_m_idx + 1) * BLOCK_M + BLOCK_N - 1) / BLOCK_N);
    }

    // -------------------------------------------------------------------------
    // 2. Iterate over Key and Value blocks in Shared Memory (FlashAttention Loop)
    // -------------------------------------------------------------------------
    for (int kv_idx = 0; kv_idx < max_kv_block; ++kv_idx) {
        // Cooperative Load K and V tiles into Shared Memory
        int total_kv_elements = BLOCK_N * HEAD_DIM;
        for (int i = tid; i < total_kv_elements; i += blockDim.x) {
            int r = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            int g_r = kv_idx * BLOCK_N + r;
            if (g_r < M) {
                s_K[r][c] = K_bh[g_r * HEAD_DIM + c];
                s_V[r][c] = V_bh[g_r * HEAD_DIM + c];
            } else {
                s_K[r][c] = 0.0f;
                s_V[r][c] = 0.0f;
            }
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // Compute S = (Q * K^T) * scale for local row
        // ---------------------------------------------------------------------
        if (local_row < BLOCK_M && global_row < N) {
            float s_row[BLOCK_N];
            float tile_max = -1e30f;

            #pragma unroll 4
            for (int j = 0; j < BLOCK_N; ++j) {
                int g_col = kv_idx * BLOCK_N + j;
                if (g_col < M && (!is_causal || global_row >= g_col)) {
                    float dot = 0.0f;
                    #pragma unroll 4
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dot += s_Q[local_row][d] * s_K[j][d];
                    }
                    float val = dot * scale;
                    s_row[j] = val;
                    tile_max = fmaxf(tile_max, val);
                } else {
                    s_row[j] = -1e30f;
                }
            }

            // -----------------------------------------------------------------
            // Online Softmax Rescaling Step
            // m_new = max(m_prev, tile_max)
            // correction = exp(m_prev - m_new)
            // l_new = l_prev * correction + sum(exp(s_row - m_new))
            // O = O * correction + P * V
            // -----------------------------------------------------------------
            float m_prev = m_i;
            float m_new = fmaxf(m_prev, tile_max);
            float correction = __expf(m_prev - m_new);

            float sum_p = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < BLOCK_N; ++j) {
                if (s_row[j] > -1e25f) {
                    s_row[j] = __expf(s_row[j] - m_new);
                    sum_p += s_row[j];
                } else {
                    s_row[j] = 0.0f;
                }
            }

            l_i = l_i * correction + sum_p;
            m_i = m_new;

            // Rescale accumulator and accumulate P * V
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; ++d) {
                float pv = 0.0f;
                #pragma unroll 4
                for (int j = 0; j < BLOCK_N; ++j) {
                    pv += s_row[j] * s_V[j][d];
                }
                acc_o[d] = acc_o[d] * correction + pv;
            }
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // 3. Final Normalization and Write-back to HBM
    // -------------------------------------------------------------------------
    if (local_row < BLOCK_M && global_row < N) {
        float inv_l = 1.0f / (l_i + 1e-6f);
        float* o_dest = O_bh + global_row * HEAD_DIM;

        // Vectorized write-back if HEAD_DIM is multiple of 4
        if constexpr (HEAD_DIM % 4 == 0) {
            for (int d = 0; d < HEAD_DIM; d += 4) {
                float4 out_val = make_float4(
                    acc_o[d + 0] * inv_l,
                    acc_o[d + 1] * inv_l,
                    acc_o[d + 2] * inv_l,
                    acc_o[d + 3] * inv_l
                );
                flash_engine::cuda::store_float4(&o_dest[d], out_val);
            }
        } else {
            for (int d = 0; d < HEAD_DIM; ++d) {
                o_dest[d] = acc_o[d] * inv_l;
            }
        }
    }
}

// Host Dispatch for FP32 FlashAttention-2
cudaError_t launch_flash_attention_forward_fp32(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    const AttentionConfig& config,
    cudaStream_t stream
) {
    int B = config.batch_size;
    int H = config.num_heads;
    int N = config.seq_len_q;
    int M = config.seq_len_kv;
    int d = config.head_dim;

    if (d == 32) {
        constexpr int BM = 64, BN = 64, HD = 32;
        dim3 grid((N + BM - 1) / BM, 1, B * H);
        dim3 block(128);
        flash_attention_2_forward_kernel_fp32<BM, BN, HD><<<grid, block, 0, stream>>>(
            d_Q, d_K, d_V, d_O, B, H, N, M, config.softmax_scale, config.is_causal
        );
    } else if (d == 64) {
        constexpr int BM = 64, BN = 64, HD = 64;
        dim3 grid((N + BM - 1) / BM, 1, B * H);
        dim3 block(128);
        flash_attention_2_forward_kernel_fp32<BM, BN, HD><<<grid, block, 0, stream>>>(
            d_Q, d_K, d_V, d_O, B, H, N, M, config.softmax_scale, config.is_causal
        );
    } else if (d == 128) {
        constexpr int BM = 32, BN = 64, HD = 128;
        dim3 grid((N + BM - 1) / BM, 1, B * H);
        dim3 block(128);
        flash_attention_2_forward_kernel_fp32<BM, BN, HD><<<grid, block, 0, stream>>>(
            d_Q, d_K, d_V, d_O, B, H, N, M, config.softmax_scale, config.is_causal
        );
    } else {
        // Fallback for arbitrary head dimensions
        return launch_naive_attention_forward_fp32(d_Q, d_K, d_V, d_O, config, stream);
    }

    return cudaGetLastError();
}

// =============================================================================
// FLASHATTENTION-2 FORWARD KERNEL (FP16 / Half-Precision)
// =============================================================================
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128, 2) flash_attention_2_forward_kernel_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int N, int M,
    float scale,
    bool is_causal
) {
    int block_m_idx = blockIdx.x;
    int batch_head  = blockIdx.z;
    int b = batch_head / H;
    int h = batch_head % H;

    int tid = threadIdx.x;

    size_t q_stride_bh = static_cast<size_t>(b * H + h) * N * HEAD_DIM;
    size_t kv_stride_bh = static_cast<size_t>(b * H + h) * M * HEAD_DIM;

    const half* Q_bh = Q + q_stride_bh;
    const half* K_bh = K + kv_stride_bh;
    const half* V_bh = V + kv_stride_bh;
    half* O_bh       = O + q_stride_bh;

    constexpr int SMEM_PAD = 2;
    constexpr int STRIDE_Q = HEAD_DIM + SMEM_PAD;
    constexpr int STRIDE_KV = HEAD_DIM + SMEM_PAD;

    __shared__ half s_Q[BLOCK_M][STRIDE_Q];
    __shared__ half s_K[BLOCK_N][STRIDE_KV];
    __shared__ half s_V[BLOCK_N][STRIDE_KV];

    int local_row = tid % BLOCK_M;
    int global_row = block_m_idx * BLOCK_M + local_row;

    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc_o[HEAD_DIM];

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc_o[d] = 0.0f;
    }

    // Cooperative Load Q tile into Shared Memory
    int total_q = BLOCK_M * HEAD_DIM;
    for (int i = tid; i < total_q; i += blockDim.x) {
        int r = i / HEAD_DIM;
        int c = i % HEAD_DIM;
        int g_r = block_m_idx * BLOCK_M + r;
        s_Q[r][c] = (g_r < N) ? Q_bh[g_r * HEAD_DIM + c] : __float2half(0.0f);
    }
    __syncthreads();

    int num_kv_blocks = (M + BLOCK_N - 1) / BLOCK_N;
    int max_kv_block = is_causal ? std::min(num_kv_blocks, ((block_m_idx + 1) * BLOCK_M + BLOCK_N - 1) / BLOCK_N) : num_kv_blocks;

    for (int kv_idx = 0; kv_idx < max_kv_block; ++kv_idx) {
        int total_kv = BLOCK_N * HEAD_DIM;
        for (int i = tid; i < total_kv; i += blockDim.x) {
            int r = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            int g_r = kv_idx * BLOCK_N + r;
            if (g_r < M) {
                s_K[r][c] = K_bh[g_r * HEAD_DIM + c];
                s_V[r][c] = V_bh[g_r * HEAD_DIM + c];
            } else {
                s_K[r][c] = __float2half(0.0f);
                s_V[r][c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        if (local_row < BLOCK_M && global_row < N) {
            float s_row[BLOCK_N];
            float tile_max = -1e30f;

            for (int j = 0; j < BLOCK_N; ++j) {
                int g_col = kv_idx * BLOCK_N + j;
                if (g_col < M && (!is_causal || global_row >= g_col)) {
                    float dot = 0.0f;
                    #pragma unroll 4
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dot += __half2float(s_Q[local_row][d]) * __half2float(s_K[j][d]);
                    }
                    float val = dot * scale;
                    s_row[j] = val;
                    tile_max = fmaxf(tile_max, val);
                } else {
                    s_row[j] = -1e30f;
                }
            }

            float m_prev = m_i;
            float m_new = fmaxf(m_prev, tile_max);
            float correction = __expf(m_prev - m_new);

            float sum_p = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < BLOCK_N; ++j) {
                if (s_row[j] > -1e25f) {
                    s_row[j] = __expf(s_row[j] - m_new);
                    sum_p += s_row[j];
                } else {
                    s_row[j] = 0.0f;
                }
            }

            l_i = l_i * correction + sum_p;
            m_i = m_new;

            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; ++d) {
                float pv = 0.0f;
                #pragma unroll 4
                for (int j = 0; j < BLOCK_N; ++j) {
                    pv += s_row[j] * __half2float(s_V[j][d]);
                }
                acc_o[d] = acc_o[d] * correction + pv;
            }
        }
        __syncthreads();
    }

    if (local_row < BLOCK_M && global_row < N) {
        float inv_l = 1.0f / (l_i + 1e-6f);
        half* o_dest = O_bh + global_row * HEAD_DIM;

        for (int d = 0; d < HEAD_DIM; ++d) {
            o_dest[d] = __float2half(acc_o[d] * inv_l);
        }
    }
}

cudaError_t launch_flash_attention_forward_fp16(
    const half* d_Q,
    const half* d_K,
    const half* d_V,
    half* d_O,
    const AttentionConfig& config,
    cudaStream_t stream
) {
    int B = config.batch_size;
    int H = config.num_heads;
    int N = config.seq_len_q;
    int M = config.seq_len_kv;
    int d = config.head_dim;

    if (d == 64) {
        constexpr int BM = 64, BN = 64, HD = 64;
        dim3 grid((N + BM - 1) / BM, 1, B * H);
        dim3 block(128);
        flash_attention_2_forward_kernel_fp16<BM, BN, HD><<<grid, block, 0, stream>>>(
            d_Q, d_K, d_V, d_O, B, H, N, M, config.softmax_scale, config.is_causal
        );
    } else if (d == 128) {
        constexpr int BM = 32, BN = 64, HD = 128;
        dim3 grid((N + BM - 1) / BM, 1, B * H);
        dim3 block(128);
        flash_attention_2_forward_kernel_fp16<BM, BN, HD><<<grid, block, 0, stream>>>(
            d_Q, d_K, d_V, d_O, B, H, N, M, config.softmax_scale, config.is_causal
        );
    } else {
        constexpr int BM = 64, BN = 64, HD = 32;
        dim3 grid((N + BM - 1) / BM, 1, B * H);
        dim3 block(128);
        flash_attention_2_forward_kernel_fp16<BM, BN, HD><<<grid, block, 0, stream>>>(
            d_Q, d_K, d_V, d_O, B, H, N, M, config.softmax_scale, config.is_causal
        );
    }

    return cudaGetLastError();
}

} // namespace attention
} // namespace flash_engine
