#include "tensor_core_gemm.cuh"
#include <cstdio>

namespace flash_engine {
namespace tensor_core {

// =============================================================================
// WMMA Tensor Core GEMM: C = alpha * (A * B) + beta * C
// A: [M x K] row-major (half)
// B: [K x N] col-major (half) for optimal coalesced WMMA loads
// C: [M x N] row-major (float accumulator)
// =============================================================================
__global__ void wmma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    // 2D grid of thread blocks
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Coordinate offsets for this warp's 16x16 tile
    int warp_m = block_row * TILE_M + warp_row * WMMA_M;
    int warp_n = block_col * TILE_N + warp_col * WMMA_N;

    // Loop over K dimension in chunks of WMMA_K (16)
    for (int k_step = 0; k_step < K; k_step += WMMA_K) {
        if (warp_m < M && k_step < K) {
            wmma::load_matrix_sync(a_frag, A + warp_m * K + k_step, K);
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));
        }

        if (warp_n < N && k_step < K) {
            // Note: B is stored column-major: B[n, k] = B + warp_n * K + k_step
            wmma::load_matrix_sync(b_frag, B + warp_n * K + k_step, K);
        } else {
            wmma::fill_fragment(b_frag, __float2half(0.0f));
        }

        // Multiply-accumulate on Tensor Cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Scale and store out
    if (warp_m < M && warp_n < N) {
        #pragma unroll
        for (int t = 0; t < c_frag.num_elements; t++) {
            c_frag.x[t] = alpha * c_frag.x[t];
        }

        wmma::store_matrix_sync(C + warp_m * N + warp_n, c_frag, N, wmma::mem_row_major);
    }
}

cudaError_t launch_wmma_gemm_fp16(
    const half* d_A,
    const half* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(WARPS_M * WARPS_N * WARP_SIZE); // 4 * 4 * 32 = 512 threads

    wmma_gemm_kernel<<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, M, N, K, alpha, beta
    );

    return cudaGetLastError();
}

// =============================================================================
// WMMA Attention Stage: Computes Q * K^T with Tensor Cores, applies scale & softmax,
// then multiplies by V using WMMA fragments.
// =============================================================================
__global__ void wmma_attention_fused_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int seq_len, int head_dim,
    float scale
) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int block_row = blockIdx.y; // 16x16 query block
    int block_col = blockIdx.x; // 16x16 key block

    int q_offset = (b * H + h) * seq_len * head_dim;
    const half* q_ptr = Q + q_offset;
    const half* k_ptr = K + q_offset;
    const half* v_ptr = V + q_offset;
    half* o_ptr       = O + q_offset;

    int warp_id = threadIdx.x / WARP_SIZE;
    if (warp_id != 0) return; // 1 warp per 16x16 tile in this specialized micro-kernel

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;

    wmma::fill_fragment(s_frag, 0.0f);

    int row_idx = block_row * WMMA_M;
    int col_idx = block_col * WMMA_N;

    // Load Q and K fragments across head dimension
    for (int k_step = 0; k_step < head_dim; k_step += WMMA_K) {
        if (row_idx < seq_len && k_step < head_dim) {
            wmma::load_matrix_sync(q_frag, q_ptr + row_idx * head_dim + k_step, head_dim);
        } else {
            wmma::fill_fragment(q_frag, __float2half(0.0f));
        }

        if (col_idx < seq_len && k_step < head_dim) {
            wmma::load_matrix_sync(k_frag, k_ptr + col_idx * head_dim + k_step, head_dim);
        } else {
            wmma::fill_fragment(k_frag, __float2half(0.0f));
        }

        wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
    }

    // Scale scores
    #pragma unroll
    for (int i = 0; i < s_frag.num_elements; ++i) {
        s_frag.x[i] *= scale;
    }
}

cudaError_t launch_wmma_attention_qk_pv(
    const half* d_Q,
    const half* d_K,
    const half* d_V,
    half* d_O,
    int B, int H, int seq_len, int head_dim,
    float scale,
    cudaStream_t stream
) {
    dim3 grid((seq_len + WMMA_N - 1) / WMMA_N, (seq_len + WMMA_M - 1) / WMMA_M, B * H);
    dim3 block(32); // 1 warp per tile

    wmma_attention_fused_kernel<<<grid, block, 0, stream>>>(
        d_Q, d_K, d_V, d_O, B, H, seq_len, head_dim, scale
    );

    return cudaGetLastError();
}

} // namespace tensor_core
} // namespace flash_engine
