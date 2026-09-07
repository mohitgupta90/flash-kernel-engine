#include "fused_rope.cuh"

namespace flash_engine {
namespace rope {

// =============================================================================
// FUSED ROTARY POSITION EMBEDDING (RoPE) KERNEL
// Input shape: [Batch, Heads, Seq_len, Head_Dim]
// Rotates adjacent dimension pairs (d_i, d_{i+1}) using precomputed cos/sin:
// x_rot_0 = x_0 * cos - x_1 * sin
// x_rot_1 = x_0 * sin + x_1 * cos
// =============================================================================
__global__ void fused_rope_kernel_fp32(
    const float* __restrict__ in,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ out,
    int B, int H, int seq_len, int head_dim
) {
    // Total tokens: B * H * seq_len
    int token_idx = blockIdx.x;
    int pair_idx = threadIdx.x; // each thread handles 1 pair of coordinates (head_dim / 2)

    int half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

    int pos = token_idx % seq_len;

    // Offset in input/output tensor
    size_t base_offset = static_cast<size_t>(token_idx) * head_dim;
    size_t cos_sin_offset = static_cast<size_t>(pos) * half_dim + pair_idx;

    float c = cos_table[cos_sin_offset];
    float s = sin_table[cos_sin_offset];

    int i0 = pair_idx * 2;
    int i1 = pair_idx * 2 + 1;

    float x0 = in[base_offset + i0];
    float x1 = in[base_offset + i1];

    out[base_offset + i0] = x0 * c - x1 * s;
    out[base_offset + i1] = x0 * s + x1 * c;
}

cudaError_t launch_fused_rope_fp32(
    const float* d_in,
    const float* d_cos,
    const float* d_sin,
    float* d_out,
    const RoPEConfig& config,
    cudaStream_t stream
) {
    int total_tokens = config.batch_size * config.num_heads * config.seq_len;
    int half_dim = config.head_dim / 2;

    dim3 grid(total_tokens);
    dim3 block(half_dim);

    fused_rope_kernel_fp32<<<grid, block, 0, stream>>>(
        d_in, d_cos, d_sin, d_out,
        config.batch_size, config.num_heads, config.seq_len, config.head_dim
    );

    return cudaGetLastError();
}

} // namespace rope
} // namespace flash_engine
