#include "fused_rmsnorm.cuh"
#include <cuda_fp16.h>

namespace flash_engine {
namespace norm {

// =============================================================================
// FUSED RMSNORM FORWARD KERNEL (FP32)
// Formula: y = (x / sqrt(mean(x^2) + eps)) * gamma
// Single pass over global memory: Loads x once, reduces sum of squares in SRAM
// via warp-shuffles, normalizes and scales, writes out with 128-bit float4.
// =============================================================================
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE) fused_rmsnorm_kernel_fp32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int total_tokens,
    int hidden_dim,
    float epsilon
) {
    int row = blockIdx.x;
    if (row >= total_tokens) return;

    const float* x_row = input + row * hidden_dim;
    float* y_row = output + row * hidden_dim;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ float s_warp_sums[NUM_WARPS];

    // 1. Accumulate sum of squares in registers
    float sum_sq = 0.0f;

    // Vectorized 128-bit stride if hidden_dim is a multiple of 4
    if (hidden_dim % 4 == 0) {
        int vec_len = hidden_dim / 4;
        const float4* x_vec = reinterpret_cast<const float4*>(x_row);

        for (int i = tid; i < vec_len; i += BLOCK_SIZE) {
            float4 v = x_vec[i];
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
            float val = x_row[i];
            sum_sq += val * val;
        }
    }

    // 2. Intra-warp reduction using register shuffle instructions (__shfl_down_sync)
    sum_sq = flash_engine::cuda::warp_reduce_sum(sum_sq);

    // Write warp sum to shared memory
    if (lane == 0) {
        s_warp_sums[wid] = sum_sq;
    }
    __syncthreads();

    // 3. Inter-warp reduction by the first warp
    float block_sum = 0.0f;
    if (wid == 0) {
        block_sum = (lane < NUM_WARPS) ? s_warp_sums[lane] : 0.0f;
        block_sum = flash_engine::cuda::warp_reduce_sum(block_sum);
        if (lane == 0) {
            // Compute rsqrt(mean + eps)
            float mean_sq = block_sum / static_cast<float>(hidden_dim);
            s_warp_sums[0] = rsqrtf(mean_sq + epsilon);
        }
    }
    __syncthreads();

    // 4. Broadcast reciprocal standard deviation to all threads
    float rsqrt_val = s_warp_sums[0];

    // 5. Vectorized write-back: output = (x * rsqrt) * gamma
    if (hidden_dim % 4 == 0) {
        int vec_len = hidden_dim / 4;
        const float4* x_vec = reinterpret_cast<const float4*>(x_row);
        const float4* g_vec = reinterpret_cast<const float4*>(gamma);
        float4* y_vec = reinterpret_cast<float4*>(y_row);

        for (int i = tid; i < vec_len; i += BLOCK_SIZE) {
            float4 x_val = x_vec[i];
            float4 g_val = g_vec[i];
            float4 out;
            out.x = (x_val.x * rsqrt_val) * g_val.x;
            out.y = (x_val.y * rsqrt_val) * g_val.y;
            out.z = (x_val.z * rsqrt_val) * g_val.z;
            out.w = (x_val.w * rsqrt_val) * g_val.w;
            y_vec[i] = out;
        }
    } else {
        for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
            y_row[i] = (x_row[i] * rsqrt_val) * gamma[i];
        }
    }
}

// =============================================================================
// FUSED RMSNORM FORWARD KERNEL (FP16 / Half)
// =============================================================================
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE) fused_rmsnorm_kernel_fp16(
    const half* __restrict__ input,
    const half* __restrict__ gamma,
    half* __restrict__ output,
    int total_tokens,
    int hidden_dim,
    float epsilon
) {
    int row = blockIdx.x;
    if (row >= total_tokens) return;

    const half* x_row = input + row * hidden_dim;
    half* y_row = output + row * hidden_dim;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    __shared__ float s_warp_sums[NUM_WARPS];

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = __half2float(x_row[i]);
        sum_sq += val * val;
    }

    sum_sq = flash_engine::cuda::warp_reduce_sum(sum_sq);

    if (lane == 0) {
        s_warp_sums[wid] = sum_sq;
    }
    __syncthreads();

    if (wid == 0) {
        float b_sum = (lane < NUM_WARPS) ? s_warp_sums[lane] : 0.0f;
        b_sum = flash_engine::cuda::warp_reduce_sum(b_sum);
        if (lane == 0) {
            float mean_sq = b_sum / static_cast<float>(hidden_dim);
            s_warp_sums[0] = rsqrtf(mean_sq + epsilon);
        }
    }
    __syncthreads();

    float rsqrt_val = s_warp_sums[0];

    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float x_val = __half2float(x_row[i]);
        float g_val = __half2float(gamma[i]);
        y_row[i] = __float2half((x_val * rsqrt_val) * g_val);
    }
}

// Host Dispatch FP32
cudaError_t launch_fused_rmsnorm_fp32(
    const float* d_input,
    const float* d_gamma,
    float* d_output,
    const RMSNormConfig& config,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    dim3 grid(config.batch_size);
    dim3 block(BLOCK_SIZE);

    fused_rmsnorm_kernel_fp32<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        d_input, d_gamma, d_output,
        config.batch_size, config.hidden_dim, config.epsilon
    );

    return cudaGetLastError();
}

// Host Dispatch FP16
cudaError_t launch_fused_rmsnorm_fp16(
    const half* d_input,
    const half* d_gamma,
    half* d_output,
    const RMSNormConfig& config,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    dim3 grid(config.batch_size);
    dim3 block(BLOCK_SIZE);

    fused_rmsnorm_kernel_fp16<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        d_input, d_gamma, d_output,
        config.batch_size, config.hidden_dim, config.epsilon
    );

    return cudaGetLastError();
}

} // namespace norm
} // namespace flash_engine
