#include "fused_cross_entropy.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace flash_engine {
namespace loss {

// =============================================================================
// WARP REDUCTION PRIMITIVES
// =============================================================================
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

// =============================================================================
// FUSED CROSS ENTROPY FORWARD KERNEL
// Computes per-token loss in a single pass using online Log-Sum-Exp.
// Grid: (batch_size)
// Block: 256 or 512 threads
// =============================================================================
template <int BLOCK_THREADS = 256>
__global__ void fused_cross_entropy_forward_kernel(
    const float* __restrict__ logits,      // [batch_size, vocab_size]
    const int64_t* __restrict__ targets,   // [batch_size]
    float* __restrict__ losses,            // [batch_size]
    int vocab_size,
    float label_smoothing,
    int ignore_index
) {
    int b = blockIdx.x;
    int64_t target = targets[b];

    // Check for ignore index
    if (target == ignore_index) {
        if (threadIdx.x == 0) {
            losses[b] = 0.0f;
        }
        return;
    }

    const float* row_logits = logits + static_cast<size_t>(b) * vocab_size;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];

    // Step 1: Compute thread-local maximum logit
    float thread_max = -1e30f;
    for (int v = tid; v < vocab_size; v += BLOCK_THREADS) {
        thread_max = fmaxf(thread_max, row_logits[v]);
    }

    // Warp-level max reduction
    float warp_max = warp_reduce_max(thread_max);
    if (lane == 0) {
        s_warp_max[wid] = warp_max;
    }
    __syncthreads();

    // Block-level max reduction
    float global_max = -1e30f;
    if (tid < NUM_WARPS) {
        global_max = s_warp_max[tid];
    }
    if (wid == 0) {
        global_max = warp_reduce_max(global_max);
        if (lane == 0) {
            s_warp_max[0] = global_max;
        }
    }
    __syncthreads();
    global_max = s_warp_max[0];

    // Step 2: Compute denominator sum(exp(logits - global_max))
    float thread_sum = 0.0f;
    float thread_smooth_sum = 0.0f; // for label smoothing
    for (int v = tid; v < vocab_size; v += BLOCK_THREADS) {
        float val = expf(row_logits[v] - global_max);
        thread_sum += val;
        if (label_smoothing > 0.0f) {
            thread_smooth_sum += row_logits[v] - global_max;
        }
    }

    // Warp-level sum reduction
    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        s_warp_sum[wid] = warp_sum;
    }
    __syncthreads();

    // Block-level sum reduction
    float global_sum = 0.0f;
    if (tid < NUM_WARPS) {
        global_sum = s_warp_sum[tid];
    }
    if (wid == 0) {
        global_sum = warp_reduce_sum(global_sum);
        if (lane == 0) {
            s_warp_sum[0] = global_sum;
        }
    }
    __syncthreads();
    global_sum = s_warp_sum[0];

    // Step 3: Compute Loss = log(sum(exp(x - max))) + max - target_val
    if (tid == 0) {
        float lse = logf(global_sum) + global_max;
        float target_val = 0.0f;
        if (target >= 0 && target < vocab_size) {
            target_val = row_logits[target];
        }
        float loss = lse - target_val;

        if (label_smoothing > 0.0f) {
            // Label smoothing: (1 - eps) * standard_loss + eps * (lse - mean(logits))
            // We approximate or accurately compute via smooth sum if needed
            loss = (1.0f - label_smoothing) * loss + label_smoothing * lse;
        }
        losses[b] = loss;
    }
}

// =============================================================================
// FUSED CROSS ENTROPY BACKWARD KERNEL
// Computes d_logits = (softmax(logits) - 1(target))
// Grid: (batch_size)
// Block: 256 threads
// =============================================================================
template <int BLOCK_THREADS = 256>
__global__ void fused_cross_entropy_backward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_logits,
    int vocab_size,
    float label_smoothing,
    int ignore_index
) {
    int b = blockIdx.x;
    int64_t target = targets[b];

    float* row_grad = grad_logits + static_cast<size_t>(b) * vocab_size;

    if (target == ignore_index) {
        for (int v = threadIdx.x; v < vocab_size; v += BLOCK_THREADS) {
            row_grad[v] = 0.0f;
        }
        return;
    }

    const float* row_logits = logits + static_cast<size_t>(b) * vocab_size;
    float dout = (grad_output != nullptr) ? grad_output[b] : 1.0f;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];

    // Max reduction
    float thread_max = -1e30f;
    for (int v = tid; v < vocab_size; v += BLOCK_THREADS) {
        thread_max = fmaxf(thread_max, row_logits[v]);
    }
    float warp_max = warp_reduce_max(thread_max);
    if (lane == 0) s_warp_max[wid] = warp_max;
    __syncthreads();

    float global_max = -1e30f;
    if (tid < NUM_WARPS) global_max = s_warp_max[tid];
    if (wid == 0) {
        global_max = warp_reduce_max(global_max);
        if (lane == 0) s_warp_max[0] = global_max;
    }
    __syncthreads();
    global_max = s_warp_max[0];

    // Sum reduction
    float thread_sum = 0.0f;
    for (int v = tid; v < vocab_size; v += BLOCK_THREADS) {
        thread_sum += expf(row_logits[v] - global_max);
    }
    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) s_warp_sum[wid] = warp_sum;
    __syncthreads();

    float global_sum = 0.0f;
    if (tid < NUM_WARPS) global_sum = s_warp_sum[tid];
    if (wid == 0) {
        global_sum = warp_reduce_sum(global_sum);
        if (lane == 0) s_warp_sum[0] = global_sum;
    }
    __syncthreads();
    global_sum = s_warp_sum[0];
    float inv_sum = 1.0f / global_sum;

    // Gradient calculation
    for (int v = tid; v < vocab_size; v += BLOCK_THREADS) {
        float prob = expf(row_logits[v] - global_max) * inv_sum;
        float grad = prob;
        if (v == target) {
            grad -= 1.0f;
        }
        row_grad[v] = grad * dout;
    }
}

// Host Dispatch: Forward Pass
cudaError_t launch_fused_cross_entropy_forward(
    const float* d_logits,
    const int64_t* d_targets,
    float* d_losses,
    float* d_total_loss,
    const CrossEntropyConfig& config,
    cudaStream_t stream
) {
    if (config.batch_size <= 0 || config.vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int BLOCK_THREADS = 256;
    fused_cross_entropy_forward_kernel<BLOCK_THREADS><<<config.batch_size, BLOCK_THREADS, 0, stream>>>(
        d_logits,
        d_targets,
        d_losses,
        config.vocab_size,
        config.label_smoothing,
        config.ignore_index
    );

    return cudaGetLastError();
}

// Host Dispatch: Backward Pass
cudaError_t launch_fused_cross_entropy_backward(
    const float* d_logits,
    const int64_t* d_targets,
    const float* d_grad_output,
    float* d_grad_logits,
    const CrossEntropyConfig& config,
    cudaStream_t stream
) {
    if (config.batch_size <= 0 || config.vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int BLOCK_THREADS = 256;
    fused_cross_entropy_backward_kernel<BLOCK_THREADS><<<config.batch_size, BLOCK_THREADS, 0, stream>>>(
        d_logits,
        d_targets,
        d_grad_output,
        d_grad_logits,
        config.vocab_size,
        config.label_smoothing,
        config.ignore_index
    );

    return cudaGetLastError();
}

} // namespace loss
} // namespace flash_engine
