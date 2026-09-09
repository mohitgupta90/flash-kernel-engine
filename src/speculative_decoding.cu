#include "speculative_decoding.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace flash_engine {
namespace speculative {

// =============================================================================
// WARP REDUCTION FOR ARGMAX (VALUE + INDEX PAIR)
// =============================================================================
struct MaxPair {
    float val;
    int idx;
};

__device__ __forceinline__ MaxPair warp_reduce_argmax(MaxPair pair) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, pair.val, mask);
        int other_idx = __shfl_down_sync(0xffffffff, pair.idx, mask);
        if (other_val > pair.val || (other_val == pair.val && other_idx < pair.idx)) {
            pair.val = other_val;
            pair.idx = other_idx;
        }
    }
    return pair;
}

// =============================================================================
// PARALLEL SPECULATIVE VERIFICATION KERNEL
// Evaluates draft tokens against target logits in a single fused pass
// Grid: (batch_size)
// Block: 256 threads (8 warps)
// =============================================================================
template <int BLOCK_THREADS = 256>
__global__ void speculative_verification_kernel(
    const float* __restrict__ target_logits,  // [B, K + 1, V]
    const int64_t* __restrict__ draft_tokens, // [B, K]
    const float* __restrict__ draft_probs,    // [B, K] optional
    const float* __restrict__ rand_uniform,   // [B, K] optional
    int64_t* __restrict__ accepted_tokens,    // [B, K + 1]
    int* __restrict__ accepted_counts,        // [B]
    int K,                                    // num_draft_tokens
    int V,                                    // vocab_size
    float temperature,
    bool use_stochastic
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;

    __shared__ MaxPair s_warp_pairs[NUM_WARPS];
    __shared__ int s_accepted_count;
    __shared__ bool s_stop_speculation;

    if (tid == 0) {
        s_accepted_count = 0;
        s_stop_speculation = false;
    }
    __syncthreads();

    const int64_t* b_draft_tokens = draft_tokens + static_cast<size_t>(b) * K;
    int64_t* b_accepted = accepted_tokens + static_cast<size_t>(b) * (K + 1);

    // Verify draft positions sequentially from k = 0 to K - 1
    for (int k = 0; k < K; ++k) {
        if (s_stop_speculation) break;

        const float* logits_k = target_logits + (static_cast<size_t>(b) * (K + 1) + k) * V;

        // Step 1: Compute thread-local maximum logit and argmax index
        MaxPair thread_max = {-1e30f, 0};
        for (int v = tid; v < V; v += BLOCK_THREADS) {
            float val = logits_k[v];
            if (val > thread_max.val) {
                thread_max.val = val;
                thread_max.idx = v;
            }
        }

        // Intra-warp reduction
        MaxPair warp_max = warp_reduce_argmax(thread_max);
        if (lane == 0) {
            s_warp_pairs[wid] = warp_max;
        }
        __syncthreads();

        // Block-level reduction
        if (wid == 0) {
            MaxPair block_pair = (lane < NUM_WARPS) ? s_warp_pairs[lane] : MaxPair{-1e30f, -1};
            block_pair = warp_reduce_argmax(block_pair);
            if (lane == 0) {
                s_warp_pairs[0] = block_pair;
            }
        }
        __syncthreads();

        int target_argmax = s_warp_pairs[0].idx;
        int64_t draft_token = b_draft_tokens[k];

        // Step 2: Verification condition (thread 0 determines acceptance)
        if (tid == 0) {
            if (target_argmax == draft_token) {
                // Draft token accepted!
                b_accepted[s_accepted_count++] = draft_token;
            } else {
                // Draft token rejected!
                // Emit target model's corrected prediction as the recovery token
                b_accepted[s_accepted_count++] = target_argmax;
                s_stop_speculation = true;
            }
        }
        __syncthreads();
    }

    // Step 3: If all K draft tokens were accepted, sample bonus token from target position K
    if (!s_stop_speculation) {
        const float* logits_bonus = target_logits + (static_cast<size_t>(b) * (K + 1) + K) * V;

        MaxPair thread_max = {-1e30f, 0};
        for (int v = tid; v < V; v += BLOCK_THREADS) {
            float val = logits_bonus[v];
            if (val > thread_max.val) {
                thread_max.val = val;
                thread_max.idx = v;
            }
        }

        MaxPair warp_max = warp_reduce_argmax(thread_max);
        if (lane == 0) {
            s_warp_pairs[wid] = warp_max;
        }
        __syncthreads();

        if (wid == 0) {
            MaxPair block_pair = (lane < NUM_WARPS) ? s_warp_pairs[lane] : MaxPair{-1e30f, -1};
            block_pair = warp_reduce_argmax(block_pair);
            if (lane == 0) {
                s_warp_pairs[0] = block_pair;
            }
        }
        __syncthreads();

        if (tid == 0) {
            b_accepted[s_accepted_count++] = s_warp_pairs[0].idx;
        }
    }
    __syncthreads();

    // Step 4: Write final accepted count for sequence
    if (tid == 0) {
        accepted_counts[b] = s_accepted_count;
    }
}

// Host Dispatch for Speculative Verification
cudaError_t launch_speculative_verification(
    const float* d_target_logits,
    const int64_t* d_draft_tokens,
    const float* d_draft_probs,
    const float* d_rand_uniform,
    int64_t* d_accepted_tokens,
    int* d_accepted_counts,
    const SpeculativeVerifyConfig& config,
    cudaStream_t stream
) {
    if (config.batch_size <= 0 || config.num_draft_tokens <= 0 || config.vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int BLOCK_THREADS = 256;
    speculative_verification_kernel<BLOCK_THREADS><<<config.batch_size, BLOCK_THREADS, 0, stream>>>(
        d_target_logits,
        d_draft_tokens,
        d_draft_probs,
        d_rand_uniform,
        d_accepted_tokens,
        d_accepted_counts,
        config.num_draft_tokens,
        config.vocab_size,
        config.temperature,
        config.use_stochastic
    );

    return cudaGetLastError();
}

} // namespace speculative
} // namespace flash_engine
