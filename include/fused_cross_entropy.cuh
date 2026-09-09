#pragma once

#include "cuda_utils.cuh"
#include <cstdint>

namespace flash_engine {
namespace loss {

// Configuration for Fused Cross Entropy
struct CrossEntropyConfig {
    int batch_size;       // Number of sequences or batch * seq_len
    int vocab_size;       // Vocabulary size (V) e.g. 32000, 128256
    float label_smoothing;// Label smoothing factor (0.0f = standard cross-entropy)
    int ignore_index;     // Target index to ignore (-100 standard in PyTorch)

    CrossEntropyConfig(int b, int v, float smoothing = 0.0f, int ignore = -100)
        : batch_size(b), vocab_size(v), label_smoothing(smoothing), ignore_index(ignore) {}
};

// Host dispatch for Fused Online Cross-Entropy forward pass
// Computes per-token cross entropy loss with online log-sum-exp in a single kernel pass
cudaError_t launch_fused_cross_entropy_forward(
    const float* d_logits,              // [batch_size, vocab_size]
    const int64_t* d_targets,           // [batch_size]
    float* d_losses,                    // [batch_size]
    float* d_total_loss,                // Scalar total loss (optional, nullptr if unused)
    const CrossEntropyConfig& config,
    cudaStream_t stream = nullptr
);

// Host dispatch for Fused Online Cross-Entropy backward pass (gradient calculation)
// Computes d_logits = (softmax(logits) - 1(target)) / N
cudaError_t launch_fused_cross_entropy_backward(
    const float* d_logits,              // [batch_size, vocab_size]
    const int64_t* d_targets,           // [batch_size]
    const float* d_grad_output,         // [batch_size] or scalar
    float* d_grad_logits,               // [batch_size, vocab_size] output gradient
    const CrossEntropyConfig& config,
    cudaStream_t stream = nullptr
);

} // namespace loss
} // namespace flash_engine
