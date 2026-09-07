#pragma once

#include "cuda_utils.cuh"
#include <cuda_fp16.h>

namespace flash_engine {
namespace activation {

// Fused SwiGLU: output = (x * sigmoid(x)) * y
// Both gate (x) and up (y) are packed or contiguous of size num_elements
cudaError_t launch_fused_swiglu_fp32(
    const float* d_gate,
    const float* d_up,
    float* d_out,
    int num_elements,
    cudaStream_t stream = nullptr
);

cudaError_t launch_fused_swiglu_fp16(
    const half* d_gate,
    const half* d_up,
    half* d_out,
    int num_elements,
    cudaStream_t stream = nullptr
);

// Fused GeLU with tanh approximation: output = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
cudaError_t launch_fused_gelu_fp32(
    const float* d_in,
    float* d_out,
    int num_elements,
    cudaStream_t stream = nullptr
);

} // namespace activation
} // namespace flash_engine
