#include "fused_activations.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace flash_engine {
namespace activation {

// Fast sigmoid approximation using __expf
__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// =============================================================================
// Fused SwiGLU Kernel (FP32) - Vectorized 128-bit
// output = (gate * sigmoid(gate)) * up
// =============================================================================
__global__ void fused_swiglu_kernel_fp32_vec4(
    const float4* __restrict__ gate,
    const float4* __restrict__ up,
    float4* __restrict__ out,
    int vec4_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec4_elements) {
        float4 g = gate[idx];
        float4 u = up[idx];
        float4 result;

        result.x = (g.x * fast_sigmoid(g.x)) * u.x;
        result.y = (g.y * fast_sigmoid(g.y)) * u.y;
        result.z = (g.z * fast_sigmoid(g.z)) * u.z;
        result.w = (g.w * fast_sigmoid(g.w)) * u.w;

        out[idx] = result;
    }
}

__global__ void fused_swiglu_kernel_fp32_tail(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int start_idx,
    int total_elements
) {
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = gate[idx];
        float u = up[idx];
        out[idx] = (g * fast_sigmoid(g)) * u;
    }
}

cudaError_t launch_fused_swiglu_fp32(
    const float* d_gate,
    const float* d_up,
    float* d_out,
    int num_elements,
    cudaStream_t stream
) {
    int vec4_count = num_elements / 4;
    int remainder = num_elements % 4;

    if (vec4_count > 0) {
        dim3 block(256);
        dim3 grid((vec4_count + 255) / 256);
        fused_swiglu_kernel_fp32_vec4<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float4*>(d_gate),
            reinterpret_cast<const float4*>(d_up),
            reinterpret_cast<float4*>(d_out),
            vec4_count
        );
    }

    if (remainder > 0) {
        dim3 block(32);
        dim3 grid(1);
        fused_swiglu_kernel_fp32_tail<<<grid, block, 0, stream>>>(
            d_gate, d_up, d_out, vec4_count * 4, num_elements
        );
    }

    return cudaGetLastError();
}

// =============================================================================
// Fused SwiGLU Kernel (FP16)
// =============================================================================
__global__ void fused_swiglu_kernel_fp16(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float res = (g * fast_sigmoid(g)) * u;
        out[idx] = __float2half(res);
    }
}

cudaError_t launch_fused_swiglu_fp16(
    const half* d_gate,
    const half* d_up,
    half* d_out,
    int num_elements,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_elements + 255) / 256);

    fused_swiglu_kernel_fp16<<<grid, block, 0, stream>>>(
        d_gate, d_up, d_out, num_elements
    );

    return cudaGetLastError();
}

// =============================================================================
// Fused GeLU Kernel (FP32) - Tanh approximation:
// 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
// =============================================================================
__device__ __forceinline__ float gelu_tanh_approx(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_gelu_kernel_fp32_vec4(
    const float4* __restrict__ in,
    float4* __restrict__ out,
    int vec4_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec4_elements) {
        float4 v = in[idx];
        float4 res;
        res.x = gelu_tanh_approx(v.x);
        res.y = gelu_tanh_approx(v.y);
        res.z = gelu_tanh_approx(v.z);
        res.w = gelu_tanh_approx(v.w);
        out[idx] = res;
    }
}

cudaError_t launch_fused_gelu_fp32(
    const float* d_in,
    float* d_out,
    int num_elements,
    cudaStream_t stream
) {
    int vec4_count = num_elements / 4;
    int remainder = num_elements % 4;

    if (vec4_count > 0) {
        dim3 block(256);
        dim3 grid((vec4_count + 255) / 256);
        fused_gelu_kernel_fp32_vec4<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float4*>(d_in),
            reinterpret_cast<float4*>(d_out),
            vec4_count
        );
    }

    if (remainder > 0) {
        // Handle tail
        dim3 block(32);
        dim3 grid(1);
        fused_swiglu_kernel_fp32_tail<<<grid, block, 0, stream>>>(
            d_in, d_in, d_out, vec4_count * 4, num_elements
        );
    }

    return cudaGetLastError();
}

} // namespace activation
} // namespace flash_engine
