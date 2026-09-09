#include "quantized_gemm.cuh"
#include <cuda_fp16.h>

namespace flash_engine {
namespace quantized {

// =============================================================================
// INT8 GEMM KERNEL WITH DP4A INTRINSICS
// Accumulates 4x INT8 multiplies into a 32-bit integer in a single clock cycle:
// c = __dp4a(a_vec4, b_vec4, c)
// =============================================================================
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;

__global__ void __launch_bounds__(256) int8_gemm_dp4a_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha
) {
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    int accum = 0;

    // Loop across K in chunks of 4 (packed as 32-bit int)
    int k_packed = K / 4;
    const int* A_packed = reinterpret_cast<const int*>(A);
    const int* B_packed = reinterpret_cast<const int*>(B);

    if (row < M && col < N) {
        for (int k = 0; k < k_packed; ++k) {
            int a_val = A_packed[row * k_packed + k];
            int b_val = B_packed[col * k_packed + k]; // column-packed B for coalesced access

            // Hardware DP4A instruction
            #if __CUDA_ARCH__ >= 610
                accum = __dp4a(a_val, b_val, accum);
            #else
                // Software fallback for older compute capabilities
                const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&a_val);
                const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&b_val);
                accum += a_bytes[0] * b_bytes[0] +
                         a_bytes[1] * b_bytes[1] +
                         a_bytes[2] * b_bytes[2] +
                         a_bytes[3] * b_bytes[3];
            #endif
        }

        // Dequantize and store float32
        C[row * N + col] = static_cast<float>(accum) * alpha;
    }
}

cudaError_t launch_int8_gemm_dp4a(
    const int8_t* d_A,
    const int8_t* d_B,
    float* d_C,
    float scale_a,
    float scale_b,
    const QuantGEMMConfig& config,
    cudaStream_t stream
) {
    dim3 grid((config.N + TILE_N - 1) / TILE_N, (config.M + TILE_M - 1) / TILE_M);
    dim3 block(TILE_N, TILE_M); // 16 * 16 = 256 threads

    float combined_scale = scale_a * scale_b;

    int8_gemm_dp4a_kernel<<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, config.M, config.N, config.K, combined_scale
    );

    return cudaGetLastError();
}

// =============================================================================
// FP8 (E4M3) EMULATION / SCALED GEMM
// Decodes 8-bit float: sign(1), exponent(4), mantissa(3) -> float32
// =============================================================================
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x) {
    int sign = (x >> 7) & 0x1;
    int exp  = (x >> 3) & 0xF;
    int mant = x & 0x7;

    if (exp == 0) {
        // Subnormal
        float val = static_cast<float>(mant) / 8.0f * (1.0f / 64.0f);
        return sign ? -val : val;
    } else if (exp == 15 && mant == 7) {
        return 0.0f; // NaN/Inf clamp
    } else {
        // Normal
        float val = (1.0f + static_cast<float>(mant) / 8.0f) * powf(2.0f, static_cast<float>(exp - 7));
        return sign ? -val : val;
    }
}

__global__ void fp8_gemm_scaled_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = fp8_e4m3_to_float(A[row * K + k]);
            float b = fp8_e4m3_to_float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = sum * alpha;
    }
}

cudaError_t launch_fp8_gemm_scaled(
    const uint8_t* d_A,
    const uint8_t* d_B,
    float* d_C,
    float scale_a,
    float scale_b,
    const QuantGEMMConfig& config,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((config.N + 15) / 16, (config.M + 15) / 16);

    float combined_scale = scale_a * scale_b;

    fp8_gemm_scaled_kernel<<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, config.M, config.N, config.K, combined_scale
    );

    return cudaGetLastError();
}

} // namespace quantized
} // namespace flash_engine
