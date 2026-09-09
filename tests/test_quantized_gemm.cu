#include "quantized_gemm.cuh"
#include <iostream>
#include <vector>
#include <cmath>

using namespace flash_engine;
using namespace flash_engine::quantized;

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "[INFO] No CUDA GPU found. Skipping quantized GEMM test." << std::endl;
        return 0;
    }

    std::cout << "[TEST] Running INT8 DP4A Quantized GEMM Correctness Test..." << std::endl;

    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 64;
    QuantGEMMConfig config(M, N, K);

    std::vector<int8_t> h_A(M * K);
    std::vector<int8_t> h_B(N * K); // column-packed: col * K + k
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<int8_t>((i % 7) - 3);
    }
    for (int i = 0; i < N * K; ++i) {
        h_B[i] = static_cast<int8_t>((i % 5) - 2);
    }

    float scale_a = 0.05f;
    float scale_b = 0.02f;

    // CPU reference: C[m, n] = sum_k(A[m, k] * B[n, k]) * scale
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(h_A[m * K + k]) * static_cast<int32_t>(h_B[n * K + k]);
            }
            h_ref[m * N + n] = static_cast<float>(acc) * (scale_a * scale_b);
        }
    }

    // Device allocation
    int8_t *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    cudaError_t status = launch_int8_gemm_dp4a(d_A, d_B, d_C, scale_a, scale_b, config);
    CUDA_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (size_t i = 0; i < h_C.size(); ++i) {
        float diff = std::fabs(h_C[i] - h_ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout << "[INFO] Max absolute difference: " << max_diff << std::endl;
    if (max_diff < 1e-3f) {
        std::cout << "[PASS] INT8 DP4A GEMM verified against reference within numerical tolerance." << std::endl;
        return 0;
    } else {
        std::cerr << "[FAIL] INT8 DP4A GEMM deviation too large: " << max_diff << std::endl;
        return 1;
    }
}
