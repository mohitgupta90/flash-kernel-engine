#include "flash_engine.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace flash_engine::norm;

// Standard naive two-pass RMSNorm for baseline
__global__ void naive_rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int total_tokens,
    int hidden_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= total_tokens) return;

    const float* x = input + row * hidden_dim;
    float* y = output + row * hidden_dim;

    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rsqrt_val = rsqrtf(sum_sq / static_cast<float>(hidden_dim) + eps);

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        y[i] = (x[i] * rsqrt_val) * gamma[i];
    }
}

void run_norm_benchmark(int tokens, int hidden_dim) {
    RMSNormConfig config(tokens, hidden_dim, 1e-5f);
    size_t total_elements = static_cast<size_t>(tokens) * hidden_dim;
    size_t bytes = total_elements * sizeof(float);
    size_t gamma_bytes = hidden_dim * sizeof(float);

    std::cout << "\n========================================================================\n";
    std::cout << " Benchmark: Fused Single-Pass RMSNorm vs Naive Baseline\n";
    std::cout << " Tokens=" << tokens << ", HiddenDim=" << hidden_dim << "\n";
    std::cout << " Total Data Size: " << (bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "========================================================================\n";

    std::vector<float> h_in(total_elements, 1.0f);
    std::vector<float> h_gamma(hidden_dim, 1.0f);
    std::vector<float> h_out_naive(total_elements, 0.0f);
    std::vector<float> h_out_fused(total_elements, 0.0f);

    float *d_in, *d_gamma, *d_out_naive, *d_out_fused;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, gamma_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_naive, bytes));
    CUDA_CHECK(cudaMalloc(&d_out_fused, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), gamma_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iters = 50;

    // 1. Naive Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        naive_rmsnorm_kernel<<<tokens, 256>>>(d_in, d_gamma, d_out_naive, tokens, hidden_dim, 1e-5f);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    ms_naive /= iters;

    // 2. Fused Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch_fused_rmsnorm_fp32(d_in, d_gamma, d_out_fused, config);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_fused = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_fused, start, stop));
    ms_fused /= iters;

    // Bandwidth: Read input (bytes) + Read gamma (gamma_bytes) + Write output (bytes) = 2 * bytes + gamma_bytes
    double total_io_bytes = 2.0 * bytes + gamma_bytes;
    double gbps_naive = (total_io_bytes / (ms_naive * 1e-3)) / 1e9;
    double gbps_fused = (total_io_bytes / (ms_fused * 1e-3)) / 1e9;

    std::cout << std::left << std::setw(20) << "Kernel"
              << std::setw(15) << "Latency (ms)"
              << std::setw(20) << "Bandwidth (GB/s)"
              << std::setw(15) << "Speedup" << "\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Naive RMSNorm"
              << std::setw(15) << ms_naive
              << std::setw(20) << gbps_naive
              << std::setw(15) << "1.00x" << "\n";
    std::cout << std::left << std::setw(20) << "Fused RMSNorm"
              << std::setw(15) << ms_fused
              << std::setw(20) << gbps_fused
              << std::setw(15) << (std::to_string(ms_naive / ms_fused) + "x") << "\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_out_naive));
    CUDA_CHECK(cudaFree(d_out_fused));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    run_norm_benchmark(4096, 2048); // Small model (e.g. 7B prompt phase)
    run_norm_benchmark(4096, 4096); // LLaMA-2/3 7B/8B hidden size
    run_norm_benchmark(2048, 8192); // LLaMA-3 70B hidden size
    return 0;
}
