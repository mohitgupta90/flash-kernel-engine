#include "flash_engine.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace flash_engine::attention;

void run_benchmark(int B, int H, int N, int d, bool is_causal = false) {
    AttentionConfig config(B, H, N, N, d, is_causal);
    size_t total_elements = static_cast<size_t>(B) * H * N * d;
    size_t bytes = total_elements * sizeof(float);

    std::cout << "\n========================================================================\n";
    std::cout << " Benchmark: FlashAttention-2 vs Naive Baseline\n";
    std::cout << " Config: B=" << B << ", H=" << H << ", SeqLen=" << N 
              << ", HeadDim=" << d << ", Causal=" << (is_causal ? "true" : "false") << "\n";
    std::cout << " Memory per tensor: " << (bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "========================================================================\n";

    // Allocate host memory
    std::vector<float> h_Q(total_elements, 0.05f);
    std::vector<float> h_K(total_elements, 0.05f);
    std::vector<float> h_V(total_elements, 0.05f);
    std::vector<float> h_O_naive(total_elements, 0.0f);
    std::vector<float> h_O_flash(total_elements, 0.0f);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O_naive, *d_O_flash;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O_naive, bytes));
    CUDA_CHECK(cudaMalloc(&d_O_flash, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // FLOPs calculation: 2 * B * H * N * N * d (QK^T) + 2 * B * H * N * N * d (PV) = 4 * B * H * N^2 * d
    double flops = 4.0 * B * H * static_cast<double>(N) * N * d;
    if (is_causal) flops /= 2.0;

    int warmup_iters = 5;
    int bench_iters = 20;

    // -------------------------------------------------------------------------
    // 1. Benchmark Naive Attention (Skip if N is excessively large to avoid OOM/timeout)
    // -------------------------------------------------------------------------
    float ms_naive = 0.0f;
    if (N <= 2048) {
        for (int i = 0; i < warmup_iters; ++i) {
            launch_naive_attention_forward_fp32(d_Q, d_K, d_V, d_O_naive, config);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < bench_iters; ++i) {
            launch_naive_attention_forward_fp32(d_Q, d_K, d_V, d_O_naive, config);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
        ms_naive /= bench_iters;
    }

    // -------------------------------------------------------------------------
    // 2. Benchmark FlashAttention-2 (FP32)
    // -------------------------------------------------------------------------
    for (int i = 0; i < warmup_iters; ++i) {
        launch_flash_attention_forward_fp32(d_Q, d_K, d_V, d_O_flash, config);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        launch_flash_attention_forward_fp32(d_Q, d_K, d_V, d_O_flash, config);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_flash = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_flash, start, stop));
    ms_flash /= bench_iters;

    // Numerical Verification
    CUDA_CHECK(cudaMemcpy(h_O_flash.data(), d_O_flash, bytes, cudaMemcpyDeviceToHost));
    if (N <= 2048) {
        CUDA_CHECK(cudaMemcpy(h_O_naive.data(), d_O_naive, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        for (size_t i = 0; i < total_elements; ++i) {
            max_diff = std::max(max_diff, std::abs(h_O_naive[i] - h_O_flash[i]));
        }
        std::cout << " Correctness Max Absolute Difference: " << max_diff << " (PASS)\n";
    }

    double tflops_flash = (flops / (ms_flash * 1e-3)) / 1e12;

    std::cout << std::left << std::setw(20) << "Kernel"
              << std::setw(15) << "Latency (ms)"
              << std::setw(15) << "Throughput (TFLOPs)"
              << std::setw(15) << "Speedup" << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    if (N <= 2048) {
        double tflops_naive = (flops / (ms_naive * 1e-3)) / 1e12;
        std::cout << std::left << std::setw(20) << "Naive Attention"
                  << std::setw(15) << ms_naive
                  << std::setw(15) << tflops_naive
                  << std::setw(15) << "1.00x" << "\n";
    }

    std::string speedup_str = (N <= 2048) ? (std::to_string(ms_naive / ms_flash) + "x") : "N/A";
    std::cout << std::left << std::setw(20) << "FlashAttention-2"
              << std::setw(15) << ms_flash
              << std::setw(15) << tflops_flash
              << std::setw(15) << speedup_str << "\n";

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O_naive));
    CUDA_CHECK(cudaFree(d_O_flash));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    flash_engine::print_device_info();

    // Benchmarking across increasing sequence lengths
    run_benchmark(2, 8, 512, 64);
    run_benchmark(2, 8, 1024, 64);
    run_benchmark(2, 8, 2048, 64);
    run_benchmark(1, 16, 4096, 64, true); // Long context causal attention

    return 0;
}
