#include "flash_decoding.cuh"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace flash_engine;
using namespace flash_engine::decoding;

void run_decoding_benchmark(int batch_size, int num_heads, int seq_len_kv, int head_dim, int num_splits) {
    FlashDecodingConfig config(batch_size, num_heads, seq_len_kv, head_dim, num_splits);

    size_t q_size = static_cast<size_t>(batch_size) * num_heads * head_dim;
    size_t kv_size = static_cast<size_t>(batch_size) * num_heads * seq_len_kv * head_dim;
    size_t out_size = q_size;
    size_t scratch_out_size = static_cast<size_t>(batch_size) * num_heads * num_splits * head_dim;
    size_t scratch_stats_size = static_cast<size_t>(batch_size) * num_heads * num_splits * 2;

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    float *d_scratch_out = nullptr, *d_scratch_stats = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_out, scratch_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_stats, scratch_stats_size * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_q, 0, q_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_k, 0, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v, 0, kv_size * sizeof(float)));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        launch_flash_decoding_forward_fp32(d_q, d_k, d_v, d_out, d_scratch_out, d_scratch_stats, config);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing with CUDA Events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int ITERS = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        launch_flash_decoding_forward_fp32(d_q, d_k, d_v, d_out, d_scratch_out, d_scratch_stats, config);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;

    // Memory read: Q (small) + K (large) + V (large), Write: Out
    double bytes_transferred = (static_cast<double>(kv_size) * 2.0 + q_size + out_size) * sizeof(float);
    double bandwidth_gb_s = (bytes_transferred / (avg_ms * 1e-3)) / 1e9;

    std::cout << std::setw(8) << batch_size
              << std::setw(12) << seq_len_kv
              << std::setw(10) << num_splits
              << std::setw(14) << std::fixed << std::setprecision(3) << avg_ms
              << std::setw(18) << std::fixed << std::setprecision(2) << bandwidth_gb_s
              << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scratch_out));
    CUDA_CHECK(cudaFree(d_scratch_stats));
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "[INFO] No CUDA-capable GPU found. Skipping hardware execution." << std::endl;
        return 0;
    }

    std::cout << "========================================================================\n";
    std::cout << "  Flash-Kernel-Engine: Split-KV FlashDecoding Autoregressive Benchmark  \n";
    std::cout << "========================================================================\n";
    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "ContextLen"
              << std::setw(10) << "Splits"
              << std::setw(14) << "Time (ms)"
              << std::setw(18) << "Bandwidth (GB/s)"
              << std::endl;
    std::cout << std::string(62, '-') << std::endl;

    constexpr int HEADS = 32;
    constexpr int HEAD_DIM = 128;

    std::vector<int> context_lengths = {1024, 2048, 4096, 8192, 16384, 32768};
    for (int seq_len : context_lengths) {
        int splits = (seq_len >= 16384) ? 16 : 8;
        run_decoding_benchmark(4, HEADS, seq_len, HEAD_DIM, splits);
    }

    std::cout << "========================================================================\n";
    return 0;
}
