#include "speculative_decoding.cuh"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace flash_engine;
using namespace flash_engine::speculative;

void run_speculative_bench(int batch_size, int num_draft_tokens, int vocab_size) {
    SpeculativeVerifyConfig config(batch_size, num_draft_tokens, vocab_size, 0.0f, false);

    size_t logits_size = static_cast<size_t>(batch_size) * (num_draft_tokens + 1) * vocab_size;
    size_t draft_size = static_cast<size_t>(batch_size) * num_draft_tokens;
    size_t accepted_size = static_cast<size_t>(batch_size) * (num_draft_tokens + 1);

    float* d_logits;
    int64_t* d_draft;
    int64_t* d_accepted;
    int* d_counts;

    CUDA_CHECK(cudaMalloc(&d_logits, logits_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_draft, draft_size * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_accepted, accepted_size * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, batch_size * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_logits, 0, logits_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_draft, 0, draft_size * sizeof(int64_t)));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        launch_speculative_verification(d_logits, d_draft, nullptr, nullptr, d_accepted, d_counts, config);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int ITERS = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        launch_speculative_verification(d_logits, d_draft, nullptr, nullptr, d_accepted, d_counts, config);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_us = (total_ms / ITERS) * 1000.0f; // microseconds

    double verified_tokens_per_sec = (static_cast<double>(batch_size) * (num_draft_tokens + 1)) / (avg_us * 1e-6);

    std::cout << std::setw(8) << batch_size
              << std::setw(12) << num_draft_tokens
              << std::setw(14) << vocab_size
              << std::setw(16) << std::fixed << std::setprecision(2) << avg_us
              << std::setw(22) << std::scientific << std::setprecision(2) << verified_tokens_per_sec
              << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_draft));
    CUDA_CHECK(cudaFree(d_accepted));
    CUDA_CHECK(cudaFree(d_counts));
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "[INFO] No CUDA GPU found. Skipping benchmark." << std::endl;
        return 0;
    }

    std::cout << "================================================================================\n";
    std::cout << "    Flash-Kernel-Engine: Speculative Decoding Parallel Verification Benchmark   \n";
    std::cout << "================================================================================\n";
    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "Draft (K)"
              << std::setw(14) << "Vocab (V)"
              << std::setw(16) << "Latency (us)"
              << std::setw(22) << "Tokens Verified/sec"
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    constexpr int VOCAB = 32000;
    std::vector<int> batches = {1, 4, 16};
    std::vector<int> draft_lengths = {2, 4, 8};

    for (int b : batches) {
        for (int k : draft_lengths) {
            run_speculative_bench(b, k, VOCAB);
        }
    }

    std::cout << "================================================================================\n";
    return 0;
}
