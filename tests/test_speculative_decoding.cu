#include "speculative_decoding.cuh"
#include <iostream>
#include <vector>
#include <cassert>

using namespace flash_engine;
using namespace flash_engine::speculative;

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "[INFO] No CUDA GPU found. Skipping speculative decoding test." << std::endl;
        return 0;
    }

    std::cout << "[TEST] Running Speculative Decoding Parallel Verification Test..." << std::endl;

    constexpr int B = 2;
    constexpr int K = 3; // 3 draft tokens, K+1 = 4 target positions
    constexpr int V = 32;

    SpeculativeVerifyConfig config(B, K, V, 0.0f, false);

    // Draft tokens:
    // Seq 0: [5, 12, 20]
    // Seq 1: [8, 15, 25]
    std::vector<int64_t> h_draft_tokens = {5, 12, 20, 8, 15, 25};

    // Target logits: [B, K + 1, V]
    std::vector<float> h_target_logits(B * (K + 1) * V, 0.0f);

    // Setup ground truth argmaxes:
    // Seq 0:
    //  Pos 0: argmax = 5  (match draft 0)
    //  Pos 1: argmax = 12 (match draft 1)
    //  Pos 2: argmax = 20 (match draft 2)
    //  Pos 3: argmax = 7  (bonus token!)
    //  -> Expected accepted count: 4, tokens: [5, 12, 20, 7]
    h_target_logits[(0 * (K + 1) + 0) * V + 5] = 10.0f;
    h_target_logits[(0 * (K + 1) + 1) * V + 12] = 10.0f;
    h_target_logits[(0 * (K + 1) + 2) * V + 20] = 10.0f;
    h_target_logits[(0 * (K + 1) + 3) * V + 7] = 10.0f;

    // Seq 1:
    //  Pos 0: argmax = 8  (match draft 0)
    //  Pos 1: argmax = 30 (mismatch! draft 1 was 15) -> Recovery token = 30, terminate!
    //  Pos 2: argmax = 25
    //  Pos 3: argmax = 2
    //  -> Expected accepted count: 2, tokens: [8, 30]
    h_target_logits[(1 * (K + 1) + 0) * V + 8] = 10.0f;
    h_target_logits[(1 * (K + 1) + 1) * V + 30] = 10.0f;
    h_target_logits[(1 * (K + 1) + 2) * V + 25] = 10.0f;
    h_target_logits[(1 * (K + 1) + 3) * V + 2] = 10.0f;

    // Device memory
    float* d_logits;
    int64_t* d_draft;
    int64_t* d_accepted;
    int* d_counts;

    CUDA_CHECK(cudaMalloc(&d_logits, h_target_logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_draft, h_draft_tokens.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_accepted, B * (K + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, B * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_logits, h_target_logits.data(), h_target_logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft, h_draft_tokens.data(), h_draft_tokens.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    cudaError_t status = launch_speculative_verification(
        d_logits, d_draft, nullptr, nullptr, d_accepted, d_counts, config
    );
    CUDA_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int64_t> h_accepted(B * (K + 1), 0);
    std::vector<int> h_counts(B, 0);

    CUDA_CHECK(cudaMemcpy(h_accepted.data(), d_accepted, h_accepted.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_draft));
    CUDA_CHECK(cudaFree(d_accepted));
    CUDA_CHECK(cudaFree(d_counts));

    std::cout << "[INFO] Seq 0 Accepted Count: " << h_counts[0] << " (Expected: 4)" << std::endl;
    std::cout << "[INFO] Seq 1 Accepted Count: " << h_counts[1] << " (Expected: 2)" << std::endl;

    assert(h_counts[0] == 4);
    assert(h_accepted[0 * (K + 1) + 0] == 5);
    assert(h_accepted[0 * (K + 1) + 1] == 12);
    assert(h_accepted[0 * (K + 1) + 2] == 20);
    assert(h_accepted[0 * (K + 1) + 3] == 7);

    assert(h_counts[1] == 2);
    assert(h_accepted[1 * (K + 1) + 0] == 8);
    assert(h_accepted[1 * (K + 1) + 1] == 30); // recovered prediction

    std::cout << "[PASS] Speculative Decoding Parallel Verification verified successfully!" << std::endl;
    return 0;
}
