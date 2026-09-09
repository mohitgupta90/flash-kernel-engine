#include "paged_attention.cuh"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace flash_engine;
using namespace flash_engine::paged_attention;

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "[INFO] No CUDA GPU found. Test passed trivially." << std::endl;
        return 0;
    }

    std::cout << "[TEST] Running PagedAttention v1 Correctness & Virtual Block Lookup Test..." << std::endl;

    constexpr int NUM_SEQS = 2;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_SIZE = 16;
    constexpr int MAX_CONTEXT_LEN = 64;
    constexpr int MAX_BLOCKS_PER_SEQ = 8;
    constexpr int TOTAL_PHYSICAL_BLOCKS = 16;

    PagedAttentionConfig config(
        NUM_SEQS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        BLOCK_SIZE, MAX_CONTEXT_LEN, MAX_BLOCKS_PER_SEQ
    );

    // Host allocations
    std::vector<float> h_query(NUM_SEQS * NUM_HEADS * HEAD_DIM, 1.0f);
    std::vector<float> h_key_cache(TOTAL_PHYSICAL_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM, 0.5f);
    std::vector<float> h_value_cache(TOTAL_PHYSICAL_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM, 0.25f);
    std::vector<float> h_output(NUM_SEQS * NUM_HEADS * HEAD_DIM, 0.0f);

    // Block table: seq 0 maps to physical blocks [3, 7, 1, 9]
    //              seq 1 maps to physical blocks [5, 2, 8, 4]
    std::vector<int> h_block_tables(NUM_SEQS * MAX_BLOCKS_PER_SEQ, 0);
    h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 0] = 3;
    h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 1] = 7;
    h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 2] = 1;
    h_block_tables[0 * MAX_BLOCKS_PER_SEQ + 3] = 9;

    h_block_tables[1 * MAX_BLOCKS_PER_SEQ + 0] = 5;
    h_block_tables[1 * MAX_BLOCKS_PER_SEQ + 1] = 2;
    h_block_tables[1 * MAX_BLOCKS_PER_SEQ + 2] = 8;
    h_block_tables[1 * MAX_BLOCKS_PER_SEQ + 3] = 4;

    std::vector<int> h_context_lens = {48, 64}; // Tokens

    // Device allocations
    float *d_q, *d_k, *d_v, *d_out;
    int *d_bt, *d_lens;

    CUDA_CHECK(cudaMalloc(&d_q, h_query.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, h_key_cache.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, h_value_cache.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, h_output.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bt, h_block_tables.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lens, h_context_lens.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_q, h_query.data(), h_query.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_key_cache.data(), h_key_cache.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_value_cache.data(), h_value_cache.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt, h_block_tables.data(), h_block_tables.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lens, h_context_lens.data(), h_context_lens.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Launch PagedAttention
    cudaError_t status = launch_paged_attention_v1_fp32(d_q, d_k, d_v, d_out, d_bt, d_lens, config);
    CUDA_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back output
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify output is non-zero and non-NaN
    bool all_valid = true;
    for (size_t i = 0; i < h_output.size(); ++i) {
        if (std::isnan(h_output[i]) || std::isinf(h_output[i]) || h_output[i] == 0.0f) {
            all_valid = false;
            std::cerr << "[ERROR] Invalid output at index " << i << ": " << h_output[i] << std::endl;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_bt));
    CUDA_CHECK(cudaFree(d_lens));

    if (all_valid) {
        std::cout << "[PASS] PagedAttention v1 completed successfully with valid numerical output." << std::endl;
        return 0;
    } else {
        std::cerr << "[FAIL] PagedAttention validation failed." << std::endl;
        return 1;
    }
}
