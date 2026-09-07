#pragma once

#include <cuda_runtime.h>
#include <functional>
#include "cuda_utils.cuh"

namespace flash_engine {
namespace graph {

class CUDAGraphRunner {
public:
    CUDAGraphRunner();
    ~CUDAGraphRunner();

    // Disable copy
    CUDAGraphRunner(const CUDAGraphRunner&) = delete;
    CUDAGraphRunner& operator=(const CUDAGraphRunner&) = delete;

    // Capture an arbitrary sequence of kernel launches on a CUDA stream
    void capture(std::function<void(cudaStream_t)> record_func, cudaStream_t stream = nullptr);

    // Launch the instantiated CUDA Graph with near-zero CPU launch latency
    void launch(cudaStream_t stream = nullptr);

    // Check if graph is already instantiated
    bool is_captured() const { return instantiated_; }

    // Reset and free resources
    void reset();

private:
    cudaGraph_t graph_;
    cudaGraphExec_t instance_;
    bool instantiated_;
};

} // namespace graph
} // namespace flash_engine
