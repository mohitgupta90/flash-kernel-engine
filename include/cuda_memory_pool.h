#pragma once

#include "cuda_utils.cuh"
#include <cstddef>
#include <cstdint>
#include <string>

namespace flash_engine {
namespace memory {

struct MemoryPoolStats {
    size_t currently_allocated_bytes;
    size_t peak_allocated_bytes;
    size_t total_allocations;
    size_t total_deallocations;
};

// High-Performance Stream-Ordered Asynchronous CUDA Memory Pool
// Eliminates kernel launch stalls caused by synchronous cudaMalloc/cudaFree calls.
class CudaMemoryPool {
public:
    explicit CudaMemoryPool(int device_id = 0, size_t release_threshold_bytes = 1024 * 1024 * 512); // Default 512 MB
    ~CudaMemoryPool();

    // Disable copy
    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    // Allocate memory asynchronously on a given CUDA stream
    void* allocate_async(size_t bytes, cudaStream_t stream = nullptr);

    // Free memory asynchronously on a given CUDA stream
    void free_async(void* ptr, cudaStream_t stream = nullptr);

    // Synchronous allocation (uses default stream)
    void* allocate(size_t bytes);

    // Synchronous deallocation
    void free(void* ptr);

    // Trim pooled memory back to the OS / GPU driver
    void trim_to(size_t bytes_to_keep = 0);

    // Query pool runtime statistics
    MemoryPoolStats get_stats() const;

    // Reset peak tracking metrics
    void reset_peak_stats();

    // Check if CUDA Stream-Ordered Memory is supported on current device
    bool is_stream_ordered_supported() const { return supports_async_alloc_; }

private:
    int device_id_;
    bool supports_async_alloc_;
    cudaMemPool_t mem_pool_;
    size_t currently_allocated_bytes_;
    size_t peak_allocated_bytes_;
    size_t total_allocations_;
    size_t total_deallocations_;
};

} // namespace memory
} // namespace flash_engine
