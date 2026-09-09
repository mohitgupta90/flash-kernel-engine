#include "cuda_memory_pool.h"
#include <iostream>
#include <algorithm>

namespace flash_engine {
namespace memory {

CudaMemoryPool::CudaMemoryPool(int device_id, size_t release_threshold_bytes)
    : device_id_(device_id),
      supports_async_alloc_(false),
      mem_pool_(nullptr),
      currently_allocated_bytes_(0),
      peak_allocated_bytes_(0),
      total_allocations_(0),
      total_deallocations_(0) {
    
    CUDA_CHECK(cudaSetDevice(device_id_));

    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);

    int supported = 0;
    cudaError_t err = cudaDeviceGetAttribute(&supported, cudaDeviceAttributeMemoryPoolsSupported, device_id_);
    if (err == cudaSuccess && supported == 1) {
        supports_async_alloc_ = true;
        CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mem_pool_, device_id_));

        // Configure pool threshold to prevent excessive OS-level virtual memory mapping overhead
        uint64_t threshold = static_cast<uint64_t>(release_threshold_bytes);
        CUDA_CHECK(cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &threshold));

        // Allow opportunistic reuse across stream event dependencies
        int enable_reuse = 1;
        CUDA_CHECK(cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolReuseFollowEventDependencies, &enable_reuse));
    } else {
        supports_async_alloc_ = false;
    }
}

CudaMemoryPool::~CudaMemoryPool() {
    // Release threshold reset if pool was acquired
    if (supports_async_alloc_ && mem_pool_ != nullptr) {
        uint64_t zero_threshold = 0;
        cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &zero_threshold);
        cudaMemPoolTrimTo(mem_pool_, 0);
    }
}

void* CudaMemoryPool::allocate_async(size_t bytes, cudaStream_t stream) {
    if (bytes == 0) return nullptr;

    void* ptr = nullptr;
    if (supports_async_alloc_) {
        CUDA_CHECK(cudaMallocAsync(&ptr, bytes, stream));
    } else {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }

    currently_allocated_bytes_ += bytes;
    total_allocations_++;
    if (currently_allocated_bytes_ > peak_allocated_bytes_) {
        peak_allocated_bytes_ = currently_allocated_bytes_;
    }

    return ptr;
}

void CudaMemoryPool::free_async(void* ptr, cudaStream_t stream) {
    if (ptr == nullptr) return;

    if (supports_async_alloc_) {
        CUDA_CHECK(cudaFreeAsync(ptr, stream));
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }

    total_deallocations_++;
}

void* CudaMemoryPool::allocate(size_t bytes) {
    return allocate_async(bytes, nullptr);
}

void CudaMemoryPool::free(void* ptr) {
    free_async(ptr, nullptr);
}

void CudaMemoryPool::trim_to(size_t bytes_to_keep) {
    if (supports_async_alloc_ && mem_pool_ != nullptr) {
        CUDA_CHECK(cudaMemPoolTrimTo(mem_pool_, bytes_to_keep));
    }
}

MemoryPoolStats CudaMemoryPool::get_stats() const {
    MemoryPoolStats stats;
    stats.currently_allocated_bytes = currently_allocated_bytes_;
    stats.peak_allocated_bytes = peak_allocated_bytes_;
    stats.total_allocations = total_allocations_;
    stats.total_deallocations = total_deallocations_;
    return stats;
}

void CudaMemoryPool::reset_peak_stats() {
    peak_allocated_bytes_ = currently_allocated_bytes_;
}

} // namespace memory
} // namespace flash_engine
