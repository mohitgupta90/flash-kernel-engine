#pragma once

#include "cuda_utils.cuh"
#include "flash_attention.cuh"
#include "paged_attention.cuh"
#include "flash_decoding.cuh"
#include "quantized_gemm.cuh"
#include "fused_cross_entropy.cuh"
#include "fused_rmsnorm.cuh"
#include "fused_activations.cuh"
#include "fused_rope.cuh"
#include "tensor_core_gemm.cuh"
#include "cuda_graph_runner.h"
#include "cuda_memory_pool.h"

namespace flash_engine {

// Version information
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

inline const char* get_version() {
    return "1.1.0";
}

// Print GPU Hardware properties (SMs, Warp size, Max shared memory, Compute capability)
void print_device_info(int device_id = 0);

} // namespace flash_engine
