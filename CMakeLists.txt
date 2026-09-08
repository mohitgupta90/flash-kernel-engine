cmake_minimum_required(VERSION 3.18)

project(FlashKernelEngine
    VERSION 1.0.0
    DESCRIPTION "High-Performance CUDA & WebGPU Kernel Optimization Engine for Transformers"
    LANGUAGES CXX CUDA
)

# -----------------------------------------------------------------------------
# C++ and CUDA Standards
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# Target NVIDIA GPU Architectures (Volta, Turing, Ampere, Ada, Hopper)
# sm_70: V100
# sm_75: T4, RTX 2080
# sm_80: A100
# sm_86: RTX 3090
# sm_89: RTX 4090, L40
# sm_90: H100
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

message(STATUS "FlashKernel-Engine configured with CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# -----------------------------------------------------------------------------
# Compiler Flags
# -----------------------------------------------------------------------------
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xptxas -v --expt-relaxed-constexpr")

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/O2>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# -----------------------------------------------------------------------------
# Include Directories
# -----------------------------------------------------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -----------------------------------------------------------------------------
# Core Library Target
# -----------------------------------------------------------------------------
set(ENGINE_SOURCES
    src/flash_attention.cu
    src/tensor_core_attention.cu
    src/fused_rmsnorm.cu
    src/fused_activations.cu
    src/fused_rope.cu
    src/cuda_graph_runner.cu
    src/flash_engine_api.cu
)

set(ENGINE_HEADERS
    include/flash_engine.h
    include/cuda_utils.cuh
    include/flash_attention.cuh
    include/tensor_core_gemm.cuh
    include/fused_rmsnorm.cuh
    include/fused_activations.cuh
    include/fused_rope.cuh
    include/cuda_graph_runner.h
)

add_library(flash_kernel_engine STATIC
    ${ENGINE_SOURCES}
    ${ENGINE_HEADERS}
)

target_include_directories(flash_kernel_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# -----------------------------------------------------------------------------
# Standalone Benchmark Executables
# -----------------------------------------------------------------------------
add_executable(bench_attention benchmarks/bench_attention.cu)
target_link_libraries(bench_attention PRIVATE flash_kernel_engine)

add_executable(bench_rmsnorm benchmarks/bench_rmsnorm.cu)
target_link_libraries(bench_rmsnorm PRIVATE flash_kernel_engine)

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS flash_kernel_engine bench_attention bench_rmsnorm
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
