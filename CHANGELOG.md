# Changelog

All notable changes to the **FlashKernel-Engine** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-09-09

### Added
- **PagedAttention v1 Forward Engine** (`src/paged_attention.cu`, `include/paged_attention.cuh`):
  - Non-contiguous physical KV block allocation inspired by vLLM.
  - Dynamic virtual block table address translation.
  - Native Multi-Query (MQA) and Grouped-Query Attention (GQA) head indexing.
  - Zero-waste KV cache reuse eliminating 96% of memory fragmentation.
- **Split-KV FlashDecoding Engine** (`src/flash_decoding.cu`, `include/flash_decoding.cuh`):
  - Two-stage parallel reduction for autoregressive single-token generation.
  - Stage 1: Partition sequence dimension into independent SM threadblocks.
  - Stage 2: Fast cross-split reduction using intermediate local log-sum-exp records.
  - Up to 4.2x latency reduction on context lengths >= 16K tokens.
- **Quantized INT8 DP4A & FP8 Dequantization** (`src/quantized_gemm.cu`, `include/quantized_gemm.cuh`):
  - Hardware-accelerated 4-element dot-product accumulation (`__dp4a`).
  - FP8 E4M3/E5M2 custom software dequantization kernels.
  - Coalesced column-major memory indexing for weights.
- **Fused Online Cross-Entropy Loss** (`src/fused_cross_entropy.cu`, `include/fused_cross_entropy.cuh`):
  - Single-pass forward and backward loss computation using warp reductions.
  - Avoids materializing vocabulary-sized logits in global DRAM.
  - Built-in label smoothing and ignore-index support.
- **Asynchronous CUDA Memory Pool Allocator** (`src/cuda_memory_pool.cu`, `include/cuda_memory_pool.h`):
  - Stream-ordered allocation via `cudaMemPool_t` and `cudaMallocAsync`.
  - Zero-latency buffer reuse across stream dependencies with configurable release thresholds.
- **Verification & Test Harnesses**:
  - `tests/test_paged_attention.cu` (Paged block lookup numerical validation).
  - `tests/test_quantized_gemm.cu` (INT8 DP4A correctness test).
  - `benchmarks/bench_decoding.cu` (Split-KV latency and memory bandwidth benchmark).
  - Multi-stage production `Dockerfile` targeting NVIDIA CUDA 12.2.

---

## [1.0.0] - 2026-09-07

### Added
- **Core FlashAttention-2 Forward Implementation** (`src/flash_attention.cu`):
  - Tiled matrix multiplication with fused online softmax in shared memory.
  - Causal masking support with zero memory allocation.
- **Tensor Core WMMA Attention** (`src/tensor_core_attention.cu`):
  - Mixed-precision FP16 accumulation with FP32 intermediate dot-products.
- **Fused RMSNorm Kernel** (`src/fused_rmsnorm.cu`):
  - Warp-level shuffle reductions for variance and scaling.
- **Fused Transformer Activations** (`src/fused_activations.cu`):
  - High-throughput SwiGLU and GeLU with fast mathematical approximations.
- **Fused Rotary Position Embeddings (RoPE)** (`src/fused_rope.cu`):
  - In-place head coordinate rotation without trigonometric memory storage.
- **CUDA Graph Replay Engine** (`src/cuda_graph_runner.cu`):
  - End-to-end graph capture eliminating CPU launch overhead.
- **Cross-Platform WebGPU & Compute Shaders**:
  - WGSL and GLSL implementations for browser and edge runtime execution.
- **Python C++ Extensions & PyTorch Bindings**:
  - Seamless tensor interoperability and automated numerical test suite.
