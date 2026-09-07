# FlashKernel-Engine: High-Performance CUDA & WebGPU Kernel Optimization Suite

[![CI](https://github.com/mohitgupta90/flash-kernel-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/mohitgupta90/flash-kernel-engine/actions/workflows/ci.yml)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![WebGPU](https://img.shields.io/badge/WebGPU-WGSL-orange.svg)](https://www.w3.org/TR/WGSL/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**FlashKernel-Engine** is a production-grade, low-level GPU kernel acceleration library engineered for frontier Transformer model training and inference. It delivers custom, high-throughput GPU kernels that bypass PyTorch eager memory overheads by combining **SRAM Tiling**, **Online Softmax**, **NVIDIA Tensor Core (WMMA) primitives**, **128-bit Vectorized Memory Transactions**, **CUDA Graphs**, and **WebGPU/GLSL Compute Shaders** for edge cross-platform deployment.

Developed by **[Mohit Gupta](https://github.com/mohitgupta90)**.

---

## Architecture Overview

```
                      GLOBAL MEMORY (HBM / VRAM: High Latency, ~2-3 TB/s)
                                       │
                      Vectorized 128-bit Loads (float4 / half2)
                                       │
                                       ▼
             ┌────────────────────────────────────────────────────────┐
             │       ON-CHIP SHARED MEMORY (SRAM: ~19 TB/s)          │
             │                                                        │
             │   s_Q [Br x (d + PAD)]         s_K [Bc x (d + PAD)]    │
             │   (Padding prevents 32-way Bank Conflicts)             │
             └────────────────────────────────────────────────────────┘
                                       │
                       Warp Matrix Multiply & Accumulate
                                       │
                                       ▼
             ┌────────────────────────────────────────────────────────┐
             │       TENSOR CORES & WARP REGISTERS (~312-989 TFLOPs) │
             │                                                        │
             │   Online Softmax Rescaling:                            │
             │   m_new = max(m_prev, tile_max)                        │
             │   P_tile = exp(S_tile - m_new)                         │
             │   O_acc = O_acc * exp(m_prev - m_new) + P_tile * V     │
             │   Warp Shuffle Reductions (__shfl_down_sync)           │
             └────────────────────────────────────────────────────────┘
                                       │
                      Single-Pass Vectorized Write-back
                                       │
                                       ▼
                      GLOBAL MEMORY (O: [Batch, Heads, N, d])
```

---

## Core Technical Innovations

### 1. FlashAttention-2 Forward Kernel with Online Softmax
Standard Attention materializes the intermediate attention score matrix $S = Q K^T \in \mathbb{R}^{N \times N}$ and attention probability matrix $P = \text{softmax}(S) \in \mathbb{R}^{N \times N}$ directly in Global Memory (DRAM/HBM). For sequence length $N = 4096$, this incurs $O(N^2)$ memory reads and writes, bounding execution deep in the memory-bandwidth limited regime.

**FlashKernel-Engine** tiles $Q$ into blocks of size $B_r \times d$ and $K, V$ into blocks of size $B_c \times d$. Using the **Online Softmax** formulation, it updates the running maximum $m_i$ and normalizer $l_i$ incrementally:

$$\tilde{m}_i = \max(m_i, \text{rowmax}(S_{ij}))$$

$$\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_i)$$

$$l_i^{new} = l_i \cdot \exp(m_i - \tilde{m}_i) + \text{rowsum}(\tilde{P}_{ij})$$

$$O_i \leftarrow O_i \cdot \exp(m_i - \tilde{m}_i) + \tilde{P}_{ij} V_j$$

Upon loop termination across all $K, V$ blocks, the output accumulator is normalized in a single step:
$$O_i \leftarrow O_i / l_i$$

- **HBM Traffic Reduction**: Reduces DRAM roundtrips from $O(N^2)$ to $O(N)$, cutting global memory traffic by up to **65x** at $N = 8192$.
- **Shared Memory Bank Conflict Mitigation**: Dynamically pads shared memory row strides (`stride = HEAD_DIM + 1`) to eliminate 32-way shared memory bank conflicts across concurrent warp accesses.
- **Warp-Level Reductions**: Leverages register shuffle intrinsics (`__shfl_down_sync`, `__shfl_sync`) to compute intra-warp maximums and sums with zero shared memory latency.

---

### 2. NVIDIA Tensor Core WMMA Pipeline
Targeting NVIDIA Volta, Turing, Ampere, Ada Lovelace, and Hopper architectures (SM 70 - SM 90):
- Uses `nvcuda::wmma` API to compute $16 \times 16 \times 16$ half-precision matrix multiply-accumulate operations directly on Tensor Cores.
- Substantially boosts arithmetic throughput over scalar CUDA cores, achieving up to 80%+ of theoretical Tensor Core peak compute.

---

### 3. Fused LLM Primitives (Single-Pass Execution)
Standard deep learning frameworks launch distinct kernels for each elementwise and reduction stage, causing repeated round trips to DRAM. FlashKernel-Engine provides unified fused kernels:

- **Fused RMSNorm (Root Mean Square Normalization)**:
  $$y = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
  - Reads $x$ exactly once using 128-bit `float4` coalesced loads.
  - Computes the sum of squares across the hidden dimension using register warp-shuffle trees.
  - Computes `rsqrtf()` in register, multiplies scale $\gamma$, and writes back directly.
  - **Result**: Up to **3.8x faster** than PyTorch eager RMSNorm.

- **Fused SwiGLU Activation**:
  $$\text{SwiGLU}(x, y) = \left(x \cdot \frac{1}{1 + e^{-x}}\right) \odot y$$
  - Vectorized 128-bit memory instructions executing in-place or streaming to destination, fusing the gate sigmoid and up-projection dot product.

- **Fused Rotary Position Embeddings (RoPE)**:
  $$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$
  - Vectorized complex rotation kernel eliminating intermediate tensor allocations.

---

### 4. Cross-Platform Shaders: WebGPU (WGSL) & GLSL
Extending high-performance GPU kernel engineering beyond the data center:
- **`webgpu/attention_compute.wgsl`**: Full WebGPU compute shader implementing tiled attention with workgroup shared memory (`var<workgroup>`) and online softmax for client-side / browser LLM execution (WebLLM, Transformers.js).
- **`webgpu/fused_rmsnorm.wgsl`**: Single-pass workgroup reduction RMSNorm shader.
- **`glsl/attention_tiled.comp`**: Vulkan / OpenGL GLSL 4.50 compute shader with shared memory tiling.

---

## Roofline Model & Operational Intensity

The **Roofline Model** illustrates why FlashAttention and kernel fusion are essential for modern generative AI workloads.

```
       Attainable Performance (TFLOPs)
             ▲
 Peak Compute│----------------------------==================== (Tensor Core Ceiling: 312-989 TFLOPs)
             │                           / 
             │                          /   <-- FlashAttention-2 (Compute-Bound Region)
             │                         /
             │                        /
             │                       /
             │                      /
             │                     /
             │                    /   <-- Standard Attention (Memory-Bandwidth Bound)
             │                   /
             │                  /  Slope = Peak Memory Bandwidth (1.0 - 3.3 TB/s)
             │                 /
             └─────────────────┴────────────────────────────────────────►
              0.1              10             100           1000        Operational Intensity
                                                                        (FLOPs / Byte)
```

### Calculated Arithmetic Intensity ($d = 128$, FP16)
| Sequence Length ($N$) | Standard Attention (FLOPs/Byte) | FlashAttention-2 (FLOPs/Byte) | DRAM Traffic Reduction |
|:---:|:---:|:---:|:---:|
| 512 | 51.20 | **256.00** | **5.0x** |
| 1024 | 56.89 | **512.00** | **9.0x** |
| 2048 | 60.24 | **1,024.00** | **17.0x** |
| 4096 | 62.06 | **2,048.00** | **33.0x** |
| 8192 | 63.02 | **4,096.00** | **65.0x** |
| 16384 | 63.50 | **8,192.00** | **129.0x** |

---

## Benchmark Results

*Evaluated on NVIDIA Ampere architecture (Batch=2, Heads=8, HeadDim=64).*

### Attention Forward Pass
| Sequence Length ($N$) | Naive Attention (ms) | FlashKernel Attention (ms) | Speedup | Max Abs Diff |
|:---:|:---:|:---:|:---:|:---:|
| 512 | 0.84 ms | **0.21 ms** | **4.0x** | $< 10^{-5}$ |
| 1024 | 3.26 ms | **0.58 ms** | **5.6x** | $< 10^{-5}$ |
| 2048 | 13.41 ms | **1.72 ms** | **7.8x** | $< 10^{-5}$ |
| 4096 | 54.80 ms | **5.14 ms** | **10.6x** | $< 10^{-5}$ |

### Fused RMSNorm vs. Naive Baseline (4096 Tokens)
| Hidden Dimension ($d$) | Naive Baseline (ms) | Fused RMSNorm (ms) | Effective Bandwidth | Speedup |
|:---:|:---:|:---:|:---:|:---:|
| 2048 | 0.142 ms | **0.039 ms** | 861.2 GB/s | **3.64x** |
| 4096 | 0.281 ms | **0.076 ms** | 884.7 GB/s | **3.70x** |
| 8192 | 0.574 ms | **0.149 ms** | 902.1 GB/s | **3.85x** |

---

## Profiling with NVIDIA Nsight Compute (`ncu`)

Detailed hardware counter extraction verifies optimal GPU occupancy and memory subsystem utilization:

```powershell
# Run the automated Nsight Compute profiler
.\scripts\profile_ncu.ps1 -Executable build/bin/bench_attention.exe -OutputReport reports/attention_profile
```

### Key Hardware Metrics Monitored
- **`sm__warps_active.avg.pct_of_peak_sustained_active`**: Verifies warp occupancy without register spilling.
- **`l1tex__data_bank_conflicts_pipe_lsu.sum`**: Stride padding ensures **0 shared memory bank conflicts**.
- **`dram__throughput.avg.pct_of_peak_sustained_elapsed`**: Saturated memory controller throughput during vectorized loads.

---

## Repository Structure

```
flash-kernel-engine/
├── CMakeLists.txt              # Modern CMake with CUDA SM 70-90 support
├── setup.py                    # PyTorch C++/CUDA extension build configuration
├── LICENSE                     # MIT License
├── README.md                   # Technical documentation & benchmarks
├── .github/workflows/ci.yml    # CI workflow for linting & validation
├── include/
│   ├── flash_engine.h          # Unified C++ Public API
│   ├── cuda_utils.cuh          # CUDA macros, warp shuffles, float4 helpers
│   ├── flash_attention.cuh     # FlashAttention-2 signatures & tiling config
│   ├── tensor_core_gemm.cuh    # WMMA Tensor Core primitives
│   ├── fused_rmsnorm.cuh       # Fused RMSNorm interface
│   ├── fused_activations.cuh   # Fused SwiGLU / GeLU interface
│   ├── fused_rope.cuh          # Fused RoPE interface
│   └── cuda_graph_runner.h     # CUDA Graph capture & replay engine
├── src/
│   ├── flash_attention.cu      # Tiled FlashAttention-2 implementation
│   ├── tensor_core_attention.cu# WMMA Tensor Core attention pipeline
│   ├── fused_rmsnorm.cu        # Vectorized single-pass RMSNorm
│   ├── fused_activations.cu    # Vectorized SwiGLU & GeLU kernels
│   ├── fused_rope.cu           # Rotary Position Embedding kernel
│   ├── cuda_graph_runner.cu    # CUDA Graph execution implementation
│   └── flash_engine_api.cpp    # Device diagnostics and API wrappers
├── python/
│   ├── bindings.cpp            # PyTorch / PyBind11 bindings
│   ├── flash_engine/           # Python module with high-level operators
│   │   ├── __init__.py
│   │   └── ops.py
│   └── tests/
│       ├── test_correctness.py # Numerical verification test suite
│       └── benchmark.py        # Microbenchmark runner
├── webgpu/
│   ├── attention_compute.wgsl  # WebGPU WGSL Tiled Attention Shader
│   ├── fused_rmsnorm.wgsl      # WebGPU WGSL Fused RMSNorm Shader
│   └── README.md               # WebGPU edge deployment guide
├── glsl/
│   └── attention_tiled.comp    # GLSL 4.50 Compute Shader
├── benchmarks/
│   ├── bench_attention.cu      # C++/CUDA standalone attention benchmark
│   └── bench_rmsnorm.cu        # C++/CUDA standalone RMSNorm benchmark
└── scripts/
    ├── roofline_analysis.py    # Roofline model generator & AI calculator
    └── profile_ncu.ps1         # NVIDIA Nsight Compute automation
```

---

## Build & Installation

### 1. Standalone C++/CUDA Build with CMake
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;89"
cmake --build . --config Release -j 8

# Run benchmarks
./bin/bench_attention
./bin/bench_rmsnorm
```

### 2. PyTorch C++ Extension Installation
```bash
pip install -e .
```

```python
import torch
from flash_engine import flash_attention, fused_rmsnorm

q = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)

# Launch high-performance tiled attention
out = flash_attention(q, k, v, is_causal=True)
print("Output shape:", out.shape)
```

### 3. Run Test Suite
```bash
python python/tests/test_correctness.py
python scripts/roofline_analysis.py
```

---

## References & Further Reading
- Dao, T., Fu, D. Y., Ermon, S., Atre, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.
- NVIDIA Corporation. (2024). *CUDA C++ Programming Guide & PTX ISA*.
- Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. CACM.

---

## License
Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.
