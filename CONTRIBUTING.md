# Contributing to FlashKernel-Engine

Thank you for your interest in contributing to **FlashKernel-Engine**! We welcome contributions from systems engineers, AI kernel developers, and researchers aiming to advance the state of the art in high-performance GPU compute.

---

## 1. Development Principles & Coding Standards

All CUDA and C++ code within this repository must adhere to production-grade system performance criteria:

1. **Memory Hierarchy & Coalescing**:
   - Maximize global memory transaction efficiency through 128-bit vectorized loads (`float4`, `int4`, `half8`).
   - Eliminate shared memory bank conflicts through proper padding (`HEAD_DIM + 1` or 8-byte alignment offsets).
2. **Device-Side Portability**:
   - Device kernels must never call host runtime routines or exceptions (`std::cout`, `<stdexcept>`, `std::min`/`std::max`). Use CUDA device intrinsics (`fminf`, `fmaxf`, `__shfl_down_sync`).
   - Guard architecture-specific intrinsics (e.g. `__dp4a`, PTX `wmma`) behind `__CUDA_ARCH__` feature macros with suitable software fallbacks.
3. **C++ & CUDA Standards**:
   - C++17 standard is strictly enforced.
   - Code must compile cleanly across all target architectures: Volta (`sm_70`), Turing (`sm_75`), Ampere (`sm_80`), Ada (`sm_89`), and Hopper (`sm_90`).
   - Scoped CMake flags: C++ host compiler flags (`-Wall -Wextra -O3`) must be protected via `$<COMPILE_LANGUAGE:CXX>` generator expressions to prevent NVCC syntax rejection.
4. **Precision Guarantees**:
   - All FP16/BF16 kernels must maintain intermediate accumulations in IEEE-754 FP32 to prevent catastrophic cancellation during online exponentiation.

---

## 2. Setting Up Local Environment

### Prerequisites
- Linux (Ubuntu 20.04/22.04) or Windows 10/11 with WSL2
- NVIDIA Driver >= 525.60.13
- CUDA Toolkit >= 12.0 (NVCC)
- CMake >= 3.18
- Python >= 3.10 with PyTorch >= 2.1 (optional for Python bindings)

### Building the Library and Benchmarks
```bash
# Clone the repository with submodules
git clone https://github.com/mohitgupta90/flash-kernel-engine.git
cd flash-kernel-engine

# Configure CMake with Ampere & Ada architecture targets
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;89"

# Build all static targets and benchmarks
cmake --build build --config Release -j $(nproc)
```

### Running Unit Tests
```bash
# Run algorithmic test suite
python python/tests/test_correctness.py

# Run PagedAttention virtual block validation (requires CUDA GPU)
./build/test_paged_attention

# Run Quantized INT8 DP4A validation (requires CUDA GPU)
./build/test_quantized_gemm
```

---

## 3. Pull Request Guidelines

1. **Branch Naming**:
   - `feat/feature-name` for new kernels or memory allocators.
   - `perf/optimization-name` for throughput or latency improvements.
   - `fix/issue-description` for bug fixes.
2. **Benchmarking Requirement**:
   - Any PR proposing performance optimizations must provide before/after profiling data from `ncu` (NVIDIA Nsight Compute) or standalone benchmarks demonstrating throughput improvements (TFLOPs or GB/s).
3. **Commit Messages**:
   - Use conventional commit style: `feat: add Hopper TMA async copy`, `perf: optimize shared memory tile stride in FlashDecoding`.

---

## 4. Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for everyone, regardless of background, gender, or identity. Please maintain professional, constructive, and respectful discussions across all issues and pull requests.
