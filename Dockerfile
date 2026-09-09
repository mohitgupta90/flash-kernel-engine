# =============================================================================
# FlashKernel-Engine Multi-Stage Production & Benchmark Container
# Target: NVIDIA Ampere / Ada Lovelace / Hopper Architectures (sm_80, sm_89, sm_90)
# =============================================================================

# --- Stage 1: Build & Compilation ---
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install build dependencies, modern CMake, and Ninja
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/flash-kernel-engine

# Copy project manifests and source code
COPY CMakeLists.txt setup.py pyproject.toml ./
COPY include/ ./include/
COPY src/ ./src/
COPY benchmarks/ ./benchmarks/
COPY tests/ ./tests/
COPY python/ ./python/

# Build C++/CUDA static engine and standalone benchmark binaries
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
    && cmake --build build --target all -j $(nproc)

# Build Python C++ PyTorch Extension Wheel
RUN pip3 install --upgrade pip setuptools wheel numpy \
    && pip3 wheel --no-deps -w /workspace/wheels .

# --- Stage 2: Minimal High-Performance Runtime & Execution ---
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled binaries from builder stage
COPY --from=builder /workspace/flash-kernel-engine/build/bench_attention /usr/local/bin/bench_attention
COPY --from=builder /workspace/flash-kernel-engine/build/bench_rmsnorm /usr/local/bin/bench_rmsnorm
COPY --from=builder /workspace/flash-kernel-engine/build/bench_decoding /usr/local/bin/bench_decoding
COPY --from=builder /workspace/flash-kernel-engine/build/test_paged_attention /usr/local/bin/test_paged_attention
COPY --from=builder /workspace/flash-kernel-engine/build/test_quantized_gemm /usr/local/bin/test_quantized_gemm
COPY --from=builder /workspace/wheels /tmp/wheels

# Install pre-built python wheel
RUN pip3 install --no-cache-dir /tmp/wheels/*.whl numpy \
    && rm -rf /tmp/wheels

# Environment defaults
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bench_attention"]
