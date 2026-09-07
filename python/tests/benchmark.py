#!/usr/bin/env python3
"""
Python Microbenchmark Suite
FlashKernel-Engine | Micro1 CUDA Engineering Validation
"""

import time
import math
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flash_engine.ops import flash_attention, fused_rmsnorm, fused_swiglu

def benchmark_attention(B=2, H=8, N=1024, d=64, warmup=5, iters=20):
    print(f"\n--- Benchmarking Attention: B={B}, H={H}, N={N}, d={d} ---")
    Q = np.random.randn(B, H, N, d).astype(np.float32) * 0.1
    K = np.random.randn(B, H, N, d).astype(np.float32) * 0.1
    V = np.random.randn(B, H, N, d).astype(np.float32) * 0.1
    
    # Warmup
    for _ in range(warmup):
        _ = flash_attention(Q, K, V)
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = flash_attention(Q, K, V)
    elapsed = (time.perf_counter() - start) / iters
    
    flops = 4.0 * B * H * (N ** 2) * d
    tflops = (flops / elapsed) / 1e12
    
    print(f"  Execution Time: {elapsed * 1000.0:.3f} ms")
    print(f"  Compute Throughput: {tflops:.4f} TFLOPs")
    return elapsed

def benchmark_rmsnorm(tokens=4096, hidden_dim=4096, warmup=5, iters=50):
    print(f"\n--- Benchmarking RMSNorm: Tokens={tokens}, Dim={hidden_dim} ---")
    x = np.random.randn(tokens, hidden_dim).astype(np.float32)
    gamma = np.ones((hidden_dim,), dtype=np.float32)
    
    for _ in range(warmup):
        _ = fused_rmsnorm(x, gamma)
        
    start = time.perf_counter()
    for _ in range(iters):
        _ = fused_rmsnorm(x, gamma)
    elapsed = (time.perf_counter() - start) / iters
    
    total_bytes = (2 * tokens * hidden_dim + hidden_dim) * 4
    gbps = (total_bytes / elapsed) / 1e9
    
    print(f"  Execution Time: {elapsed * 1000.0:.3f} ms")
    print(f"  Memory Bandwidth: {gbps:.2f} GB/s")
    return elapsed

if __name__ == "__main__":
    print("=" * 60)
    print(" FLASHKERNEL-ENGINE PYTHON BENCHMARK RUNNER")
    print("=" * 60)
    benchmark_attention(B=1, H=8, N=512, d=64)
    benchmark_attention(B=1, H=8, N=1024, d=64)
    benchmark_rmsnorm(tokens=2048, hidden_dim=2048)
    benchmark_rmsnorm(tokens=4096, hidden_dim=4096)
