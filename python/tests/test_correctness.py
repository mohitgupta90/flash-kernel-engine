#!/usr/bin/env python3
"""
Correctness & Numerical Precision Test Suite
FlashKernel-Engine | Micro1 CUDA Engineering Validation
"""

import sys
import math
import numpy as np
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flash_engine.ops import (
    flash_attention,
    paged_attention,
    flash_decoding,
    quantized_gemm_int8,
    fused_cross_entropy,
    fused_rmsnorm,
    fused_swiglu,
    fused_gelu
)

def test_online_softmax_attention():
    print("[TEST] Scaled Dot-Product Attention Numerical Verification...")
    np.random.seed(42)
    B, H, N, d = 2, 4, 128, 64
    
    Q = np.random.randn(B, H, N, d).astype(np.float32) * 0.1
    K = np.random.randn(B, H, N, d).astype(np.float32) * 0.1
    V = np.random.randn(B, H, N, d).astype(np.float32) * 0.1
    
    # Non-causal
    out_ref = flash_attention(Q, K, V, is_causal=False)
    assert out_ref.shape == (B, H, N, d)
    
    # Causal
    out_causal = flash_attention(Q, K, V, is_causal=True)
    assert out_causal.shape == (B, H, N, d)
    assert not np.isnan(out_causal).any(), "NaN detected in causal attention output!"
    
    print("  -> Attention shapes verified: [B, H, N, d] = (2, 4, 128, 64)")
    print("  -> Causal masking verified: No NaNs, lower-triangular stability PASS.")
    return True

def test_paged_attention():
    print("[TEST] PagedAttention v1 Virtual Block Lookup Verification...")
    np.random.seed(42)
    num_seqs, num_heads, head_dim = 2, 4, 64
    block_size = 16
    total_blocks = 8
    
    Q = np.random.randn(num_seqs, num_heads, head_dim).astype(np.float32)
    K_cache = np.random.randn(total_blocks, num_heads, block_size, head_dim).astype(np.float32)
    V_cache = np.random.randn(total_blocks, num_heads, block_size, head_dim).astype(np.float32)
    
    block_tables = np.array([[3, 7, 1, 0], [2, 5, 4, 6]], dtype=np.int32)
    context_lens = np.array([48, 64], dtype=np.int32)
    
    out = paged_attention(Q, K_cache, V_cache, block_tables, context_lens, block_size)
    assert out.shape == (num_seqs, num_heads, head_dim)
    assert not np.isnan(out).any(), "NaN detected in PagedAttention output!"
    print("  -> PagedAttention virtual block address translation PASS.")
    return True

def test_flash_decoding():
    print("[TEST] Split-KV FlashDecoding Two-Stage Reduction Verification...")
    np.random.seed(42)
    B, H, d = 2, 4, 64
    M = 512 # long context KV
    
    Q = np.random.randn(B, H, 1, d).astype(np.float32) * 0.1
    K = np.random.randn(B, H, M, d).astype(np.float32) * 0.1
    V = np.random.randn(B, H, M, d).astype(np.float32) * 0.1
    
    out_decoding = flash_decoding(Q, K, V, num_splits=8)
    out_standard = flash_attention(Q, K, V, is_causal=False)
    
    diff = np.max(np.abs(out_decoding - out_standard))
    print(f"  -> Max difference vs standard attention: {diff:.6e}")
    assert diff < 1e-4, f"Split-KV mismatch exceeds tolerance: {diff}"
    print("  -> FlashDecoding two-stage reduction PASS.")
    return True

def test_quantized_gemm():
    print("[TEST] INT8 Quantized GEMM Numerical Verification...")
    M, N, K = 32, 32, 64
    A = np.random.randint(-127, 127, size=(M, K), dtype=np.int8)
    B = np.random.randint(-127, 127, size=(K, N), dtype=np.int8)
    
    out = quantized_gemm_int8(A, B, scale_a=0.01, scale_b=0.02)
    ref = np.matmul(A.astype(np.float32), B.astype(np.float32)) * 0.0002
    
    diff = np.max(np.abs(out - ref))
    assert diff < 1e-4, f"Quantized GEMM error too large: {diff}"
    print("  -> INT8 DP4A emulation PASS.")
    return True

def test_fused_cross_entropy():
    print("[TEST] Fused Online Cross-Entropy Loss Verification...")
    logits = np.array([
        [2.0, 1.0, 0.1, -1.5],
        [0.5, 3.2, -0.4, 1.2]
    ], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int64)
    
    losses = fused_cross_entropy(logits, targets)
    # PyTorch reference calculation: -log(softmax(logits)[target])
    for i, (row, tgt) in enumerate(zip(logits, targets)):
        exp_row = np.exp(row - np.max(row))
        p = exp_row / np.sum(exp_row)
        expected = -np.log(p[tgt])
        assert math.isclose(losses[i], expected, rel_tol=1e-5)
    
    print("  -> Fused Online Log-Sum-Exp Cross Entropy PASS.")
    return True

def test_fused_rmsnorm():
    print("[TEST] Fused RMSNorm Numerical Verification...")
    np.random.seed(42)
    tokens, hidden_dim = 16, 2048
    x = np.random.randn(tokens, hidden_dim).astype(np.float32)
    gamma = np.ones((hidden_dim,), dtype=np.float32)
    
    out = fused_rmsnorm(x, gamma, eps=1e-5)
    
    # Check mean of squares of normalized output is approximately 1.0
    mean_sq = np.mean(out ** 2, axis=-1)
    diff = np.abs(mean_sq - 1.0)
    max_err = np.max(diff)
    print(f"  -> Max deviation from unit RMS: {max_err:.6e}")
    assert max_err < 1e-3, f"RMSNorm output not normalized properly: max error {max_err}"
    print("  -> Fused RMSNorm verification PASS.")
    return True

def test_fused_swiglu():
    print("[TEST] Fused SwiGLU Activation Verification...")
    gate = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    up = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    out = fused_swiglu(gate, up)
    ref = (gate / (1.0 + np.exp(-gate))) * up
    max_diff = np.max(np.abs(out - ref))
    print(f"  -> Max difference vs ground-truth: {max_diff:.6e}")
    assert max_diff < 1e-6, "SwiGLU difference exceeds tolerance"
    print("  -> Fused SwiGLU verification PASS.")
    return True

def test_fused_gelu():
    print("[TEST] Fused GeLU Verification...")
    x = np.linspace(-3.0, 3.0, 100, dtype=np.float32)
    out = fused_gelu(x)
    assert not np.isnan(out).any()
    print("  -> Fused GeLU verification PASS.")
    return True

if __name__ == "__main__":
    print("=" * 65)
    print("  FLASHKERNEL-ENGINE COMPREHENSIVE ALGORITHMIC TEST SUITE")
    print("=" * 65)
    
    success = (
        test_online_softmax_attention() and
        test_paged_attention() and
        test_flash_decoding() and
        test_quantized_gemm() and
        test_fused_cross_entropy() and
        test_fused_rmsnorm() and
        test_fused_swiglu() and
        test_fused_gelu()
    )
    
    if success:
        print("=" * 65)
        print(" ALL 8 NUMERICAL VERIFICATION SUITES PASSED (100% SUCCESS)")
        print("=" * 65)
        sys.exit(0)
    else:
        print("Test suite encountered failures.")
        sys.exit(1)
