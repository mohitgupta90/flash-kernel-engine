"""
High-Level Python Operations for FlashKernel-Engine
"""

import math
import numpy as np

# Try importing the compiled C++/CUDA extension module
try:
    import flash_engine_cuda
    _HAS_CUDA_EXT = True
except ImportError:
    _HAS_CUDA_EXT = False

def flash_attention(q, k, v, is_causal: bool = False, scale: float = None):
    """
    Computes Scaled Dot-Product Attention using tiled FlashAttention-2 online softmax algorithm.
    
    Args:
        q: Query tensor of shape [Batch, Heads, Seq_Q, Head_Dim]
        k: Key tensor of shape [Batch, Heads, Seq_KV, Head_Dim]
        v: Value tensor of shape [Batch, Heads, Seq_KV, Head_Dim]
        is_causal: Whether to apply lower-triangular causal autoregressive mask
        scale: Softmax scaling factor (defaults to 1.0 / sqrt(Head_Dim))
    
    Returns:
        Output tensor of shape [Batch, Heads, Seq_Q, Head_Dim]
    """
    if _HAS_CUDA_EXT and hasattr(q, "is_cuda") and q.is_cuda:
        return flash_engine_cuda.flash_attention_forward(q, k, v, is_causal)
    
    # High-precision Reference implementation (PyTorch or NumPy)
    if hasattr(q, "shape"):
        B, H, N, d = q.shape
        M = k.shape[2]
        if scale is None:
            scale = 1.0 / math.sqrt(d)
        
        # Reference execution using Torch or NumPy
        if type(q).__module__.startswith("torch"):
            import torch
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if is_causal:
                mask = torch.triu(torch.full((N, M), float("-inf"), device=q.device), diagonal=1)
                scores = scores + mask
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)
        else:
            scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
            if is_causal:
                mask = np.triu(np.full((N, M), -1e30), k=1)
                scores = scores + mask
            # Numerically stable softmax
            m = np.max(scores, axis=-1, keepdims=True)
            e = np.exp(scores - m)
            probs = e / np.sum(e, axis=-1, keepdims=True)
            return np.matmul(probs, v)

def paged_attention(query, key_cache, value_cache, block_tables, context_lens, block_size: int = 16, scale: float = None):
    """
    PagedAttention forward lookup over virtual block tables.
    
    Args:
        query: [num_seqs, num_heads, head_dim]
        key_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables: [num_seqs, max_blocks_per_seq] int32 array
        context_lens: [num_seqs] int32 array
    """
    num_seqs, num_heads, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    output = np.zeros_like(query)
    for b in range(num_seqs):
        ctx_len = context_lens[b]
        if ctx_len <= 0:
            continue
        
        # Assemble sequence KV from physical blocks
        num_blocks = (ctx_len + block_size - 1) // block_size
        k_chunks, v_chunks = [], []
        for blk_idx in range(num_blocks):
            p_blk = block_tables[b, blk_idx]
            k_chunks.append(key_cache[p_blk])  # [num_kv_heads, block_size, head_dim]
            v_chunks.append(value_cache[p_blk])
        
        # Concatenate tokens along block_size dimension
        k_seq = np.concatenate(k_chunks, axis=1)[:, :ctx_len, :]  # [num_kv_heads, ctx_len, head_dim]
        v_seq = np.concatenate(v_chunks, axis=1)[:, :ctx_len, :]
        
        # Query: [num_heads, head_dim] -> [num_heads, 1, head_dim]
        q_b = query[b][:, np.newaxis, :]
        scores = np.matmul(q_b, np.swapaxes(k_seq, -2, -1)) * scale  # [num_heads, 1, ctx_len]
        m = np.max(scores, axis=-1, keepdims=True)
        p = np.exp(scores - m)
        probs = p / np.sum(p, axis=-1, keepdims=True)
        out_b = np.matmul(probs, v_seq)  # [num_heads, 1, head_dim]
        output[b] = out_b.squeeze(1)
        
    return output

def flash_decoding(query, key, value, num_splits: int = 4, scale: float = None):
    """
    Two-stage Split-KV FlashDecoding reference execution.
    """
    B, H, _, d = query.shape
    M = key.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    
    split_size = (M + num_splits - 1) // num_splits
    partial_outputs = []
    partial_maxs = []
    partial_sums = []
    
    for s in range(num_splits):
        start = s * split_size
        end = min(start + split_size, M)
        if start >= M:
            continue
        k_part = key[:, :, start:end, :]
        v_part = value[:, :, start:end, :]
        scores = np.matmul(query, np.swapaxes(k_part, -2, -1)) * scale
        m_s = np.max(scores, axis=-1, keepdims=True)
        p_s = np.exp(scores - m_s)
        l_s = np.sum(p_s, axis=-1, keepdims=True)
        out_s = np.matmul(p_s, v_part)  # unnormalized sum
        partial_outputs.append(out_s)
        partial_maxs.append(m_s)
        partial_sums.append(l_s)
    
    # Stage 2: Reduction across splits
    global_m = np.maximum.reduce(partial_maxs)
    global_sum = np.zeros_like(partial_sums[0])
    global_out = np.zeros_like(partial_outputs[0])
    
    for out_s, m_s, l_s in zip(partial_outputs, partial_maxs, partial_sums):
        rescale = np.exp(m_s - global_m)
        global_sum += l_s * rescale
        global_out += out_s * rescale
        
    return global_out / global_sum

def quantized_gemm_int8(A, B, scale_a: float = 1.0, scale_b: float = 1.0):
    """
    Simulated INT8 DP4A Matrix Multiplication with dequantization:
    C = (A_int8 @ B_int8) * (scale_a * scale_b)
    """
    int_acc = np.matmul(A.astype(np.int32), B.astype(np.int32))
    return int_acc.astype(np.float32) * (scale_a * scale_b)

def fused_cross_entropy(logits, targets, ignore_index: int = -100):
    """
    Online Log-Sum-Exp Cross Entropy Loss.
    """
    losses = []
    for row, tgt in zip(logits, targets):
        if tgt == ignore_index:
            losses.append(0.0)
            continue
        m = np.max(row)
        lse = np.log(np.sum(np.exp(row - m))) + m
        losses.append(float(lse - row[tgt]))
    return np.array(losses, dtype=np.float32)

def fused_rmsnorm(x, gamma, eps: float = 1e-5):
    """
    Computes Root Mean Square Normalization with single-pass warp reduction.
    y = (x / sqrt(mean(x^2) + eps)) * gamma
    """
    if _HAS_CUDA_EXT and hasattr(x, "is_cuda") and x.is_cuda:
        return flash_engine_cuda.fused_rmsnorm_forward(x, gamma, eps)
    
    if type(x).__module__.startswith("torch"):
        import torch
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * gamma
    else:
        var = np.mean(x ** 2, axis=-1, keepdims=True)
        return (x / np.sqrt(var + eps)) * gamma

def fused_swiglu(gate, up):
    """
    Computes Fused SwiGLU activation: (gate * sigmoid(gate)) * up
    """
    if _HAS_CUDA_EXT and hasattr(gate, "is_cuda") and gate.is_cuda:
        return flash_engine_cuda.fused_swiglu_forward(gate, up)
    
    if type(gate).__module__.startswith("torch"):
        import torch
        return (gate * torch.sigmoid(gate)) * up
    else:
        sig = 1.0 / (1.0 + np.exp(-gate))
        return (gate * sig) * up

def fused_gelu(x):
    """
    Computes Fused GeLU with fast tanh approximation:
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    c = 0.7978845608028654
    if type(x).__module__.startswith("torch"):
        import torch
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * (x ** 3))))
