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
