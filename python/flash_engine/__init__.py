"""
FlashKernel-Engine: High-Performance GPU Kernel Acceleration Suite
"""

from .ops import (
    flash_attention,
    fused_rmsnorm,
    fused_swiglu,
    fused_gelu
)

__version__ = "1.0.0"
__author__ = "Mohit Gupta"

__all__ = [
    "flash_attention",
    "fused_rmsnorm",
    "fused_swiglu",
    "fused_gelu",
]
