"""Independent PyTorch reference for per-token FP8 wire quantization.

These functions are for validation and analysis. The communication runtime
quantizes in CUDA, and its external payload tensors remain BF16.
"""

from __future__ import annotations

from typing import Tuple

import torch


def quantize_fp8_reference(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize finite input to E4M3FN bytes [T,H] and FP32 scales [T].

    A nonzero row uses max(maxabs/448, FP32 normal minimum), keeping its
    reciprocal finite even for BF16 subnormals. An all-zero row uses one.
    NaN and infinity inputs are outside this wire-format contract.
    """
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("FP8 reference expects a floating [T,H] tensor")
    values = x.float()
    maximum = values.abs().amax(dim=1)
    nonzero_scales = (maximum / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    scales = torch.where(maximum == 0, torch.ones_like(maximum), nonzero_scales)
    # Match CUDA's multiplication by the FP32 reciprocal before RNE conversion.
    normalized = (values * scales.reciprocal().unsqueeze(1)).clamp(-448.0, 448.0)
    codes = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
    return codes, scales


def dequantize_fp8_reference(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode wire bytes using the token scale and round back to BF16."""
    if codes.dtype != torch.uint8 or codes.ndim != 2:
        raise ValueError("FP8 codes must be a uint8 [T,H] tensor")
    if scales.shape != codes.shape[:1] or scales.dtype != torch.float32:
        raise ValueError("FP8 scales must be FP32 [T]")
    return (codes.view(torch.float8_e4m3fn).float() * scales.unsqueeze(1)).to(torch.bfloat16)


def fp8_roundtrip_reference(x: torch.Tensor) -> torch.Tensor:
    """Expected remote BF16 payload; local bypass retains the original x."""
    return dequantize_fp8_reference(*quantize_fp8_reference(x))
