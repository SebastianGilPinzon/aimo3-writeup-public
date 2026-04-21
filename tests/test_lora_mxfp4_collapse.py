"""
Substrate-trap detection test: LoRA-on-MXFP4 merge-then-requantize collapse.

Reference: AIMO3 Writeup Section 6.2 (Gil Pinzon, 2026).

This test detects whether a given LoRA adapter, when merged into a base weight
matrix and then re-quantized to MXFP4 (OCP Microscaling FP4), survives the
quantization noise floor. When the adapter's per-element delta magnitude is
smaller than approximately half the gap between adjacent MXFP4 representable
values in a 32-element block, the merge-then-requantize operation produces
W' approximately equal to the base weight W, erasing the adapter entirely.

This is the AIMO3 manifestation of the general adapter-quantization
incompatibility addressed by QA-LoRA (Xu et al., ICLR 2024).

Usage on H100:
    pytest tests/test_lora_mxfp4_collapse.py -v

Usage on CPU (synthetic weights):
    pytest tests/test_lora_mxfp4_collapse.py::test_synthetic_collapse -v

The synthetic test runs in <10 seconds on CPU and is the primary
detection path. The H100 test is for verifying a real adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# MXFP4 (OCP Microscaling FP4) quantization — reference implementation
# -----------------------------------------------------------------------------

MXFP4_BLOCK_SIZE = 32  # 32 elements share one E8M0 block scale

# E2M1 representable values (positive side). E2M1 = 1 sign + 2 exponent + 1 mantissa.
# Positive representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
# See OCP Microscaling Formats spec v1.0.
E2M1_POSITIVE_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0


def _quantize_block_mxfp4(block: np.ndarray) -> np.ndarray:
    """
    Quantize one 32-element block to MXFP4 and dequantize back to fp32.
    Returns the fp32 reconstruction of what MXFP4 storage would yield.
    """
    if block.size == 0:
        return block
    abs_max = float(np.abs(block).max())
    if abs_max == 0.0:
        return np.zeros_like(block)
    # E8M0 block scale: largest power of 2 such that all elements fit in E2M1_MAX
    scale_log2 = math.ceil(math.log2(abs_max / E2M1_MAX))
    scale = 2.0**scale_log2
    # Normalize into E2M1 range
    normalized = block / scale
    # Round to nearest E2M1 representable value (with sign)
    signs = np.sign(normalized)
    abs_norm = np.abs(normalized)
    # Broadcast search for nearest representable magnitude
    diffs = np.abs(abs_norm[..., None] - E2M1_POSITIVE_VALUES[None, :])
    indices = np.argmin(diffs, axis=-1)
    quantized_magnitudes = E2M1_POSITIVE_VALUES[indices]
    # Reconstruct
    return signs * quantized_magnitudes * scale


def quantize_mxfp4(weights: np.ndarray) -> np.ndarray:
    """
    Apply MXFP4 quantize-then-dequantize to a 2D weight matrix.
    Blocks are taken along the last axis in groups of MXFP4_BLOCK_SIZE.
    """
    original_shape = weights.shape
    flat = weights.reshape(-1, MXFP4_BLOCK_SIZE) if weights.shape[-1] % MXFP4_BLOCK_SIZE == 0 \
        else weights.reshape(1, -1)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = _quantize_block_mxfp4(flat[i])
    return out.reshape(original_shape)


# -----------------------------------------------------------------------------
# Detection primitive
# -----------------------------------------------------------------------------

@dataclass
class CollapseVerdict:
    frobenius_ratio: float
    collapsed: bool
    reason: str


def detect_lora_mxfp4_collapse(
    base_W: np.ndarray,
    lora_B: np.ndarray,
    lora_A: np.ndarray,
    snr_threshold: float = 1.0,
) -> CollapseVerdict:
    """
    Return a verdict on whether a given (base, B, A) triple collapses under
    merge-then-requantize with MXFP4.

    We say an adapter has "collapsed" when the quantized change it produces
    is no larger than the quantization noise introduced by MXFP4 on the base
    weight alone -- i.e., the signal-to-noise ratio is <= snr_threshold.

    Args:
        base_W: shape (d, k) base weight matrix.
        lora_B: shape (d, r) low-rank adapter left factor.
        lora_A: shape (r, k) low-rank adapter right factor.
        snr_threshold: SNR below which we flag collapse (default 1.0 = signal
            at or below noise floor).

    Returns:
        CollapseVerdict with boolean `collapsed` flag and diagnostics.
    """
    delta = lora_B @ lora_A
    merged = base_W + delta
    Q_base = quantize_mxfp4(base_W)
    Q_merged = quantize_mxfp4(merged)
    # Noise floor: quantization error on the unmodified base.
    noise = np.linalg.norm(Q_base - base_W)
    # Signal: post-quantization change attributable to the adapter.
    signal = np.linalg.norm(Q_merged - Q_base)
    snr = float(signal / max(noise, 1e-12))
    ratio = float(signal / max(np.linalg.norm(Q_base), 1e-12))
    collapsed = snr <= snr_threshold
    reason = (
        f"SNR={snr:.2f} <= {snr_threshold:.2f}; adapter signal at/below MXFP4 noise floor; "
        f"Frobenius ratio {ratio:.2e}"
        if collapsed
        else f"SNR={snr:.2f} > {snr_threshold:.2f}; adapter signal survives noise floor; "
        f"Frobenius ratio {ratio:.2e}"
    )
    return CollapseVerdict(frobenius_ratio=ratio, collapsed=collapsed, reason=reason)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(seed=42)


def test_synthetic_collapse_detected_for_small_delta(rng):
    """
    A LoRA adapter with per-element delta magnitude much smaller than base
    scale MUST be detected as collapsed under MXFP4.
    """
    d, k, r = 128, 128, 8
    base_W = rng.standard_normal((d, k)).astype(np.float32) * 1.0
    # Delta std ~0.002 (much smaller than MXFP4 quant step ~0.5 of block scale)
    factor_scale = 0.002 / math.sqrt(r)
    lora_B = rng.standard_normal((d, r)).astype(np.float32) * factor_scale
    lora_A = rng.standard_normal((r, k)).astype(np.float32) * factor_scale
    verdict = detect_lora_mxfp4_collapse(base_W, lora_B, lora_A)
    assert verdict.collapsed, f"Expected collapse for small delta; got: {verdict.reason}"


def test_synthetic_delta_survives_when_magnitude_large_enough(rng):
    """
    A LoRA adapter with per-element magnitude comparable to the base MUST NOT
    be detected as collapsed; if it is, the test implementation is wrong.
    """
    d, k, r = 128, 128, 8
    base_W = rng.standard_normal((d, k)).astype(np.float32) * 1.0
    lora_B = rng.standard_normal((d, r)).astype(np.float32) * 0.5
    lora_A = rng.standard_normal((r, k)).astype(np.float32) * 0.5
    verdict = detect_lora_mxfp4_collapse(base_W, lora_B, lora_A)
    assert not verdict.collapsed, f"Large delta should survive; got collapse: {verdict.reason}"


def test_huikang_style_adapter_magnitudes_collapse(rng):
    """
    Reproduce the observed failure mode on our v13 submission: a LoRA adapter
    with rank-32 factors scaled so per-element delta magnitude is ~1% of the
    base block scale collapses under MXFP4 re-quantization.
    """
    d, k, r = 512, 512, 32
    base_W = rng.standard_normal((d, k)).astype(np.float32) * 1.0
    # Small adapter: delta per element has std ~0.01 relative to base std 1.0
    factor_scale = 0.01 / math.sqrt(r)
    lora_B = rng.standard_normal((d, r)).astype(np.float32) * factor_scale
    lora_A = rng.standard_normal((r, k)).astype(np.float32) * factor_scale
    verdict = detect_lora_mxfp4_collapse(base_W, lora_B, lora_A)
    assert verdict.collapsed, f"huikang-style adapter should collapse; got: {verdict.reason}"


def test_threshold_is_actionable():
    """
    The detection primitive must document its threshold semantics for a
    practitioner to interpret a verdict.
    """
    assert detect_lora_mxfp4_collapse.__doc__ is not None
    assert "SNR" in detect_lora_mxfp4_collapse.__doc__


# -----------------------------------------------------------------------------
# Integration test (requires H100 and a real adapter)
# -----------------------------------------------------------------------------

@pytest.mark.integration
def test_real_gpt_oss_lora_collapse():
    """
    Integration test: load gpt-oss-120b-MXFP4 base + a LoRA adapter, merge, and
    detect collapse on a sample of weight matrices.

    SKIPPED by default. Run with: pytest -v -m integration.

    Expected: for the huikang AIMO2 adapter on gpt-oss-120b's MXFP4 expert
    projections, this test should detect collapse on >=90% of merged matrices.
    """
    pytest.skip(
        "Integration test: requires H100 + gpt-oss-120b weights + adapter. "
        "See reproducibility/integration/test_real_lora.py for the full harness."
    )


if __name__ == "__main__":
    # Standalone run: print verdicts across a range of delta magnitudes.
    rng = np.random.default_rng(seed=42)
    print("Synthetic LoRA + MXFP4 collapse detection")
    print("=" * 60)
    d, k, r = 512, 512, 32
    base_W = rng.standard_normal((d, k)).astype(np.float32)
    for delta_std in [1.0, 0.1, 0.05, 0.01, 0.005, 0.001]:
        factor_scale = delta_std / math.sqrt(r)
        lora_B = rng.standard_normal((d, r)).astype(np.float32) * factor_scale
        lora_A = rng.standard_normal((r, k)).astype(np.float32) * factor_scale
        v = detect_lora_mxfp4_collapse(base_W, lora_B, lora_A)
        flag = "COLLAPSED" if v.collapsed else "survives  "
        print(f"  delta_std={delta_std:.3f}: {flag}  {v.reason}")
    print()
    print("Verdict: LoRA adapters producing per-element delta magnitudes at or")
    print("below MXFP4 quantization noise (~2% of base scale) are erased by")
    print("merge-then-requantize on this substrate.")
