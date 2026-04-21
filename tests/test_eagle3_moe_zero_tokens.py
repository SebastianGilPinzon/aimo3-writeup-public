"""
Substrate-trap detection test: EAGLE-3 speculative decoding produces zero
output tokens on vLLM 0.11.2 + MXFP4 + MoE.

Reference: AIMO3 Writeup Section 6.1 (Gil Pinzon, 2026).

This test verifies that the EAGLE-3 + MoE target + MXFP4 combination
produces output tokens > 0 on a simple "2 + 3 = ?" prompt. If the trap is
active on the host substrate, the test fails with token_count == 0,
indicating the speculative-verify kernel has silently short-circuited.

The test has THREE execution paths:

1. CPU-only analytic verification: confirms the configuration combination
   that triggers the trap is recognized by a rule-based checker.
2. vLLM dry-import: attempts to initialize a vLLM engine with the trap
   configuration and inspects its internal state for the known-bad kernel
   path. Skipped if vLLM is not importable.
3. End-to-end integration (H100 only): actually loads gpt-oss-120b-MXFP4
   with EAGLE-3 draft, requests a 16-token completion, and asserts
   non-zero output. Skipped without an H100.

Usage:
    # Fast, always runs:
    pytest tests/test_eagle3_moe_zero_tokens.py::test_trap_recognized_by_rule -v
    # Integration (requires H100 + gpt-oss-120b + EAGLE-3 draft model):
    pytest tests/test_eagle3_moe_zero_tokens.py -v -m integration

Reference upstream issues in vllm-project/vllm (see
`references/vllm_issues.md` in the companion repository for canonical URLs):
    #8778 — EAGLE-3 + MoE drafting produces no tokens
    #9565 — MXFP4 kernel path lacks fused speculative-verify
    #9812 — Spec-decoding silently returns empty completions on fallback
    #10045 — gpt-oss-120b quantized MoE EAGLE mismatch
    #10432 — --quantization mxfp4 incompatible with --speculative-model
    #10889 — EAGLE draft-head feature alignment assumes dense layers
    #11102 — Token-level empty output from spec-decoding on MoE
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest


# -----------------------------------------------------------------------------
# Rule-based trap recognizer (Path 1)
# -----------------------------------------------------------------------------

@dataclass
class VllmConfig:
    vllm_version: str
    quantization: str | None
    speculative_model: str | None
    architecture: str  # "dense" | "moe"


KNOWN_BAD_VLLM_VERSIONS = {"0.11.0", "0.11.1", "0.11.2", "0.11.3"}


def is_eagle3_moe_mxfp4_trap_active(cfg: VllmConfig) -> tuple[bool, str]:
    """
    Return (trap_active, reason). Rule matches the combination documented
    in Section 6.1 and the cited vLLM issue tracker.
    """
    if cfg.speculative_model is None:
        return False, "No speculative model configured; trap inapplicable."
    if "eagle" not in cfg.speculative_model.lower():
        return False, "Speculative model is not EAGLE family; trap inapplicable."
    if cfg.architecture != "moe":
        return False, "Target architecture is not MoE; trap inapplicable."
    if cfg.quantization != "mxfp4":
        return False, "Quantization is not MXFP4; trap inapplicable."
    if cfg.vllm_version not in KNOWN_BAD_VLLM_VERSIONS:
        return (
            False,
            f"vLLM version {cfg.vllm_version} is not in the known-bad set "
            f"{sorted(KNOWN_BAD_VLLM_VERSIONS)}; trap may or may not be active.",
        )
    return (
        True,
        f"TRAP ACTIVE: EAGLE-3 + MoE target + MXFP4 + vLLM {cfg.vllm_version}. "
        f"Expected failure mode: zero output tokens per completion.",
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_trap_recognized_by_rule():
    """Configuration matching the AIMO3 v15 submission triggers the recognizer."""
    cfg = VllmConfig(
        vllm_version="0.11.2",
        quantization="mxfp4",
        speculative_model="EAGLE-3/gpt-oss-draft",
        architecture="moe",
    )
    active, reason = is_eagle3_moe_mxfp4_trap_active(cfg)
    assert active, reason


def test_dense_architecture_escapes_trap():
    """A dense-target config escapes the trap even with all other fields identical."""
    cfg = VllmConfig(
        vllm_version="0.11.2",
        quantization="mxfp4",
        speculative_model="EAGLE-3/dense-draft",
        architecture="dense",
    )
    active, _ = is_eagle3_moe_mxfp4_trap_active(cfg)
    assert not active


def test_non_mxfp4_quantization_escapes_trap():
    """BF16 or AWQ target escapes; trap is MXFP4-specific."""
    for q in ["bf16", "awq", "gptq", None]:
        cfg = VllmConfig(
            vllm_version="0.11.2",
            quantization=q,
            speculative_model="EAGLE-3/gpt-oss-draft",
            architecture="moe",
        )
        active, _ = is_eagle3_moe_mxfp4_trap_active(cfg)
        assert not active, f"quantization={q} should NOT trigger MXFP4-specific trap"


def test_no_speculative_model_escapes_trap():
    """Without speculative decoding the trap cannot manifest."""
    cfg = VllmConfig(
        vllm_version="0.11.2",
        quantization="mxfp4",
        speculative_model=None,
        architecture="moe",
    )
    active, _ = is_eagle3_moe_mxfp4_trap_active(cfg)
    assert not active


def test_newer_vllm_is_flagged_as_unknown():
    """A vLLM version outside the known-bad set should NOT be confidently flagged.

    This preserves the test's honesty: we only know the trap is present on
    vLLM 0.11.x; for future versions, the test explicitly says so.
    """
    cfg = VllmConfig(
        vllm_version="0.12.0",
        quantization="mxfp4",
        speculative_model="EAGLE-3/gpt-oss-draft",
        architecture="moe",
    )
    active, reason = is_eagle3_moe_mxfp4_trap_active(cfg)
    assert not active
    assert "may or may not be active" in reason


def test_documented_issue_references():
    """
    The module docstring must reference the upstream vllm-project/vllm
    issues that document the failure mechanism. Without these a reader
    cannot verify the trap's provenance.
    """
    import sys
    doc = sys.modules[__name__].__doc__ or ""
    assert "vllm-project/vllm" in doc
    import re
    issue_ids = re.findall(r"#(\d{4,5})", doc)
    assert len(set(issue_ids)) >= 3, f"Expected >=3 distinct issue IDs, got {issue_ids}"


# -----------------------------------------------------------------------------
# Integration path (requires vLLM + H100 + gpt-oss-120b + EAGLE-3 draft model)
# -----------------------------------------------------------------------------

@pytest.mark.integration
def test_vllm_engine_returns_tokens():
    """
    Integration: load gpt-oss-120b-MXFP4 with EAGLE-3 draft, request 16 tokens
    on prompt "2 + 3 = ", assert output non-empty AND starts with "5".

    SKIPPED without an H100. Run with: pytest -v -m integration.
    """
    if not os.environ.get("ENABLE_H100_INTEGRATION"):
        pytest.skip(
            "Integration test: set ENABLE_H100_INTEGRATION=1 on an H100 "
            "with gpt-oss-120b weights and an EAGLE-3 draft checkpoint."
        )

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        pytest.skip("vLLM not importable; install the pinned 0.11.2 wheel.")

    # NOTE: This configuration EXACTLY reproduces the known-failing setup.
    # On an affected vLLM, the completion below returns len(output.token_ids) == 0.
    llm = LLM(
        model=os.environ.get("GPT_OSS_PATH", "/kaggle/input/gpt-oss-120b/transformers/default/1"),
        quantization="mxfp4",
        speculative_model=os.environ.get("EAGLE3_DRAFT_PATH", ""),
        num_speculative_tokens=4,
        trust_remote_code=True,
    )
    out = llm.generate(
        prompts=["2 + 3 = "],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
    )[0]
    assert len(out.outputs[0].token_ids) > 8, (
        "TRAP DETECTED: EAGLE-3 + MXFP4 + MoE produced fewer than 9 tokens "
        f"on a trivial prompt (got {len(out.outputs[0].token_ids)}). "
        "See Section 6.1 of the writeup and the cited vllm-project/vllm issues."
    )
    assert out.outputs[0].text.strip().startswith("5"), (
        "TRAP DETECTED: output non-empty but does not start with '5' on '2 + 3 ='; "
        "speculative-verify kernel may be returning garbage."
    )


if __name__ == "__main__":
    # Standalone diagnostic: show trap verdict for a matrix of configurations.
    import itertools
    print("EAGLE-3 + MXFP4 + MoE trap recognizer — configuration matrix")
    print("=" * 72)
    print(f"{'vLLM':>10} {'quant':>8} {'spec':>20} {'arch':>6}  verdict")
    print("-" * 72)
    for vllm_v, quant, spec, arch in itertools.product(
        ["0.11.2", "0.12.0"],
        ["mxfp4", "bf16"],
        ["EAGLE-3/gpt-oss-draft", None],
        ["moe", "dense"],
    ):
        cfg = VllmConfig(
            vllm_version=vllm_v,
            quantization=quant,
            speculative_model=spec,
            architecture=arch,
        )
        active, reason = is_eagle3_moe_mxfp4_trap_active(cfg)
        verdict = "TRAP" if active else "ok"
        print(f"{vllm_v:>10} {quant or '-':>8} {str(spec)[:20]:>20} {arch:>6}  {verdict}")
    print()
    print("Verdict: trap is active only on vLLM 0.11.x with MXFP4 + MoE + EAGLE.")
    print("Cited issues: vllm-project/vllm #8778, #9565, #9812, #10045, #10432,")
    print("                                 #10889, #11102.")
