# AIMO3 Writeup — Companion Package

This directory contains the submission for the AIMO3 Kaggle Writeup Prize
($15K × 2), deadline 2026-04-22 23:59 UTC.

**Paper title:** *A Practitioner's Plateau: Sixteen Falsified Modifications
to the AIMO3 Public-Consensus Pipeline on gpt-oss-120b-MXFP4.*

**Author:** Juan Sebastian Gil Pinzon (Kaggle: `sebastiangil00`, team *Hail Mary*)

**Final result:** public 42/50, private 42/50 (zero shake-down), final rank 284.

**Submission attached to:**
[`sebastiangil00/aimo3-v7-winner-fork` Version 12](https://www.kaggle.com/code/sebastiangil00/aimo3-v7-winner-fork)

---

## What this package is

Three artifacts per Section 1.5 of the main paper:

1. **Three substrate-trap detection scripts** (in `tests/`) — pytest-runnable,
   CPU-friendly, each produces a boolean verdict in <5 minutes.
2. **A reproduction specification** (`reproducibility/`) — one-command
   `reproduce.sh` with two modes (strict bitwise, stochastic faithful).
3. **A sixteen-row falsification catalog** (§4, Appendix A1) — every tested
   departure from the public-consensus AIMO3 pipeline, with hypothesis,
   mechanism, validation effect, leaderboard effect, and commit SHA.

---

## Directory layout

```
writeup/
├── main.md                         Academic-register §§1-7 (≤ ~8000 words)
├── appendix.md                     Narrative-register Appendix A1-A7
├── README.md                       This file
├── pytest.ini                      Pytest configuration (registers integration mark)
├── tests/
│   ├── test_lora_mxfp4_collapse.py        §6.2 detector — CPU, <10 s
│   ├── test_bayesian_sqrt_inversion.py    §6.3 detector — CPU, <1 s
│   └── test_eagle3_moe_zero_tokens.py     §6.1 detector — rule-based (CPU) + optional H100 integration
├── reproducibility/
│   ├── reproduce.sh                Two-mode reproduction script
│   ├── verify.py                   Strict-SHA or stochastic-support verifier
│   ├── environment.lock            Pinned versions (docker SHA, vLLM, weights, Python)
│   ├── expected_hashes.json        Reference-run answer distributions (A5)
│   └── model_weights.sha256        (TODO: populate on publication)
├── figures/
│   ├── generate_figures.py         Single-script figure generator
│   ├── fig1_ablation_forest.png    16 variants with 95% CI + BH q-values
│   ├── fig2_mxfp4_collapse_boundary.png   LoRA delta vs MXFP4 noise floor
│   ├── fig3_submission_timeline.png       22-day LB trajectory
│   └── fig4_category_breakdown.png        Per-problem-category accuracy
└── submission/
    └── notebook.py                  Verbatim source of the final submission
```

---

## How a reviewer should evaluate this package

### Rubric (100 pts, per competition rules)

| Criterion | Weight | Where to look |
|---|---|---|
| **Reproducibility** | **60** | §3 + `reproducibility/` + `tests/` (run `pytest tests/` in <2 min) |
| Clarity (lifecycle) | 10 | §2 (solution lifecycle) |
| Ablation Studies | 10 | §4 (16-row catalog) + Appendix A1 + Figure 1 |
| SOTA Comparison | 10 | §5 (table with Numina, NemoSkills, public 44/50, top-3 private) + Figure 4 (per-category) |
| Graphs & Charts | 10 | 4 figures in `figures/` |

### Two-minute evaluation path (for a time-constrained reviewer)

```bash
# 1. Read the abstract (main.md, first ~500 words)
# 2. Skim figures (figures/fig1_ablation_forest.png first)
# 3. Run the trap-detection tests on your machine:
pip install pytest numpy
pytest writeup/tests/ -v
# Expected: 15 passed, 2 skipped (integration)
# Runtime: <2 s on CPU
```

### Five-minute evaluation path

Additionally:

```bash
# 4. Read §4 (ablation catalog table) and §6 (deep dives on three traps)
# 5. Verify the reproducibility package structure exists:
ls writeup/reproducibility/
# Expected: reproduce.sh, verify.py, environment.lock, expected_hashes.json
```

### Full evaluation path (H100 required)

```bash
# 6. On an H100 80GB host with Docker image:
#    gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9
bash writeup/reproducibility/reproduce.sh
# Expected runtime: ≈25 min (ten reference problems)
python writeup/reproducibility/verify.py submission.parquet --mode stochastic
# Expected: PASS: all 10 problems verified.
```

---

## What we claim

1. The six-teen documented falsifications are **pre-committed** (Git
   timestamps precede submission timestamps; verifiable via `git log`).
2. The three trap detection scripts **run** (`pytest tests/` passes
   15/15 non-integration tests in <2 seconds on CPU).
3. The reproduction specification **executes** — on an H100 with the pinned
   Docker image, `reproduce.sh` produces a `submission.parquet` whose
   per-problem answers fall within the distributional support documented
   in `expected_hashes.json`.

## What we do NOT claim

1. That 42–44/50 is a true inference-time ceiling — at least three private
   teams scored ≥47 on this substrate.
2. That the 3 novel findings are general phenomena — each is an AIMO3-specific
   manifestation of a known general mechanism (QA-LoRA / EAGLE-3 feature
   alignment / Condorcet Jury Theorem).
3. That our validation set (n=199) is an unbiased AIMO3 sample —
   it is opportunistically aggregated.

See main.md §7 (Limitations) for full scope-of-claim discussion.

---

## Licensing

Code (tests, reproducibility scripts, figure generators): Apache 2.0.
Writeup text (main.md, appendix.md): CC-BY 4.0 — as required by AIMO3 rules for
Winner-class submissions.

---

## Contact

Issues / questions about reproducibility: open a GitHub issue at the companion
repository (linked in main.md header). For competition-specific correspondence,
the author handle is `sebastiangil00` on Kaggle.
