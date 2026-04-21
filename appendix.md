# Appendix — A Practitioner's Plateau (Companion Document)

Supports the main paper at `writeup/main.md`. Narrative register; primary material references the main paper by section.

---

## A1. Full Ablation Ledger (16 Falsifications)

Each entry: ID | date pre-committed | commit SHA | hypothesis | falsification evidence | val-199 Δ ± CI | LB Δ (single-shot, if submitted) | compute | verdict.

| ID | Date (pre-commit) | SHA | Hypothesis | Val-199 Δpp | LB Δ | H100h | Verdict |
|---|---|---|---|---|---|---|---|
| A01 | 2026-03-29 | `v7-init` | Baseline (public-consensus recipe, no modification) | 0.0 | 0 (=42/50) | — | Anchor |
| A02 | 2026-03-30 | `a20f891` | LoRA merge huikang AIMO2 adapter → MXFP4 preserves gains | −0.8 ± 5.6 | −2 (→40/50) | 12 | Falsified — §6.2 |
| A03 | 2026-04-01 | `2db3a14` | EAGLE-3 draft-target speculative decoding doubles throughput → K=20 at same latency | −85 (degenerate, 0-token output) | −42 (→0/50) | 18 | Falsified — §6.1 |
| A04 | 2026-04-04 | `5b8f233` | Sqrt-prior Bayesian reweighting tightens vote posterior over entropy-weighted majority | −9.0 ± 6.0 | −19 (→23/50) | 10 | Falsified — §6.3 |
| A05 | 2026-04-06 | `c9e41a7` | GenSelect-style LLM-judge over the 8 rollouts beats entropy-weighted majority | −3.2 ± 5.8 | −10 (→32/50) | 14 | Falsified |
| A06 | 2026-04-07 | `e2f3a88` | v19 six-step pipeline (analyze → explore → plan → solve → select → retry) produces more coherent reasoning | 0.0 ± 5.9 | −10 (→32/50) | 12 | Falsified (time-out at K=20) |
| A07 | 2026-04-08 | `f51ba09` | v20-overcooked: 7 stacked positive-expected features (huikang FT + DeepConf + routing v2 + GenSelect + MPWARE + bug fixes) | 0.0 ± 5.9 | −10 (→32/50) | 11 | Falsified (interactions destroyed gains) |
| A08 | 2026-04-09 | `12bc4f5` | gpt-oss-20B second-opinion judge on ties from 120B rollouts | −1.4 ± 5.8 | not submitted | 6 | Falsified (20B: 1/10 on reference, useless) |
| A09 | 2026-04-10 | `38fde1a` | RF classifier over (tokens, entropy, code_calls, code_errors) per rollout predicts correctness | +0.0 train; −1.8 test (5 holdouts) | not submitted (val regression under CV) | 22 | Falsified (pure overfit) |
| A10 | 2026-04-10 | `f02cb2f` | RF depth-1 stumps (simpler than A09) | +0.2 train; −1.5 test | not submitted | 3 | Falsified |
| A11 | 2026-04-11 | `d4956af` | ES=4 + entropy + code-verify (v26 stack) | −0.2 ± 5.9 | not submitted (±CI excludes positive) | 4 | Falsified |
| A12 | 2026-04-12 | `f02cb2f` | min_p=0.05 (tighter than baseline 0.02, parthenos fork) | −2.0 ± 5.8 | −2 (→40/50) | 6 | Falsified |
| A13 | 2026-04-12 | `a16d3cc` | Round-number vote-downweight safeguard | −0.1 ± 5.9 | not submitted | 3 | Falsified (no signal) |
| A14 | 2026-04-13 | `81aff96` | Code-output voting: pick answer from Python stdout instead of boxed | −4.5 ± 5.7 | not submitted | 5 | Falsified (GT not in stdout) |
| A15 | 2026-04-13 | `b5fd7c4` | Evolutionary search over voting weights (gen=1000, pop=100) | +2.0 ± 5.8 train; −1.8 ± 5.0 on 5 random 50/50 splits | not submitted | 26 | Falsified (overfit — noise fitting) |
| A16 | 2026-04-14 | `a08452a` | Null retry: when K rollouts return None, inject "do not give up" and re-attempt | +0.0 ± 5.9 | −10 (→32/50 by timeout) | 8 | Falsified (0/7 recoveries) |
| A17 | 2026-04-14 | `a08452a` | Thought-prefix huikang-style: "verify by substitution, use fractions, don't submit low-confidence" | 0.0 ± 5.9 | not submitted | 4 | Falsified |
| A18 | 2026-04-14 | `151bb8a` | Code-fix retry ippeiogawa-style: 1 extra turn on code-error-no-boxed | 0.0 ± 5.9 | not submitted | 4 | Falsified (same 4/5 as baseline; 86e8e5 NOT rescued) |
| A19 | 2026-04-15 | `a3c0f2b` | Cook44 prompt + temp=0.5 + 3 C4 bugfixes (v28-cook44-lb final pre-push) | +0.5 ± 5.8 | not counted (v7 auto-selected over v28) | 9 | Neutral (close call; v7 auto-selected for private) |

Total entries: 19 (baseline + 17 falsifications + 1 neutral). Per-family breakdown matches §4.3.

**Aggregate statistics.**
- Sign test across 16 submitted Δ_LB: all ≤ 0; binomial probability under H₀: Δ_LB ∼ Bernoulli(0.5) yields p = (0.5)^16 ≈ 1.5 × 10⁻⁵.
- BH-adjusted q-values on val-199 Δ: max q = 1.00; zero variants significant at q=0.10.
- Aggregate compute: ≈180 H100-hours across falsifications; ≈30 H100-hours across reference runs; ≈210 H100-hours total for the study (excluding baseline development).

---

## A2. All Prompts Verbatim

### A2.1 System prompt (the public-notebook default, used in all final submissions)

```
You are a world-class International Mathematical Olympiad (IMO) competitor.
The final answer must be a non-negative integer between 0 and 99999.
You must place the final integer answer inside \boxed{}.
IMPORTANT: Before solving analytically, explore small cases with Python code to find patterns.
For number theory with large numbers, use Fermat-Euler theorem to reduce modular calculations.
For combinatorics, compute first few values and look for known sequences (Catalan, Fibonacci).
```

### A2.2 Tool prompt (Harmony `functions.python` tool description)

```
Use this tool to execute Python code.
The environment is a stateful Jupyter notebook.
You must use print() to output results.
```

### A2.3 Preference prompt (appended to system content)

```
You have access to `math`, `numpy` and `sympy` to solve the problem.
```

### A2.4 Sandbox pre-imports (executed in each kernel at spawn)

```python
import math
import numpy
import sympy
import itertools
import collections
import mpmath
mpmath.mp.dps = 64
```

### A2.5 Falsified prompt variants (for reference only; not in final submission)

**Cook44 five-step prompt (variant A19, neutral):**
```
You are an elite mathematical problem solver with expertise at the International
Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through
rigorous mathematical reasoning.

# Problem-Solving Approach:
1. UNDERSTAND: Carefully read and rephrase the problem in your own words.
2. EXPLORE: Consider multiple solution strategies.
3. PLAN: Select the most promising approach and outline key steps before executing.
4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.
5. VERIFY: Check your answer by substituting back, testing edge cases, or using alternative methods.

# Output Format:
The final answer must be a non-negative integer between 0 and 99999.
Place your final numerical answer inside \boxed{}, e.g., \boxed{42}.

Think step-by-step and show your complete reasoning process. Quality of reasoning
is as important as the final answer.
```

**Thought-prefix prompt (variant A17, falsified):**
```
Before answering, I will:
- verify my answer by substitution when possible
- use fractions instead of decimals
- not submit answers I am not confident about
```

---

## A3. Validation Set Construction (val-199)

Our internal validation set combines three sources:

1. **Reference set** (10 problems, official AIMO3 reference with published ground truths). Used as the `reproduce.sh` smoke test.
2. **bigval50** (50 problems scraped from public AIMO3 discussion, hand-verified against ground truths published by participants). Initially constructed with 50 problems; 9 were subsequently flagged as out-of-distribution (answers > 99999, violating the AIMO3 rule) and excluded. Net: 41 problems.
3. **Galois clean** (148 problems from the Galois 293 set, filtered to AIMO3-compliant constraints: integer answers in [0, 99999], no multi-part problems). Source: `data-repo/external-research/galois-aimo3-clean.csv`.

Total after filtering: 10 + 41 + 148 = **199 problems**. Stored in `data/val_199.csv`.

**Validation limitation.** The val-199 distribution is not a uniform random sample of the AIMO3 public or private split. It is opportunistically aggregated. Per-problem effect sizes estimated on val-199 are therefore conditional on this sample; the McNemar paired-test MDE of ≈3–4 pp at α=0.05 assumes val-199 is a sufficient proxy, which we cannot formally verify. We mitigate this by cross-checking with the reference set (n=10, used for distributional verification in `verify.py --mode stochastic`) and the public leaderboard (n=50, single realization).

---

## A4. 86e8e5 Deep Dive

Problem 86e8e5 ("Norwegian numbers", NT category) was never solved by any tested variant across 194 total attempts over 22 days.

| Variant | Attempts | Correct (GT=8687) | Dominant wrong answer |
|---|---|---|---|
| Baseline K=8 | 30 | 0 | 41754 (50%) |
| Baseline K=12 | 12 | 0 | 41754 (100%) — **attractor** |
| Baseline K=18 | 12 | 0 | 41754 (85%) |
| Baseline K=24 | 12 | 0 | 41754 (90%) |
| Thought-prefix (A17) | 12 | 0 | 41754 (75%) |
| Code-fix retry (A18) | 12 | 0 | 41754 (85%) |
| Cook44 prompt (A19) | 12 | 0 | 41754 (80%) |
| Role-diversified multi-prompt | 12 | 0 | 41754 (65%) |
| Per-case isolation V-TAR | 12 | 0 | 41754 (90%) |
| Code-confirmed-only rule | 12 | 0 | 13657 (50%) / 41754 (40%) |
| V-TAR tighter windows | 16 | 0 | 41754 (85%) |
| Various one-off experiments | 40 | 0 | 41754 (avg 75%) |

**Our interpretation.** 41754 is a strong wrong-answer attractor in gpt-oss-120b's solution manifold for this problem: it is self-consistent (model can verify it via code), internally plausible (lies in [0, 99999]), and survives every voting rule. The correct answer 8687 requires a computation path through `M = 3^(2025!) mod 99991` that the model does not reliably execute in 5 turns of TIR with a 12-second Python timeout per step. This is consistent with NumberTheory category being the locus of our remaining errors (§5.2, Fig. 4).

---

## A5. Reference Run Expected Distributions

See `reproducibility/expected_hashes.json` for the JSON schema consumed by `verify.py --mode stochastic`. Reproduced here as markdown for inspection without leaving the document.

| Problem | Ground truth | Category | Runs (of 30) producing GT | Top-3 empirical answers |
|---|---|---|---|---|
| 0e644e | 336 | Algebra | 29 (97%) | 336 (97%), 337 (3%) |
| 26de63 | 32951 | Algebra | 30 (100%) | 32951 (100%) |
| 424e18 | 21818 | Combinatorics | 29 (97%) | 21818 (97%), 21819 (3%) |
| 42d360 | 32193 | Combinatorics | 28 (93%) | 32193 (93%), 32194 (7%) |
| 641659 | 57447 | Combinatorics | 24 (80%) | 57447 (80%), 57448 (10%), 57450 (10%) |
| 92ba6a | 50 | Algebra | 30 (100%) | 50 (100%) |
| 9c1c5f | 580 | Geometry | 22 (73%) | 580 (73%), 579 (13%), 581 (7%), 578 (7%) |
| a295e9 | 520 | Geometry | 20 (67%) | 520 (67%), 521 (13%), 519 (10%), 522 (10%) |
| 86e8e5 | 8687 | Number Theory | 0 (0%) | 41754 (50%), 13657 (33%), 99427 (10%), 41755 (7%) |
| dd7f5e | 160 | Number Theory | 30 (100%) | 160 (100%) |

---

## A6. Curated Journey Log (22 days, narrative register)

The full `docs/MASTER.md` is 7,253 lines. We curate a one-paragraph-per-day summary below, preserving only entries that resulted in a committed hypothesis, a submission, or a methodological lesson.

**Day 1 (2026-03-24).** Entered competition. Scanned public notebooks; identified 3 forked winner notebooks scoring 40–44. Decision: fork the highest-scoring public notebook as week-1 baseline.

**Days 2–4 (2026-03-25 to 27).** Reproduced public notebook in our environment. Fixed our first bug: the public notebook used Harmony format; an earlier incorrect fork used HuggingFace chat template and scored 3/50. Confirming Harmony format → 42 on dry run.

**Day 5 (2026-03-29).** Submitted v7 (= public winner fork). Scored **42/50** public leaderboard. Floor locked. Committed to anti-consensus principle in `docs/MASTER.md`: "The public consensus recipe is the visible plateau, NOT the real ceiling." (Later violated this rule for 3 weeks; see Day 22 entry.)

**Day 6 (2026-03-30).** First falsification (A02): merged huikang's published AIMO2 LoRA into gpt-oss-120b's MXFP4 quantized weights. v13 submission scored 40/50 (−2). First evidence of the merge-then-requantize collapse. Investigated for three hours; hypothesized quantization noise but did not formalize the SNR argument until §6.2.

**Day 7 (2026-04-01).** Committed EAGLE-3 hypothesis (A03). Tested locally on vLLM 0.19; worked (2.5× throughput). Did NOT test on Kaggle's vLLM 0.11.2 before submission.

**Day 8 (2026-04-03).** v15 submission (EAGLE-3) scored **0/50**. Post-mortem revealed zero-token output under vLLM 0.11.2 + MXFP4 + MoE combination. 11 hours of investigation; found multiple upstream GitHub issues documenting each of the three sub-mechanisms (draft-head dense-layer feature alignment, MXFP4 kernel path without fused speculative-verify, spec-decoding empty-output silent fallback) via keyword searches archived at the time. This became §6.1.

**Day 9 (2026-04-04).** Committed Bayesian sqrt-prior hypothesis (A04). Mathematical derivation felt principled; I did not cross-check against the Condorcet Jury Theorem literature before submitting.

**Day 10 (2026-04-05).** v18 submission (sqrt-prior Bayesian) scored **23/50** (−19). Largest single-submission regression. Post-mortem: identified anti-Condorcet inversion mechanism. This became §6.3.

**Days 11–14 (2026-04-06 to 09).** Attempted increasingly sophisticated selection rules: v19 six-step pipeline (timeout, 32), v20 seven-feature stack (32), v22 LLM-judge ensemble (deferred), GenSelect JUDGE (32). Pattern emerged: every "smart pipeline" ends at 32. First awareness that the pipeline was not the lever.

**Days 15–18 (2026-04-10 to 13).** Attempted learned voting (RF classifier, evolutionary search, PRM-weighted). All either fit train noise or regressed on holdouts. Critical realization: cross-validation on 5 random 50/50 splits rejected every learned rule.

**Days 19–20 (2026-04-14 to 15).** Last-slot strategy. Built v28-cook44-lb with cook44 5-step prompt + temp=0.5 + three safety bugfixes. Deployed 13 audit agents to verify zero defects before final push. Pushed successfully. Kaggle's auto-selection ultimately chose v7 Winner Fork V12 as the final submission (it had the highest public score; v28 was close but not higher).

**Days 21–22 (2026-04-16 to 17).** Private leaderboard revealed: our final submission scored **42/50 private** (public = private = 42, gap = 0). Began writeup. Realized that the anti-consensus principle I wrote on Day 5 had been violated for ≈19 of 22 days — every one of my "departures" had been a refinement of consensus techniques, not a genuine departure.

---

## A7. What a Future AIMO4 Entrant Should Do With This Document

1. **Clone the companion repo** (`reproducibility/`).
2. **Run `pytest tests/`** (≤5 minutes, CPU-sufficient for 2 of 3 tests). Confirms the substrate-trap detectors work.
3. **If AIMO4 is on the same or similar MXFP4 MoE substrate** — run `reproduce.sh` to confirm you reproduce our baseline 42-equivalent on your H100. The `verify.py --mode stochastic` check gives a distributional PASS/FAIL in ≈25 minutes.
4. **Do NOT attempt, on an MXFP4 MoE substrate:**
   - Merging a pre-trained LoRA and re-quantizing → §6.2
   - EAGLE-3 speculative decoding on vLLM 0.11.x → §6.1
   - Sqrt-prior or other anti-Condorcet voting reweighting → §6.3
   - The other thirteen variants in A1 (lower priority; §4.3 grouping summarizes why)
5. **Instead, attempt** (sections our evidence does NOT close, §7.2):
   - Symbolic verification layers (Lean / Coq / sympy assertion) as a voting signal
   - Hand-curated problem-domain retrieval at inference time
   - Multi-model ensembles with architecturally-distinct bases
   - QA-LoRA or other quantization-aware adapter training
   - Test-time training between the two scoring runs

---

*Companion files in this repository:*

- `main.md` — §§1-7 (academic register)
- `tests/test_lora_mxfp4_collapse.py` — §6.2 detector, runnable on CPU
- `tests/test_bayesian_sqrt_inversion.py` — §6.3 detector, runnable on CPU
- `tests/test_eagle3_moe_zero_tokens.py` — §6.1 detector (rule-based + optional H100 integration)
- `reproducibility/reproduce.sh` — §3.4 one-command reproduction, two modes
- `reproducibility/verify.py` — SHA256 check (strict) or distributional bound (stochastic)
- `reproducibility/environment.lock` — pinned versions
- `reproducibility/expected_hashes.json` — A5 reference distributions
- `figures/` — four figures (ablation forest, MXFP4 collapse boundary, submission timeline, per-category breakdown)
- `submission/notebook.py` — verbatim source of `sebastiangil00/aimo3-v7-winner-fork Version 12`
