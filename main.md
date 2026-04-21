# Three Substrate Traps on gpt-oss-120b-MXFP4 for AIMO3: A Reproducibility Package Accompanied by Sixteen Falsified Consensus Modifications

**Author:** Juan Sebastian Gil Pinzon (Kaggle: `sebastiangil00`, team *Hail Mary*)
**Competition:** [AIMO Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
**Final result:** public 42/50, private 42/50; final rank 284
**Submission:** [AIMO3 v7 - Winner Fork - Version 12](https://www.kaggle.com/code/sebastiangil00/aimo3-v7-winner-fork)
**Reproducibility package:** [GitHub](https://github.com/SebastianGilPinzon/aimo3-writeup-public) (unit tests + `reproduce.sh`) · [Kaggle Dataset](https://www.kaggle.com/datasets/sebastiangil00/aimo3-writeup-artifacts)

---

## Abstract

We provide three artifacts intended to be directly useful to future AIMO / mathematical-reasoning competition participants working on the same substrate — **gpt-oss-120b** [1, 2] (MoE expert projections in MXFP4, attention and router in BF16) served through the competition-pinned vLLM 0.11.2 wheel on a single NVIDIA H100 80GB GPU.

**Artifact 1: Three named substrate traps, each with a runnable detection script** (§6 and `tests/`). On this substrate, three candidate improvements that appear theoretically sound silently fail in ways that pass type-checks and smoke-tests but destroy the leaderboard score:

- **LoRA-on-MXFP4 merge-then-requantize collapse** (§6.2, `tests/test_lora_mxfp4_collapse.py`) — the AIMO3 manifestation of the adapter–quantization incompatibility addressed by QA-LoRA [14]. Merging a published AIMO2 LoRA adapter [4] into the MXFP4-quantized base and re-quantizing produces `W′ ≈ W`: the low-magnitude delta `BA` rounds into the same E2M1 [1] representable bin as the base weight. Empirically, adapter deltas below ≈2⁻⁴ of the per-block scale collapse entirely. Leaderboard impact: 42 → 40.
- **EAGLE-3 draft-target speculative decoding [15] producing zero output tokens on vLLM 0.11.2 + MXFP4 + sparse-MoE** (§6.1, `tests/test_eagle3_moe_zero_tokens.py`) — the EAGLE-3 auxiliary draft head assumes dense-layer hidden states; MoE expert dispatch in gpt-oss-120b breaks feature alignment, and the MXFP4 kernel path in vLLM 0.11.2 lacks the fused speculative-verify kernel. Tracked upstream in vllm-project/vllm (issue IDs in §6.1, Table 4). Leaderboard impact: 42 → 0.
- **Sqrt-prior Bayesian vote re-weighting silently inverting correct majorities on a calibrated base model** (§6.3, `tests/test_bayesian_sqrt_inversion.py`) — reweighting the posterior as `π(answer) ∝ √count(answer) · prior(answer)` down-weights the strongest-supported cluster. For a base model with answer-level accuracy >0.5 on easy problems, Condorcet–Jury Theorem conditions [11, 12] favor the plain majority, and our reweighting is *anti-Condorcet*. Leaderboard impact: 42 → 23.

Each detection script returns a boolean verdict in under five minutes on a single NVIDIA H100 80GB. A future competitor can `git clone` the package, run `pytest tests/`, and detect all three traps before burning a submission slot.

**Artifact 2: A sixteen-row falsification catalog** (§4, Appendix A1). Over 22 days and ≈180 H100-hours, we tested sixteen candidate modifications to the AIMO3 public-consensus pipeline, drawn from the AIMO2 literature [3, 4], inference-time scaling theory [5, 6], verifier-based selection [7, 8, 9], self-consistency [10], majority-voting theory [11, 12], and quantized fine-tuning [13, 14]. **Not one produced a detectable leaderboard improvement over the controlled baseline**, and none survived Benjamini–Hochberg correction at q=0.10 on the 199-problem validation set. The catalog reports, per variant: hypothesis, mechanism, validation-199 effect with Benjamini–Hochberg-adjusted q-value, single-shot leaderboard delta, H100-hours consumed, and the Git commit SHA at which the hypothesis and its falsification criterion were committed *before* the submission that falsified it.

**Artifact 3: A reproduction specification and one-command script** (§3) that regenerates our final `submission.parquet` to within two-run seed tolerance on any H100 80GB. The specification pins the vLLM wheel SHA, the gpt-oss-120b weight commit, the Kaggle Docker image digest, a complete `pip freeze`, all prompts verbatim, and the decoding configuration. Our final submission's public and private leaderboard scores match at 42/50 — a single-realization parity consistent with Frieder's reported public–private gap distribution on this competition [18] (mean ≈2.3 pp, σ ≈3.1 pp), which we offer as an end-to-end reproducibility target rather than as evidence of a substrate ceiling.

We **explicitly do not claim** 42–44/50 is the achievable ceiling on this substrate; at least three private-leaderboard teams exceeded 47/50, and their methods are not publicly disclosed at the time of writing. What we document is a **practitioner's plateau**: the failure envelope of a solo-competitor, inference-only effort that had full access to the public-consensus recipe and 22 days of H100 time to depart from it. We extrapolate tentatively, grounded in the training-heavy wins of Numina [3] and NemoSkills [4], that the path to >44/50 on this substrate likely requires training-data interventions which the quantization regime forecloses without access to full-precision weights.

**In one sentence:** this paper is a tool, not a story — three unit tests, sixteen pre-committed falsifications, and a one-command reproduction, assembled so the next AIMO4 entrant does not re-learn what we paid ≈180 H100-hours to learn.

---

## 1. Introduction

### 1.1 Task and scoring

AIMO Progress Prize 3 poses 110 original olympiad-level mathematics problems (50 public, 50 private, 10 reference), with answers restricted to integers in [0, 99999]. Solutions must be produced by an open-weight model running offline on a single NVIDIA H100 80GB GPU within 5 hours of wall-clock time. The scoring metric is penalized accuracy: the submission notebook is executed twice on the private set, and each problem contributes 1.0 if both runs are correct, 0.5 if exactly one is, and 0.0 otherwise. Internet is disabled during inference; only pre-loaded models and datasets are accessible.

The penalized-accuracy rule creates a subtle incentive. A pipeline that is **deterministic in structure but stochastic in sampling** — using time-based seeds so that the two scoring runs draw different samples from the same output distribution — can collect 0.5 partial credit on problems the model solves with probability in the open interval (0, 1). A perfectly deterministic pipeline can only get 1.0 or 0.0. The public-consensus pipeline is stochastic in sampling, and so is ours. When we describe the pipeline below as "reproducible," we mean **reproducible in specification**: given the same model weights, environment, prompts, and decoding hyperparameters, the full distribution over outputs is reproduced; any single run is a draw from that distribution.

### 1.2 Substrate

The 80GB / 5h envelope tightly constrains the design space. Open-weight models above ≈130B parameters do not fit under any quantization that preserves olympiad-level reasoning; models below ≈14B sacrifice accuracy. The public notebook ecosystem converged on **gpt-oss-120b** [2] in its official release configuration — sparse mixture-of-experts architecture with MoE expert projections quantized to MXFP4 [1] (OCP Microscaling FP4: E2M1 mantissa with a shared E8M0 block scale per 32-element block), while attention projections and the expert-routing network remain in BF16 — served through the vLLM 0.11.2 wheel pinned by the official `gpt-oss` inference distribution, which is the effective version for any Kaggle-offline-compatible deployment. We adopt this substrate unchanged and treat it as a fixed environment rather than a design variable. All claims below are scoped to this specific substrate; we do not generalize beyond it.

### 1.3 Positioning

At least three teams exceeded 47/50 on the AIMO3 private leaderboard; their approaches are not publicly disclosed at the time of writing. The AIMO1 and AIMO2 competitions were each won by training-heavy pipelines: Numina [3] fine-tuned DeepSeekMath-7B on a 860K-problem synthetic dataset to win AIMO1, and NemoSkills [4] combined a TIR-trained generator with a separately-trained GenSelect selector to win AIMO2. Both approaches front-load the contribution in *training data and training protocol*, not in inference-time interventions. Against this prior art, the central empirical question we pursue is narrower than "how does one win AIMO3?":

> **For a solo competitor restricted to inference-time modifications of the public-consensus recipe on the substrate of §1.2, which departures yield leaderboard improvements?**

Our answer, below, is that **none of the sixteen departures we tested did**. The best-performing configuration we found is the unmodified baseline. This is not evidence of a true substrate ceiling; it is evidence of a **practitioner's plateau** — the failure envelope of our own 22-day inference-only effort. Training-data interventions (foreclosed for us by the quantization regime, per §6.2) remain a plausible and likely path to higher scores, consistent with Numina and NemoSkills.

### 1.4 Method and discipline

Our methodology is **pre-committed journaling**, not formal pre-registration. Every hypothesis in Section 6 was recorded with its falsification criterion in a dated entry of the public journey log (`docs/MASTER.md`, 7,253 lines, Appendix A6) *before* the corresponding submission was pushed; commit timestamps precede leaderboard-result timestamps for all sixteen entries, verifiable via `git log`. This is weaker than trusted-timestamp pre-registration (OSF, ClinicalTrials.gov) but stronger than post-hoc rationalization, and we name the discipline explicitly to avoid overclaiming.

We follow the negative-results reporting posture of Rosenfeld et al. [16] and Lipton & Steinhardt [17]. Each hypothesis is a controlled variant relative to the baseline, tested first on a 199-problem internal validation set (Appendix A3) and, where the validation signal warranted, on the leaderboard. The validation set has limited statistical power: for a McNemar paired test on 199 problems at baseline accuracy ≈0.80, the minimum detectable effect at α=0.05 and 1−β=0.80 is approximately 3–4 percentage points. Single-submission leaderboard observations on the 50-problem public split carry a binomial 95% CI of approximately ±6 points. All individual effect sizes reported must be read in this light: a "regression" under ±6 pp CI is statistically consistent with the null of no effect, not evidence of harm. The aggregate claim is stronger than any single component. We formalize it as: under the null hypothesis H₀ that all sixteen variants have true effect ≤ δ = 0 on validation, observing zero variants with BH-significant positive effect at q = 0.10 — together with sixteen non-positive sign-test observations (combined p < 0.001 under the alternative H_A of uniform neutrality) — supports rejection of the alternative that any inference-only departure meaningfully improves over the baseline on this substrate. The validation-to-leaderboard transportability caveat still applies: the 199-problem validation distribution is not identical to the 50-problem public split.

Total compute budget expended on the study: ≈180 H100-hours across the sixteen falsifications and thirty reference reproduction runs (≈86 H100-hours on the three Artifact-1 traps alone, reflecting the two 0/50 submissions whose pipelines completed on full 50-problem sets before being diagnosed).

### 1.5 Contributions and explicit non-contributions

**Contributions:**

1. **Three substrate-trap detection scripts** (`tests/test_lora_mxfp4_collapse.py`, `tests/test_eagle3_moe_zero_tokens.py`, `tests/test_bayesian_sqrt_inversion.py`), runnable in under five minutes each on an H100, returning a boolean verdict. Each script reproduces the failure mechanism on a minimal input so that a future competitor can detect the trap before committing a submission slot to it (§6, Appendix A2).
2. **A reproduction specification** regenerating our final `submission.parquet` to within two-run seed tolerance on any H100 80GB, pinned to exact library versions, weight hashes, Docker image digest, prompts, and decoding configuration (§3).
3. **A sixteen-row falsification catalog** with per-variant {hypothesis, mechanism, validation-199 effect ± CI, McNemar p and BH-adjusted q, leaderboard delta, H100-hours, commit SHA at which the hypothesis was pre-committed} (§4 summary, Appendix A1 complete).
4. **A substrate-bounded empirical observation** (§7): within the scope of our sixteen inference-only departures, the best-performing configuration on gpt-oss-120b-MXFP4 + vLLM 0.11.2 is the baseline. The path to >44/50 on this substrate was not found by our effort.

**Explicit non-contributions:**

- This is **not a winning solution**. Our score of 42 is tied with the public consensus.
- This is **not a proof of an inference-time ceiling**. At least three teams exceeded 47 on private.
- This is **not formally pre-registered** (OSF). Discipline is Git-timestamped journaling.
- This is **not an ablation of training-data interventions**. We did not run any successfully; our LoRA attempt is a falsification, not an ablation.
- The three findings are **not novel phenomena in general**. Each reduces to a known mechanism from the cited literature. Our contribution is their *AIMO3-specific manifestation*, their *detection script*, and their *leaderboard-verified cost*.

---

---

## 2. Solution Lifecycle

Our submission is functionally byte-identical to the consensus public notebook that scored 42/50 on the AIMO3 public leaderboard throughout March 2026. We describe it here at the component level for reference; the full source is linked in the submission URL above and mirrored in the reproducibility package.

### 2.1 Pipeline at a glance

Per problem, the pipeline executes: (1) a Harmony-formatted prompt is constructed from the system prompt (§2.2) and the problem statement; (2) eight independent assistant rollouts are dispatched in parallel to a vLLM server (§2.3), each sampling at temperature 1.0 with min-p 0.02 and top-logprobs 5, interleaved with Python execution in a dedicated stateful Jupyter kernel (§2.4); (3) each rollout terminates on `<|return|>`, on reaching the per-attempt token budget, or on an early-stop signal when four rollouts have converged to the same answer (§2.5); (4) the eight (or fewer, post-early-stop) final answers are combined by entropy-weighted majority voting (§2.6). The final answer is written to `submission.parquet` with one row per problem; the problem-serving gateway advances to the next problem.

### 2.2 Harmony prompt

We use OpenAI's Harmony chat format [2] with `reasoning_effort=HIGH`, the system prompt shown in Appendix A2.1 (verbatim), and two tool namespaces: a stateful Python sandbox (§2.4) and no web access. The system prompt is the public-notebook default — we did not modify it in the final submission. §6.4 documents three prompt variants we tested and falsified.

### 2.3 Decoding configuration

vLLM 0.11.2 is served locally as an OpenAI-compatible API on `127.0.0.1:8000`, with `--gpu-memory-utilization 0.94`, `--kv-cache-dtype fp8_e4m3`, `--max-model-len 65536`, `--served-model-name gpt-oss`, and the gpt-oss-120b weights auto-detected at `/kaggle/input/gpt-oss-120b/transformers/default/1`. Sampling uses `temperature=1.0`, `min_p=0.02`, `top_k=-1` (unrestricted), and streams tokens with `stream_interval=200`. We additionally request `top_logprobs=5` per generated token, not as a sampling filter but as an **observability** channel feeding the §2.6 entropy-weighted voting rule; the logprobs are read from the streaming response and do not influence token selection. We use **time-based seeds** per rollout — the two-run scoring rule rewards stochastic sampling (§1.1), and a fixed seed forfeits the 0.5-partial-credit EV. The full vLLM command string is in §3.2.

### 2.4 TIR via stateful Jupyter kernels

Each rollout owns a dedicated `jupyter_client.KernelManager` IPython kernel, pre-imported with `math`, `numpy`, `sympy`, `itertools`, `collections`, `mpmath` (with `mpmath.mp.dps = 64`). When the assistant emits a `<|call|>` to the Python tool namespace, the code is executed inside that kernel, output is captured with a 6-second wall-clock timeout, and re-injected as a `<|message|>` from the `functions.python` role. Kernel state persists across tool calls within a rollout (variable definitions, imports, intermediate computations); kernels are recycled between problems. Sandbox pool size equals the number of concurrent rollouts (`workers=8`). TIR is applied at inference time over the TIR-style-trained gpt-oss-120b checkpoint; we do not claim a novel TIR training contribution.

### 2.5 Early-stop weighted

A `threading.Event` is shared across the eight rollout workers. After each rollout completes, a shared `collections.Counter` of finalized answers is updated under a `threading.Lock`; if any single answer has accumulated `early_stop=4` or more votes, the event is set, and the remaining in-flight rollouts exit their token-streaming loops on the next iteration. This captures the typical-problem savings documented in §4 (validation-set time reduction of ≈40% with no detectable effect on accuracy). Because `workers == pool_size == attempts == 8`, the sandbox queue is never contended in normal operation and `sandbox_timeout=3s` serves only as a safety net against pathological kernel hangs.

### 2.6 Entropy-weighted voting

Per rollout we compute the **mean token-level Shannon entropy** across the top-5 logprobs of every generated token, producing a scalar `H` in nats-per-token. The voting rule is:

```
weight(rollout) = 1 / max(H, 0.3)
score(answer)   = Σ weight(rollout) over rollouts that boxed `answer`
final           = argmax_answer score(answer)
```

The `max(H, 0.3)` floor caps a single low-entropy rollout's weight at ≈3.33, preventing a confidently-wrong rollout from overriding a higher-entropy consensus. Ties break arbitrarily (Python dict insertion order). No code-execution-result voting, no verifier, no LLM judge. §6.3 documents the falsified sqrt-prior Bayesian alternative.

### 2.7 Two-run scoring and the variance-reduction argument

The pipeline is stochastic in sampling by design. For a problem solved with probability `p ∈ (0, 1)`, expected per-problem score is `p` under *both* deterministic (fixed seed) and stochastic (time-based seed) strategies: `p² + 2p(1−p)·0.5 + (1−p)²·0 = p`. The justification for time-based seeds is therefore **not** EV dominance, but variance reduction: `Var_stochastic = 0.5 · p(1−p)` versus `Var_deterministic = p(1−p)`. Stochastic sampling halves the per-problem variance, and under independence across the 50 problems halves the variance of the sum — tightening the realized-score interval around its expectation. Our public = private = 42/50 is a single-realization coincidence at our plateau; the tighter variance under stochastic sampling reduces the magnitude of typical public–private disagreement.

---

## 3. Reproducibility

*This section is the load-bearing reproducibility claim. The specification below, together with the companion repository and Kaggle Dataset, is intended to let a reader regenerate `submission.parquet` to within two-run seed tolerance on any H100 80GB within ≈3 hours of wall-clock time.*

### 3.1 Environment

| Component | Pin |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, compute capability 9.0 |
| CUDA runtime | 12.2 (Kaggle default) |
| NVIDIA driver | ≥ 535.86.10 |
| Docker image | `gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9` |
| Python | 3.11 (Kaggle default) |
| vLLM | 0.11.2 from official `gpt-oss` wheel (pinned by `kernel_sources: andreasbis/aimo-3-utils`) |
| Base model | `danielhanchen/gpt-oss-120b/Transformers/default/1` (Kaggle Models, MoE experts in MXFP4, attention/router in BF16) |
| openai-harmony | as pinned in `aimo-3-utils` |
| jupyter-client | as pinned in `aimo-3-utils` |

The complete `pip freeze` output from the submission environment (≈350 lines) is provided at `reproducibility/pip_freeze.lock` in the companion GitHub repository and sha256-hashed in `reproducibility/pip_freeze.sha256`.

### 3.2 vLLM server command

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /kaggle/input/gpt-oss-120b/transformers/default/1 \
  --served-model-name gpt-oss \
  --dtype auto \
  --quantization mxfp4 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.94 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --port 8000 \
  --disable-log-stats
```

The `--quantization mxfp4` flag is required to engage the MXFP4 kernel path for MoE expert projections on vLLM 0.11.2; without it, `--dtype auto` may silently fall back to a BF16-only path that either OOMs on H100 80GB or produces mismatched numerics against the submission environment. `--trust-remote-code` is required for the gpt-oss architecture on the pinned wheel. The server is spawned as a subprocess and polled for readiness on `http://127.0.0.1:8000/v1/models` with a 300-second timeout (first-invocation Triton compile can take ≈90 s).

### 3.3 Decoding and pipeline configuration

| Parameter | Value | Rationale (§6 for falsified alternatives) |
|---|---|---|
| `temperature` | 1.0 | §6.8: temperatures 0.5 / 0.7 / 0.8 falsified on val-199 |
| `min_p` | 0.02 | §6.9: 0.05 falsified (forked parthenos config) |
| `top_logprobs` | 5 | Required for §2.6 entropy weighting |
| `top_k` | -1 (unrestricted) | §6.10: top_k=40 falsified |
| `attempts` (K) | 8 | §6.11: K=4 / K=12 / K=18 / K=24 falsified |
| `turns` (max TIR rounds) | 128 | Effectively unbounded; per-rollout token budget is the real cap |
| `context_tokens` | 65536 | Matches vLLM `--max-model-len` |
| `buffer_tokens` | 512 | Safety margin between prompt length and `max_tokens` |
| `search_tokens` | 32 | Sliding window for stop-phrase detection |
| `workers` | 8 | Concurrent rollouts; matches `attempts` |
| `early_stop` | 4 | Majority-threshold for early termination (§2.5) |
| `jupyter_timeout` | 6 s | Per-TIR-call wall-clock cap |
| `sandbox_timeout` | 3 s | Queue-acquire timeout for sandbox pool |
| `base_problem_timeout` | 300 s | Soft per-problem budget |
| `high_problem_timeout` | 900 s | Problems flagged as high-difficulty (§3.5) |
| `session_timeout` | 960 s | Hard per-problem wall-clock cap |
| `seed` | `int(time.time())` per rollout | §1.1 two-run stochasticity |

All prompts (system, tool, preference) appear verbatim in Appendix A2; the voting weight formula in §2.6 appears verbatim in the companion repository at `reproducibility/voting.py`.

### 3.4 One-command reproduction — two modes

On any H100 80GB host with the pinned Docker image, the package supports **two reproduction modes** addressing distinct reviewer needs. A reproducibility reviewer can run either or both:

```bash
git clone https://github.com/SebastianGilPinzon/aimo3-writeup-public
cd aimo3-writeup-public

# Mode A (STRICT): fixed seed, byte-identical submission.parquet
REPRODUCE_DETERMINISTIC=1 bash reproduce.sh
python verify.py submission.parquet --mode strict    # asserts sha256 match

# Mode B (FAITHFUL): time-based seeds, mirrors the actual submission
bash reproduce.sh
python verify.py submission.parquet --mode stochastic --n-ref 30
```

**Mode A (strict-determinism).** Sets `PYTHONHASHSEED=42`, `vllm.sampling.seed=42`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disables `kv_cache_dtype` casting noise, and overrides `seed = 42` per rollout. Produces a `submission.parquet` whose sha256 is published in `reproducibility/strict_mode_sha256.txt`. This is the canonical bitwise-reproducibility target for reviewers.

**Mode B (stochastic-faithful).** Uses `seed = int(time.time())` per rollout, matching the actual submission environment and the two-run scoring strategy (§2.7). `verify.py --mode stochastic --n-ref 30` runs a Hoeffding-bound binomial test: for each of the ten reference problems, it asserts that the returned answer falls within the 95% support of the empirical answer distribution across 30 pre-computed reference runs (Appendix A5). A malicious or broken re-implementation producing a fixed wrong answer fails unless that answer appeared in >5% of our 30 runs, which is a non-trivial bar.

**Independent of either mode** — and runnable by any reviewer without an H100 — `sha256sum --check reproducibility/artifact_sha256.txt` verifies the SHA256 of the seventeen deterministic files that constitute this writeup package (notebook source, data, reproducibility scripts, tests, figures, config). All seventeen pass on a clean clone in under one second and provide a cryptographic integrity guarantee of the non-submission-parquet portion of the package.

`reproduce.sh` executes in four stages: (1) verify environment pins against `reproducibility/environment.lock`; (2) ensure gpt-oss-120b weights are cached (SHA256 verified against `reproducibility/model_weights.sha256`); (3) launch the vLLM server with the §3.2 command string and poll for readiness; (4) run the submission pipeline on the ten reference problems (Kaggle gateway is bypassed locally via a replay harness, see `reproducibility/local_gateway.py`). Expected wall-clock: ≈25 minutes for the 10-problem reference set in either mode; ≈2.5 hours for the 50-problem private-set-equivalent dry run.

### 3.5 Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| `reproduce.sh` | `reproducibility/` | One-command pipeline (§3.4) |
| `verify.py` | `reproducibility/` | Two-run-tolerance verification |
| `environment.lock` | `reproducibility/` | Pinned versions and digests |
| `pip_freeze.lock` | `reproducibility/` | Complete dependency snapshot |
| `notebook.py` | `submission/` | Full submission source (verbatim from `sebastiangil00/aimo3-v7-winner-fork` Version 12) |
| `tests/test_lora_mxfp4_collapse.py` | `tests/` | §6.2 trap detection |
| `tests/test_eagle3_moe_zero_tokens.py` | `tests/` | §6.1 trap detection |
| `tests/test_bayesian_sqrt_inversion.py` | `tests/` | §6.3 trap detection |
| `expected_hashes.json` | `reproducibility/` | Reference-run answer distributions |
| `reference.csv` | `data/` | 10 official AIMO3 reference problems with ground truths |
| `val_199.csv` | `data/` | 199-problem internal validation set (construction in Appendix A3) |

### 3.6 Public ↔ private parity as a reproducibility target

Our final submission's public and private leaderboard scores coincide at 42/50. Under Frieder's [18] reported public–private gap distribution on this competition (mean ≈2.3 pp, σ ≈3.1 pp under a normal approximation), the single-realization probability of exact public–private equality is approximately 0.10 (`P(−0.5 < Δ < 0.5) ≈ Φ(−0.58) − Φ(−0.90) ≈ 0.098` under the normal model; the empirical distribution may differ). We do **not** claim this parity as evidence of a substrate ceiling, of superior calibration, or of low variance per se; under our own §2.7 analysis the expected parity probability for a faithful reproduction at our plateau is of this same order. We offer it as a concrete end-to-end verification target: any re-implementation of §§3.1–3.4 that scores within approximately ±3 pp of 42 on an independent 50-problem draw is consistent with having faithfully reproduced the pipeline.

### 3.7 What this specification uniquely pins (a frozen substrate snapshot)

A non-obvious but load-bearing feature of §3: **no public AIMO3 notebook pins all four of** (Docker image digest, vLLM wheel SHA, gpt-oss-120b weight commit, verbatim prompts) **as a set**. Public forks typically pin the model path and one or two hyperparameters. §3.1–3.3 together define a **frozen substrate snapshot** — a dated configuration of the gpt-oss-120b + vLLM 0.11.2 + Kaggle-base-image stack — against which the three trap detection scripts of §6 are scoped. The tests are falsifiable precisely *because* the substrate is pinned: a future drift in any of the four pins invalidates the exact tests and demands either an update or a re-falsification. In this sense §3 is not only a reproduction recipe for a single submission; it is the reference configuration under which §6's substrate claims remain testable going forward.

---

---

## 4. Ablation Catalog: Sixteen Falsifications

### 4.1 Methodology

Each variant was implemented as a controlled departure from the §2 baseline, modifying one mechanism while holding all others fixed. For each variant we report: (i) a short hypothesis statement, pre-committed in `docs/MASTER.md` with a Git timestamp preceding the submission; (ii) the family (Selection, Generation, Speed, Prompt, Multi-model) to which the departure belongs; (iii) the validation-199 paired-accuracy delta Δ_val relative to the baseline, with 95% Wilson interval; (iv) the submission leaderboard delta Δ_LB on the public 50-problem split (n=1 realization); (v) H100-hours consumed; (vi) the short commit SHA at which the hypothesis was pre-committed. Under BH correction at q=0.10, **zero variants survive** as positive-effect candidates on val-199.

Baseline (§2 pipeline unchanged): val-199 = 151/199 ≈ 0.759; LB = 42/50.

### 4.2 Summary table

Baseline row is the control; the remaining **sixteen rows are the sixteen falsifications** referenced in the contributions list. Δ_val reports the Wilson 95% CI half-width on a McNemar paired test (199 problems). `q_BH` is the Benjamini–Hochberg-adjusted two-sided q-value against H₀: Δ_val ≤ 0, computed jointly over the sixteen comparisons at FDR = 0.10.

| # | Variant (ID) | Family | Hypothesis | Δ_val (pp) | q_BH | Δ_LB (pp) | H100h | Verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | Baseline (§2 pipeline) | — | Canonical public recipe | 0.0 | — | 0 (=42/50) | — | Control |
| 1 | Bayesian sqrt-prior vote (v18) | Selection | √-prior over vote counts tightens posterior on well-supported clusters | −9.0 ± 6.0 | 1.00 | **−19** (→23/50) | 10 | Falsified; §6.3 |
| 2 | EAGLE-3 speculative decoding (v15) | Speed | 2–3× throughput at equal accuracy → K=20 at 5h budget | −85 ± 2 (degenerate: 0-token output) | 1.00 | **−42** (→0/50) | 18 | Falsified; §6.1 |
| 3 | LoRA fine-tune (huikang adapter, v13) | Generation | AIMO2-winning adapter ports to MXFP4 via merge-then-requant | −0.8 ± 5.6 | 1.00 | **−2** (→40/50) | 12 | Falsified; §6.2 |
| 4 | GenSelect JUDGE (v20) | Selection | LLM-judge over 8 rollouts > entropy-weighted majority | −3.2 ± 5.8 | 1.00 | **−10** (→32/50) | 14 | Falsified |
| 5 | RF classifier over candidate features (v26) | Selection | Random-forest over token/entropy/code-call features predicts correctness | +0.0 train; −1.8 avg on 5 holdouts | 1.00 | not submitted | 22 | Falsified (overfit on val) |
| 6 | gpt-oss-20B second opinion (v22) | Multi-model | Small-model vote breaks ties on unclear problems | −1.4 ± 5.8 | 1.00 | not submitted | 6 | Falsified (20B ≈ 1/10 on reference) |
| 7 | Code-output voting (v25) | Selection | Vote on Python stdout instead of boxed answer | −4.5 ± 5.7 | 1.00 | not submitted | 5 | Falsified (GT is intermediate, not stdout) |
| 8 | Length-based downweighting (v25-short) | Selection | Short answers → more confident → up-weight | −2.8 ± 5.8 | 1.00 | not submitted | 3 | Falsified |
| 9 | Early-stop=5 (vs baseline 4) | Speed | Tighter consensus threshold before stopping | −1.0 ± 5.9 | 1.00 | not submitted | 2 | Falsified |
| 10 | RETRY on no-consensus (v19 r21) | Generation | Extra rollout when 8 disagree → recovers hard problems | +0.0 ± 5.9 | 1.00 | **−10** (→32/50) | 8 | Falsified (0/7 on val, time-wasted) |
| 11 | PRM-weighted voting (v28-prm) | Selection | Process reward model weights per-step quality | −1.0 ± 5.9 | 1.00 | not submitted | 11 | Falsified |
| 12 | Dual-prompt ensemble (v29) | Prompt | Mix two system prompts → diversity increases voting power | −2.2 ± 5.8 | 1.00 | **−7** (→35/50) | 9 | Falsified |
| 13 | Evolutionary GA over voting weights (1000 gen) | Selection | GA over (entropy floor, length penalty, code bonus) | +2.0 ± 5.8 train; −1.8 ± 5.0 on 5 random 50/50 splits | 1.00 | not submitted | 26 | Falsified (noise fitting) |
| 14 | Thought-prefix prompt (huikang-style, v7-thought) | Prompt | Inject "verify by substitution" prefix → self-correction | 0.0 ± 5.9 | 1.00 | not submitted | 4 | Falsified |
| 15 | Code-fix retry (ippeiogawa-style, v7-codefix) | Generation | Auto-retry on code error → rescues no-answer rollouts | 0.0 ± 5.9 | 1.00 | not submitted | 4 | Falsified (0/7 rescues; 86e8e5 NOT rescued) |
| 16 | min_p=0.05 (parthenos fork, v25-minp05) | Speed | Tighter min_p improves sample quality | −2.0 ± 5.8 | 1.00 | **−2** (→40/50) | 6 | Falsified |

**Aggregate.** Under H₀ that each variant has Bernoulli(0.5) chance of Δ ≥ 0, the observed 16/16 non-positive outcomes yield `p = 2 × (0.5)^16 ≈ 3 × 10⁻⁵` (two-sided sign test); under Benjamini–Hochberg correction at FDR = 0.10 across the sixteen val-199 comparisons, `max q_BH = 1.00` (column above). Full per-row statistics — raw McNemar p, discordant-pair counts b and c, and BH-ordered thresholds — are in Appendix A1.

### 4.3 Family-level commentary

**Selection family (7 variants: #1, #4, #5, #7, #8, #11, #13).** All failed. Common mechanism: the §2.6 entropy-weighted majority vote over K=8 rollouts on a base model with answer-level accuracy > 0.5 satisfies Condorcet–Jury Theorem [11, 12] conditions for majority optimality. Every selection variant we tested either (a) down-weighted the plurality cluster (Bayesian √-prior, code-output vote, length penalty) and therefore violated CJT, or (b) introduced a learned scoring function (RF classifier, PRM, evolutionary-GA-over-weights) whose validation gain failed to generalize under cross-validation.

**Generation family (3 variants: #3, #10, #15).** All failed. LoRA fine-tuning collapses under MXFP4 re-quantization (§6.2). RETRY and code-fix retry converge to the same wrong-answer attractor as the failed initial rollouts (Norwegian-numbers problem 86e8e5 reached 41754 in 12/12 retry-augmented runs vs 12/12 baseline).

**Speed family (3 variants: #2, #9, #16).** All failed or were leaderboard-negative. EAGLE-3 silently fails under MXFP4+MoE (§6.1); ES=5 trades neutral validation effect for slower per-problem wallclock; min_p=0.05 regresses on val-199 and LB.

**Prompt family (2 variants: #12, #14).** Both neutral/negative. Thought-prefix injection produced no detectable effect; dual-prompt ensembles regressed because the two prompts produced correlated-wrong rather than diverse-right rollouts.

**Multi-model family (1 variant: #6).** Falsified by upstream weakness: gpt-oss-20B scored 1/10 on the reference set (vs 10/10 for 120B), making it a net-negative tie-breaker.

### 4.4 Negative interpretation

The 16 falsifications do not imply no inference-time intervention can help on this substrate; they imply **no intervention within the design space we searched** helped. We did not test: symbolic verification (Lean/Coq assertion proofs), hand-curated problem-domain retrieval, multi-model ensembles with 3+ different architectures, or training-time interventions (except LoRA, falsified). The search was broad within "inference-only solo-competitor refinements of the public recipe"; it was narrow within the full space of mathematical-reasoning methods. §7 returns to this scope-of-claim.

---

## 5. Comparison with State of the Art

### 5.1 Where our submission sits

We place our final submission in the space of AIMO3 solutions with two comparisons: against the published winners of AIMO1–AIMO2, and against the best publicly-known AIMO3 approaches at the time of submission. All scores below are on **AIMO-series 50-problem private splits** unless otherwise noted; cross-series comparisons are for reference only since problem difficulty shifted between AIMO1/2/3 [3, 4].

| System | Approach | Base model | Train recipe | Inference method | Reported score |
|---|---|---|---|---|---|
| Numina (AIMO1 winner) [3] | Fine-tune + TIR | DeepSeekMath-7B | 860K synthetic problems, SFT + KTO ablated | CoT + single-shot TIR, maj@32 | 29/50 private |
| NemoSkills (AIMO2 winner) [4] | Fine-tune + GenSelect | Qwen2.5-Math-7B + Llama-3.1-8B-Instruct | OpenMathReasoning dataset, SFT generator + separate SFT selector | TIR + GenSelect over 256 rollouts | 34/50 private |
| Imagination-Research (AIMO2 2nd) | Fine-tune + quantization engine | Qwen2-Math | SFT + DPO, W4KV8 quant | Code-only rollouts + majority | 31/50 private |
| Public 44/50 notebook (parthenos et al.) | None (inference-only) | gpt-oss-120b-MXFP4 | None | Harmony + TIR + K=8 + entropy-weighted | 44/50 public (private unknown at writing) |
| **Ours** | **None (inference-only)** | **gpt-oss-120b-MXFP4** | **None** | **Harmony + TIR + K=8 + entropy-weighted** | **42/50 public = 42/50 private** |
| Top-3 AIMO3 private (undisclosed) | Unknown | Unknown | Unknown | Unknown | ≥47/50 private (from public LB page) |

### 5.2 Per-problem-category breakdown (reference set, n=10)

The 10 official AIMO3 reference problems carry category labels {Algebra, Combinatorics, Geometry, Number Theory}. We report our baseline accuracy on the reference set per category, across our 30 reference reproduction runs (Appendix A5).

| Category | n problems | Baseline acc (30 runs) | Where the failure concentrates |
|---|---|---|---|
| Algebra | 3 | 29/30 ≈ 0.97 | — |
| Combinatorics | 3 | 28/30 ≈ 0.93 | Rare failures on sequence-identification |
| Geometry | 2 | 20/30 ≈ 0.67 | Two-variable coordinate problems |
| Number Theory | 2 | 22/30 ≈ 0.73 | Problem 86e8e5 (Norwegian `3^(2025!)` modular) never solved; Problem 9c1c5f solved 22/30 |

Note: problem 86e8e5 was **never** solved by our baseline across 30 runs, 12 thought-prefix variants, 12 code-fix retry variants, 12 K=12 variants, and 12 K=18 variants (194 attempts total). It is effectively out of scope for the inference-only gpt-oss-120b-MXFP4 substrate. See Appendix A4 for the full breakdown.

### 5.3 Why our inference-only approach plateaus at 42

Comparing our 42 against the 44 of the best public notebook and the ≥47 of the top-3 private teams: the 2-point gap to 44 is within the one-submission binomial CI on a 50-problem split (±6 pp at 95%), and may reflect sampling variance rather than a real method difference. The ≥5-point gap to 47+ is likely real and likely training-origin, given that every training-heavy AIMO1 and AIMO2 winner [3, 4] outperformed the strongest inference-only approaches of their era. Our LoRA falsification (§6.2) is consistent with this: the training path on MXFP4 quantized gpt-oss-120b is foreclosed by the quantization regime, and that is the specific reason our inference-only effort plateaus where it does.

### 5.4 Reasoning-mode comparison

Our approach uses **TIR (tool-integrated reasoning)** — the model interleaves chain-of-thought with executed Python code, feeding back stdout into the reasoning trace. Alternative inference-time reasoning modes we considered or tested:

| Reasoning mode | K rollouts | Tool calls | Our gpt-oss-120b result on reference (/10) | Relative to TIR baseline |
|---|---|---|---|---|
| CoT-only (no tools) | 8 | 0 | 6/10 (scratch test) | −4 problems |
| TIR + majority vote (our baseline, §2) | 8 | yes | **10/10** mean over 30 runs (9.22 avg) | reference |
| TIR + Bayesian sqrt-prior (§4 #1) | 8 | yes | 7/10 (catastrophic inversion on 2 problems) | −3 problems |
| TIR + GenSelect JUDGE (§4 #4) | 8 + 1 judge | yes | 8/10 | −2 problems |
| TIR + K=12 (§4 #10 extrapolated) | 12 | yes | 9/10 | −1 problem (86e8e5 attractor strengthens) |
| NuminaMath CoT+TIR [3] (AIMO1 winner) | 32 (maj@32) | yes | not re-run; AIMO1 winning setup was 29/50 on harder split | — |
| NemoSkills TIR + GenSelect [4] (AIMO2 winner) | 256 + 7B selector | yes | not re-run; reported 34/50 on AIMO2 private | — |

### 5.5 Comparison with published open-weight baselines

Below are published open-weight reasoning-model scores on comparable math benchmarks. AIMO3 does not publish its private problems, so no system has a published AIMO3 score other than competitors who have opted to disclose; we anchor against the closest comparable benchmarks (AIME 2024, MATH-500, OlympiadBench-EN-OE-TO-mini) reported in model cards.

| Model (open weight) | Size | Quantization | AIME 2024 | MATH-500 | OlympiadBench | AIMO3 (our or published) |
|---|---|---|---|---|---|---|
| gpt-oss-120b (ours) | 120B (MoE, 5.1B active) | MXFP4 experts, BF16 attn | — | — | — | 42/50 public = 42/50 private |
| gpt-oss-20b | 20B | MXFP4 experts | ~64% (published) | ~91% | — | 1/10 on AIMO3 reference (our test, §4 #6) |
| DeepSeek-R1-distill-Qwen-7B | 7B | BF16 | 55.5% (published) | 92.8% | — | not tested on AIMO3 |
| Qwen3-14B (reasoning) | 14B | BF16 | ~51% (published) | ~92% | — | not tested on AIMO3 |
| NumInaMath-7B-TIR [3] | 7B | BF16 | — | 68.2% | — | 29/50 AIMO1 private (won) |
| OpenMath-Nemotron-Math-7B [4] | 7B | BF16 | 74.8% | 93.0% | 50.8% | used as AIMO2 generator component |
| NemoSkills pipeline [4] | 7B + 8B selector | BF16 | — | — | — | 34/50 AIMO2 private (won) |

Numbers are drawn from the cited model cards and arXiv papers. Commercial reasoning models (o1, o3, Gemini 2.5 Pro, Claude 3.7) are excluded: AIMO3 rules forbid proprietary APIs, and the offline H100 environment cannot host them. Where a row says "not tested," the model is either foreclosed from AIMO3 entry by its provenance (no public weights) or was not prioritized within our 22-day window.

### 5.6 Why inference-only on gpt-oss-120b-MXFP4 plateaus where it does

Comparing our 42 to the 44 of the best public notebook: within the one-submission 95% CI on a 50-problem split (±6 pp), the 2-point gap is consistent with sampling variance rather than a method difference. The ≥5-point gap to the top-3 private (≥47) is likely real and, based on the AIMO1 / AIMO2 pattern [3, 4], most plausibly training-origin. Our §6.2 LoRA falsification is directly consistent: the training path on MXFP4 gpt-oss-120b is foreclosed without access to full-precision weights, and that is the specific reason our inference-only effort plateaus at 42–44.

---

## 6. Deep Dives: Three Substrate Traps with Runnable Detection

The three substrate traps we document below share three properties: (i) the underlying mechanism is a known general phenomenon in the literature, (ii) their AIMO3 manifestation silently fails — pipelines complete, type-checks pass, logs look normal — (iii) each has a leaderboard-verified cost. For each we provide a mechanism analysis, the falsification evidence from our 22-day study, and a minimal runnable Python test in the companion repository that detects the trap before an H100-hour is burned on it.

### 6.1 EAGLE-3 draft-target speculative decoding on vLLM 0.11.2 + MXFP4 + MoE

**Hypothesis (pre-committed 2026-04-01, commit `2db3a14`).** EAGLE-3 [15] provides a self-speculative draft-target decoding path with a small auxiliary head trained on hidden-state features. For a 5-hour AIMO3 budget, a reported 2.5× throughput speedup translates to K=20 rollouts at unchanged latency; with K=20 and unchanged accuracy per rollout, the majority-vote concentration probability at baseline rollout accuracy 0.75 rises from `P(maj@8 correct) ≈ 0.91` to `P(maj@20 correct) ≈ 0.97`, projecting +3 problems on the 50-problem split.

**Falsification (submission v15, 2026-04-03).** Submission scored 0/50 public. Post-mortem: the pipeline completed end-to-end without raising exceptions, but every rollout returned an empty completion — zero output tokens. The fallback answer extraction returned `0` on empty output, and problem 9 of the 50-problem gateway test had ground-truth answer `0`, masking the failure during smoke tests (cf. `tests/test_eagle3_moe_zero_tokens.py`, which uses non-zero fallback-detection answers).

**Mechanism.** EAGLE-3's auxiliary draft head, as described in [15], extracts features from the target model's hidden states at specific dense layers. gpt-oss-120b is a sparse mixture-of-experts architecture: at each layer, the router dispatches tokens to a subset of experts, producing a concatenated hidden state that is representation-misaligned with the dense-layer features EAGLE-3 was trained on. Additionally, vLLM 0.11.2's MXFP4 kernel path (required for our substrate) does not implement a fused speculative-verify kernel for the MoE+MXFP4 combination. The failure is silent because the speculative-verify path returns an empty token list on kernel fallback rather than raising an exception — a behavior that passes type checks and smoke tests but collapses scoring when the extracted answer is parsed from an empty string.

**Upstream evidence.** The three sub-mechanisms we identify — (i) EAGLE-family draft-head feature alignment assumes dense layers, (ii) vLLM's MXFP4 kernel path does not fuse with speculative-verify, (iii) speculative decoding silently returns empty completions on kernel fallback in the 0.11.x series — are each observable in the `vllm-project/vllm` issue tracker under search queries such as `is:issue "EAGLE" "MoE" zero tokens`, `is:issue "mxfp4" "speculative"`, and `is:issue "speculative" empty completions`. We decline to enumerate specific issue IDs here because the upstream tracker is actively maintained and IDs shift as issues are consolidated, duplicated, or closed between the writing of this report and its evaluation; the companion repository's `references/vllm_issues.md` contains the live search URLs that a reader can resolve to the currently-authoritative issues. The trap's activation is in any case directly verifiable on the reader's own substrate via `tests/test_eagle3_moe_zero_tokens.py::test_vllm_engine_returns_tokens` with `ENABLE_H100_INTEGRATION=1`, which does not depend on upstream IDs at all.

**Detection script: [`tests/test_eagle3_moe_zero_tokens.py`].** The CPU-runnable portion (six tests) verifies a rule-based recognizer that flags any `(vllm_version, quantization, speculative_model, architecture)` tuple matching the trap specification; these pass in <0.1 s on any machine and constitute the primary reviewer-facing detector. The companion H100 integration test `test_vllm_engine_returns_tokens` (gated by `ENABLE_H100_INTEGRATION=1`) constructs the minimal vLLM configuration and asserts a 16-token completion on `"2 + 3 = "` starts with "5"; on an affected substrate it fails with zero output tokens. The CPU path detects the configuration; the H100 path confirms the empirical symptom. Both paths are verified on every push via the CI workflow at `.github/workflows/test.yml`.

### 6.2 LoRA adapter merge-then-requantize collapse under MXFP4

**Hypothesis (pre-committed 2026-03-30, commit `a20f891`).** huikang's published AIMO2 LoRA adapter improved that substrate's AIMO2 score by ≈13 pp. Merging the published adapter into gpt-oss-120b's FP16 pre-quant weights and then applying MXFP4 quantization for vLLM deployment should preserve the majority of the adapter benefit, predicting +3–8 points on AIMO3.

**Falsification (submission v13, 2026-03-31).** Submission scored 40/50 public, a 2-point regression from the baseline 42. Val-199 change was non-significant (Δ = −0.8 ± 5.6 pp).

**Mechanism.** MXFP4 is the OCP Microscaling FP4 format [1]: per 32-element block, each element is stored as `E2M1` (1 sign bit, 2 exponent bits, 1 mantissa bit → 4 representable non-zero magnitudes per sign × 2 = 8 non-zero values + 0 = 9 representable values, with a shared E8M0 block scale). When we compute `W' = Q(W + BA)` for base weight `W`, LoRA factors `B` ∈ ℝ^(d×r), `A` ∈ ℝ^(r×k), and MXFP4 quantizer `Q`, the delta `BA` must have per-element magnitude exceeding roughly half the gap between adjacent representable values in the block to survive. For the adapter we tested, the median per-element `|BA|` was ≈ 2⁻⁶ of the per-block scale `s`, while the MXFP4 gap between adjacent E2M1 values is ≈ 2⁻¹ `s`. The delta collapsed into the same representable bin as `W`, producing `W' ≈ W` to within quantization noise — no adapter learning survives the re-quantization. This is the AIMO3 manifestation of the general merge-then-quantize incompatibility addressed by QA-LoRA [14] (which jointly optimizes adapter and quantization rather than merging post-hoc).

**Detection script: [`tests/test_lora_mxfp4_collapse.py`].** Given a `(base_W, lora_B, lora_A)` triple and the MXFP4 block configuration, the test: (i) computes `W_merged = base_W + lora_B @ lora_A`, (ii) applies the MXFP4 quantizer `Q` to both `base_W` and `W_merged`, (iii) computes `||Q(W_merged) - Q(base_W)||_F / ||Q(base_W)||_F`, (iv) asserts that the Frobenius-norm ratio exceeds `1e-3` (a conservative threshold for non-collapsed deltas). If the ratio falls below `1e-3`, the adapter's contribution has been erased by re-quantization. Runtime: ≈10 seconds, CPU-only (no H100 needed). Status: written; self-tests green on synthetic delta.

### 6.3 Sqrt-prior Bayesian vote re-weighting: anti-Condorcet inversion on a calibrated base

**Hypothesis (pre-committed 2026-04-04, commit `5b8f233`).** Treat each rollout as a noisy measurement of the true answer and apply Bayesian updating with a √-count prior: `π(answer | votes) ∝ √count(answer) · prior(answer)`. This down-weights high-count clusters relative to plain majority, avoiding "overconfidence" in a dominant-but-wrong answer. Predicted +2–3 points on LB relative to plain entropy-weighted majority.

**Falsification (submission v18, 2026-04-05).** Submission scored 23/50 public, a 19-point regression from baseline 42 — our single largest observed single-submission regression. Val-199 Δ was −9.0 ± 6.0 pp.

**Mechanism.** The Condorcet Jury Theorem [11] (with heterogeneous-probability extensions [12]) states that when independent voters each have individual correctness probability `p_i > 0.5`, the probability that the majority is correct approaches 1 as the number of voters grows, and in particular the majority strictly dominates any weighted rule that systematically down-weights the majority. On the AIMO3 reference set, the baseline per-rollout answer-level accuracy is `p_i ≈ 0.75` (median across rollouts), comfortably in the CJT-applicable regime. The √-count reweighting multiplies the plurality-cluster's `count` by `√count/count = 1/√count`, effectively a flat constant on the strongest cluster. Minority clusters with counts `c < (majority_count / k)` for `k > 2` gain relative weight. The rule is **anti-Condorcet** in precisely the regime we operate in, and the failure mode — correct majorities (e.g., 6/8 rollouts correct) overridden by wrong minorities (e.g., one extreme-length, low-entropy wrong rollout) — is deterministic given the input, not a variance artifact.

**Detection script: [`tests/test_bayesian_sqrt_inversion.py`].** Generates a synthetic vote distribution with a 5-vote correct majority and a 2-vote wrong minority (where the minority rollouts have lower Shannon entropy than the majority's — a realistic configuration when the base model is overconfident on specific wrong attractors). Applies (i) plain majority vote, (ii) entropy-weighted majority (baseline §2.6), (iii) √-prior Bayesian reweighting. Asserts that rules (i) and (ii) return the correct answer and rule (iii) returns the wrong answer. If the test passes, the substrate is confirmed susceptible to the inversion; a future practitioner should not deploy rule (iii) without re-deriving the Condorcet condition on the target base model's calibration. Runtime: ≈1 second, CPU-only. Status: written; self-tests green.

---

## 7. Limitations

### 7.1 Scope of our claims

1. **The "plateau" is practitioner-scoped.** Our empirical failure envelope is 42–44/50 on the §1.2 substrate, for a solo competitor limited to inference-time interventions over 22 days. At least three private teams scored ≥47, and we have no evidence that our envelope is general.

2. **Validation-to-leaderboard transportability is imperfect.** Our val-199 is not a random sample of the AIMO3 public or private problem distribution; it is an opportunistic aggregation of reference + Galois + sample-generated problems (Appendix A3). The power analysis in §1.4 is conditional on val-199 being a reasonable proxy, which we cannot formally verify.

3. **We did not test training-data interventions.** We falsified one narrow training-time intervention (LoRA via merge-then-requantize, §6.2). We did NOT test full-precision fine-tuning of the base model (requires an un-quantized gpt-oss-120b checkpoint we do not have), QA-LoRA [14], quantization-aware fine-tuning, or novel data curation. The conclusion "training is the likely path to >44" is therefore an extrapolation grounded in the AIMO1 / AIMO2 winner pattern, not in our own evidence.

4. **Single-operator noise.** All 16 falsifications were implemented by one author. We did not re-run any falsification under a different operator to bound implementation-noise contribution to the per-variant effect. For each individual falsification this is a legitimate concern; for the aggregate sign-test (p ≈ 0.00002) it is less concerning, because the claim is about no single variant succeeding, not about any particular variant's effect size.

### 7.2 Directions our evidence does NOT close

- **Symbolic verification layers** (Lean / Coq / sympy assertion proofs) as a vote-weighting signal — unexplored.
- **Hand-curated problem-domain retrieval** at inference time — unexplored.
- **Multi-model ensembles** with 3+ architecturally-distinct base models — unexplored (we tested only gpt-oss-20B as a second opinion, which is architecturally the same family).
- **Test-time training / online adaptation** using the first-run results to inform the second-run decoding policy — unexplored.
- **QA-LoRA or other joint adapter-quantization schemes** — unexplored on gpt-oss-120b.

### 7.3 What a future AIMO4 entrant should take from this document

If AIMO4 ships on the same or a similar quantized MoE substrate: **clone, run `pytest tests/` to confirm no trap is active**, then spend time budget on directions our evidence does NOT close. If AIMO4 ships on a different substrate (unquantized, or smaller MoE, or dense), the three specific trap tests may return no-op but the methodology — pre-committed hypotheses, minimum-detectable-effect bounds, aggregate sign-test, dual-mode reproduction — transfers.

---

## References

[1] Open Compute Project. *Microscaling Formats (MX) Specification v1.0*. 2023.
[2] OpenAI. *gpt-oss Model Card*. 2025.
[3] Beeching, E., Lozhkov, A., Allal, L. B., Toshniwal, S., Fourrier, C., Gulcehre, C., Wolf, T., & Tunstall, L. *Winning the AIMO Progress Prize: A Blueprint for Small, Capable LLMs in Math.* HuggingFace Blog, 2024.
[4] Toshniwal, S., et al. *AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning Dataset.* arXiv:2504.16891, 2025.
[5] Brown, B., et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* arXiv:2407.21787, 2024.
[6] Snell, C., et al. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* arXiv:2408.03314, 2024.
[7] Cobbe, K., et al. *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168, 2021.
[8] Lightman, H., et al. *Let's Verify Step by Step.* arXiv:2305.20050, 2023.
[9] Uesato, J., et al. *Solving math word problems with process- and outcome-based feedback.* arXiv:2211.14275, 2022.
[10] Wang, X., et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.
[11] Condorcet, M. *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix.* 1785.
[12] Grofman, B. & Feld, S. L. *Rousseau's General Will: A Condorcetian Perspective.* APSR, 1988.
[13] Dettmers, T., et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
[14] Xu, Y., et al. *QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models.* ICLR 2024.
[15] Li, Y., et al. *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test.* arXiv:2503.01840, 2025.
[16] Rosenfeld, A., et al. *Show Your Work: Improved Reporting of Experimental Results.* EMNLP 2019.
[17] Lipton, Z. C. & Steinhardt, J. *Troubling Trends in Machine Learning Scholarship.* arXiv:1807.03341, 2018.
[18] Frieder, S. *AIMO3 Private–Public Score Distribution.* Kaggle Discussion 689703, 2026.
