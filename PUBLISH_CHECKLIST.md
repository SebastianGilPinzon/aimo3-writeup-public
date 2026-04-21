# Publishing Checklist — AIMO3 Writeup Prize

**Deadline:** 2026-04-22 23:59 UTC (≈5 days remaining as of 2026-04-17).
**Author task list to get from THIS package to the published Kaggle Writeup.**

---

## Phase 1: Pre-publish — build the public companion repo (est. 30 min)

Our writeup promises a GitHub repo + Kaggle Dataset at these URLs:
- GitHub: `https://github.com/SebastianGilPinzon/aimo3-writeup-public`
- Kaggle Dataset: `https://www.kaggle.com/datasets/sebastiangil00/aimo3-writeup-artifacts`

Neither exists yet. The reviewers flagged this as the SINGLE biggest remaining risk ("vaporware artifacts"). They must exist before 2026-04-22.

### Step 1.1 — Create the GitHub repo

```bash
# From the parent project directory
cd c:/Users/juans/Documents/GitHub/kaggle-aimo3-2026-1

# Create a clean export of just the writeup/ directory + relevant data
mkdir -p /tmp/aimo3-writeup-public
cp -r writeup/* /tmp/aimo3-writeup-public/

# Verify nothing secret goes in
grep -rE "kaggle\.json|KGAT_|KAGGLE_API_TOKEN" /tmp/aimo3-writeup-public/ \
    && { echo "SECRETS FOUND — ABORT"; exit 1; } \
    || echo "No secrets detected."

# Push to GitHub (create empty repo first at https://github.com/new)
cd /tmp/aimo3-writeup-public
git init
git add .
git commit -m "Initial publication: AIMO3 writeup submission"
git remote add origin https://github.com/SebastianGilPinzon/aimo3-writeup-public.git
git branch -M main
git push -u origin main
```

### Step 1.2 — Create the Kaggle Dataset

```bash
cd /tmp/aimo3-writeup-public
cat > dataset-metadata.json << 'EOF'
{
  "title": "AIMO3 Writeup Artifacts",
  "id": "sebastiangil00/aimo3-writeup-artifacts",
  "licenses": [{"name": "CC-BY-4.0"}],
  "isPrivate": false,
  "keywords": ["aimo", "aimo3", "math-olympiad", "gpt-oss", "mxfp4", "reproducibility"]
}
EOF
kaggle datasets create -p . -r zip
```

(If the Kaggle API token is expired, renew at https://www.kaggle.com/settings before running.)

### Step 1.3 — Run strict mode ONCE to populate strict_mode_sha256.txt

On the H100 host with the pinned Docker image:

```bash
cd /tmp/aimo3-writeup-public
REPRODUCE_DETERMINISTIC=1 bash reproducibility/reproduce.sh
sha256sum submission.parquet > reproducibility/strict_mode_sha256.txt
# Rename the hash line to match the name verify.py expects:
sed -i 's/submission.parquet$/submission.parquet.strict_mode/' reproducibility/strict_mode_sha256.txt
git add reproducibility/strict_mode_sha256.txt
git commit -m "populate strict-mode SHA256 from canonical reference host"
git push
```

If H100 is unavailable before the deadline, the stochastic mode alone is still functional; strict mode's placeholder will be honestly documented as pending.

---

## Phase 2: Publish as Kaggle Writeup (est. 15 min)

### Step 2.1 — Navigate and create

1. Go to https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups
2. Click **"New Writeup"**.
3. Under **"Link to submission"**, select **"AIMO3 v7 - Winner Fork - Version 12"** (the submission with 42/42 selected for private LB).
4. **Title:** `A Practitioner's Plateau: Sixteen Falsified Modifications to the AIMO3 Public-Consensus Pipeline on gpt-oss-120b-MXFP4`

### Step 2.2 — Paste the main paper

1. Open `writeup/main.md`.
2. Copy the entire content (skip this PUBLISH_CHECKLIST file — it's not part of the paper).
3. Paste into the Kaggle Writeup markdown editor.
4. **Verify renders:**
   - Tables render properly (Kaggle supports GFM)
   - Code blocks render with syntax
   - LaTeX math renders (Kaggle supports `$..$` and `$$..$$` — adjust if needed)
5. Preview at least twice before publishing.

### Step 2.3 — Add figures

For each of 4 figures in `writeup/figures/`:
1. Click the image button in the Kaggle editor.
2. Upload `fig1_ablation_forest.png` (first), then `fig2`, `fig3`, `fig4`.
3. Position each figure at the `*Figure N: ...*` reference point in the main paper.

### Step 2.4 — Link companion artifacts

At the end of the main paper (after References), add:

> **Companion artifacts (linked from this writeup):**
> - GitHub repository: https://github.com/SebastianGilPinzon/aimo3-writeup-public
> - Kaggle Dataset: https://www.kaggle.com/datasets/sebastiangil00/aimo3-writeup-artifacts
> - Appendix (A1–A7): https://github.com/SebastianGilPinzon/aimo3-writeup-public/blob/main/appendix.md
> - Runnable detection tests: https://github.com/SebastianGilPinzon/aimo3-writeup-public/tree/main/tests

### Step 2.5 — Publish

1. Click **Publish** at the top-right.
2. Verify the writeup appears at `https://www.kaggle.com/competitions/.../writeups/<slug>`.
3. Verify the submission link is correctly attached.
4. Take a screenshot of the published page as evidence (for your records).

---

## Phase 3: Post-publish (optional, time-permitting)

- Post a link to the writeup in the AIMO3 discussion forum under a relevant thread, per community conventions.
- HuggingFace Community Blog cross-post (uses the same markdown; ~20 min of work).
- Monitor the writeup for reviewer comments; respond within 24 h if the host opens a rebuttal window.

---

## Pre-publish final verification (3-minute checklist — run RIGHT BEFORE clicking Publish)

- [ ] `pytest writeup/tests/` passes (15 passed, 2 skipped)
- [ ] All 4 figures open cleanly in Kaggle editor preview
- [ ] GitHub repo is public and reachable from an incognito browser tab
- [ ] Kaggle Dataset is public and appears in search
- [ ] Main paper has no "TODO", "FIXME", or "placeholder" text
- [ ] `grep -rE "TODO|FIXME|XXX" writeup/main.md writeup/appendix.md` returns nothing critical
- [ ] Submission link at top of writeup resolves to the 42/42 selected submission
- [ ] Author byline is correct (Juan Sebastian Gil Pinzon / sebastiangil00)
- [ ] License block (CC-BY 4.0) is present at the end of main paper

---

## Risk register (what could still go wrong after publish)

| Risk | Mitigation |
|---|---|
| A judge runs `bash reproduce.sh` and hits a missing dependency | The pinned Docker image should supply everything; `pip_freeze.lock` documents fallback |
| A judge runs `pytest` without numpy installed | README.md instructs `pip install pytest numpy` first |
| GitHub repo 404 at review time | Verified before publish; renewed via cron if needed |
| Kaggle Writeup markdown rendering differs from local | Preview twice before publish; known issue with certain LaTeX constructs |
| Reviewer objects to strict-mode SHA being unpopulated | Documented as pending; stochastic mode PASSES without it |
| Reviewer objects to appendix being linked not inline | Acceptable per AIMO3 rules (30k total, 5k main, appendix in "remaining sections") |

---

## Success criteria

- **Primary:** Writeup accepted and published on Kaggle before 2026-04-22 23:59 UTC.
- **Stretch:** Selected as one of the two $15K Writeup Prize winners.
- **Baseline (acceptable if primary missed):** Honorable mention in the writeup track.

If the prize is won, cross-posting to arXiv and HuggingFace Community Blog is
the next phase (est. 1-2 days work, no deadline).
