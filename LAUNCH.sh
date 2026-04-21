#!/usr/bin/env bash
# AIMO3 Writeup — Launch script
# Runs the 4 commands you need, in the right order.
# Deadline: 2026-04-22 23:59 UTC

set -euo pipefail

echo "=========================================="
echo "AIMO3 Writeup publication launcher"
echo "=========================================="
echo ""
echo "This script runs 4 commands for you:"
echo "  1. Verify integrity (tests + sha256)"
echo "  2. git init + commit"
echo "  3. Push to GitHub (after you create the repo)"
echo "  4. Optional: push Kaggle Dataset"
echo ""
read -p "Press ENTER to continue, Ctrl+C to abort..."

# ---- Step 1: Integrity check ------------------------------------------------
echo ""
echo "[1/4] Integrity check..."
python -m pytest tests/ 2>&1 | tail -3
sha256sum --check reproducibility/artifact_sha256.txt > /dev/null && echo "  17/17 artifacts verified."

# ---- Step 2: git init + commit ----------------------------------------------
echo ""
echo "[2/4] Preparing git repository..."
if [[ -d .git ]]; then
    echo "  .git already exists — skipping init"
else
    git init
fi
git add .
git commit -m "Initial publication: AIMO3 Writeup submission (100/100 projected)" || echo "  nothing to commit"
git branch -M main

# ---- Step 3: GitHub push ----------------------------------------------------
echo ""
echo "[3/4] GitHub push"
echo "  FIRST: create empty repo at https://github.com/new"
echo "  Repo name: aimo3-writeup-public"
echo "  Visibility: PUBLIC"
echo "  Do NOT initialize with README/license (we have them)"
echo ""
read -p "Press ENTER when the empty repo is created..."

if git remote -v | grep -q origin; then
    git remote set-url origin https://github.com/SebastianGilPinzon/aimo3-writeup-public.git
else
    git remote add origin https://github.com/SebastianGilPinzon/aimo3-writeup-public.git
fi
git push -u origin main

# ---- Step 4: Kaggle Dataset (optional) --------------------------------------
echo ""
echo "[4/4] Kaggle Dataset (OPTIONAL)"
echo "  If your Kaggle CLI token is fresh, this creates the Dataset:"
echo "    kaggle datasets create -p . -r zip"
echo "  If it fails with 401, skip — GitHub is sufficient for the rubric."
read -p "  Run Kaggle Dataset creation? [y/N]: " RUN_KAGGLE

if [[ "${RUN_KAGGLE:-N}" == "y" || "${RUN_KAGGLE:-N}" == "Y" ]]; then
    cp ../aimo3-kaggle-dataset/dataset-metadata.json .
    kaggle datasets create -p . -r zip || echo "  SKIP: Kaggle upload failed (likely 401); GitHub is sufficient."
    rm -f dataset-metadata.json
else
    echo "  Skipped."
fi

# ---- Done -------------------------------------------------------------------
echo ""
echo "=========================================="
echo "DONE. Next step (manual, browser-based):"
echo ""
echo "Go to:"
echo "  https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups"
echo "Click 'New Writeup', attach submission 'AIMO3 v7 - Winner Fork - Version 12'"
echo "Title: A Practitioner's Plateau: Sixteen Falsified Modifications..."
echo "Paste: main.md body (CTRL+A, CTRL+C from that file)"
echo "Upload: 4 figures from figures/"
echo "Click: Publish"
echo ""
echo "GitHub link after push: https://github.com/SebastianGilPinzon/aimo3-writeup-public"
echo "=========================================="
