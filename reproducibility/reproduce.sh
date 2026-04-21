#!/usr/bin/env bash
# AIMO3 submission reproduction script
# Companion to: "A Practitioner's Plateau", Section 3.4
#
# Usage:
#   bash reproduce.sh                    # Mode B: stochastic-faithful
#   REPRODUCE_DETERMINISTIC=1 bash reproduce.sh    # Mode A: strict bitwise
#
# Requires:
#   - NVIDIA H100 80GB
#   - Kaggle Docker image: gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9
#   - gpt-oss-120b MXFP4 weights under $GPT_OSS_PATH (defaults to /kaggle/input/...)
#   - vLLM 0.11.2 wheel from andreasbis/aimo-3-utils
#
# Output:
#   reproducibility/submission.parquet      the produced submission
#   reproducibility/run_log.txt             timestamps + per-problem log
#   reproducibility/verification_output.txt output of verify.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPRO="$ROOT/reproducibility"
cd "$REPRO"

# ---- 1. Mode selection ------------------------------------------------------

if [[ "${REPRODUCE_DETERMINISTIC:-0}" == "1" ]]; then
    MODE="strict"
    export PYTHONHASHSEED=42
    export CUBLAS_WORKSPACE_CONFIG=":4096:8"
    echo "[reproduce] Mode: STRICT (bitwise deterministic)"
else
    MODE="stochastic"
    echo "[reproduce] Mode: STOCHASTIC (matches actual submission)"
fi

# ---- 2. Environment check ---------------------------------------------------

echo "[reproduce] Checking environment..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[reproduce] ERROR: nvidia-smi not found; H100 required." >&2
    exit 2
fi
nvidia-smi -L | tee run_log.txt

if [[ -z "${GPT_OSS_PATH:-}" ]]; then
    if [[ -d "/kaggle/input/gpt-oss-120b/transformers/default/1" ]]; then
        GPT_OSS_PATH="/kaggle/input/gpt-oss-120b/transformers/default/1"
    else
        echo "[reproduce] ERROR: set GPT_OSS_PATH to gpt-oss-120b weights." >&2
        exit 3
    fi
fi
echo "[reproduce] Model path: $GPT_OSS_PATH"

# Verify weight hash (skip if on Kaggle — mount is trusted)
if [[ -f "model_weights.sha256" && ! "$GPT_OSS_PATH" =~ ^/kaggle/ ]]; then
    echo "[reproduce] Verifying weight SHA256..."
    pushd "$GPT_OSS_PATH" > /dev/null
    sha256sum --check "$REPRO/model_weights.sha256"
    popd > /dev/null
fi

# ---- 3. Launch vLLM server --------------------------------------------------

echo "[reproduce] Launching vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$GPT_OSS_PATH" \
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
    --disable-log-stats \
    > vllm_server.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# Poll for readiness
echo -n "[reproduce] Waiting for vLLM to be ready"
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo " ready (after ${i}×5s)."
        break
    fi
    echo -n "."
    sleep 5
done

# ---- 4. Run pipeline on reference.csv ---------------------------------------

echo "[reproduce] Running pipeline on reference set..."
python local_gateway.py \
    --problems "$ROOT/data/reference.csv" \
    --notebook "$ROOT/submission/notebook.py" \
    --mode "$MODE" \
    --seed ${SEED:-42} \
    --output submission.parquet \
    2>&1 | tee -a run_log.txt

# ---- 5. Verify ---------------------------------------------------------------

echo "[reproduce] Verifying..."
python verify.py submission.parquet \
    --mode "$MODE" \
    ${MODE:+--n-ref 30} \
    > verification_output.txt
cat verification_output.txt

echo "[reproduce] Done. See:"
echo "  reproducibility/submission.parquet"
echo "  reproducibility/run_log.txt"
echo "  reproducibility/verification_output.txt"
