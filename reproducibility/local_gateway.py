#!/usr/bin/env python3
"""
Local replay harness for the Kaggle AIMO3 inference gateway.

Instead of the live kaggle_evaluation.aimo_3_inference_server, this harness
reads problems from a CSV, calls the same predict() function the submission
notebook exposes, and writes a submission.parquet.

Companion to writeup/reproducibility/reproduce.sh (Section 3.4).

Usage:
    python local_gateway.py \\
        --problems ../data/reference.csv \\
        --notebook ../submission/notebook.py \\
        --mode stochastic \\
        --seed 42 \\
        --output submission.parquet
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd


def load_notebook_module(notebook_path: Path):
    """Load the submission notebook as a Python module, exposing predict()."""
    spec = importlib.util.spec_from_file_location("aimo3_submission", notebook_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aimo3_submission"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "predict"):
        raise RuntimeError(
            f"Notebook {notebook_path} does not expose a predict() function. "
            "The AIMO3 gateway requires one."
        )
    return mod


def apply_mode(mode: str, seed: int) -> None:
    """Apply deterministic or stochastic seed policy."""
    if mode == "strict":
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.cuda.manual_seed_all(seed)
        except (ImportError, AttributeError, RuntimeError):
            pass
        print(f"[local_gateway] STRICT mode: seed={seed} everywhere")
    else:
        seed_now = int(time.time())
        random.seed(seed_now)
        try:
            import numpy as np
            np.random.seed(seed_now)
        except ImportError:
            pass
        print(f"[local_gateway] STOCHASTIC mode: seed={seed_now} (per-run)")


def run_gateway(problems_csv: Path, notebook_path: Path, output: Path, mode: str, seed: int) -> None:
    apply_mode(mode, seed)
    df_in = pd.read_csv(problems_csv)
    required = {"id", "problem"}
    missing = required - set(df_in.columns)
    if missing:
        raise RuntimeError(f"{problems_csv}: missing required columns {missing}")

    module = load_notebook_module(notebook_path)

    rows = []
    t_start = time.time()
    for i, row in df_in.iterrows():
        pid = str(row["id"])
        problem = str(row["problem"])
        print(f"[local_gateway] [{i+1}/{len(df_in)}] solving {pid} ...", flush=True)
        t0 = time.time()
        try:
            answer = int(module.predict(problem_id=pid, problem=problem))
        except TypeError:
            # Some notebooks use a single positional or slightly different signature.
            try:
                answer = int(module.predict(problem))
            except Exception as e:
                print(f"[local_gateway] predict() failed for {pid}: {e}")
                answer = 0
        except Exception as e:
            print(f"[local_gateway] predict() failed for {pid}: {e}")
            answer = 0
        dt = time.time() - t0
        print(f"[local_gateway]   answer={answer}  ({dt:.1f}s)")
        rows.append({"id": pid, "answer": answer})

    out = pd.DataFrame(rows)
    out.to_parquet(output, index=False)
    elapsed = time.time() - t_start
    print(f"[local_gateway] wrote {output} ({len(out)} rows, {elapsed:.1f}s total)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problems", type=Path, required=True, help="input CSV with id, problem columns")
    p.add_argument("--notebook", type=Path, required=True, help="submission notebook.py")
    p.add_argument("--output", type=Path, required=True, help="output submission.parquet")
    p.add_argument("--mode", choices=["strict", "stochastic"], default="stochastic")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_gateway(args.problems, args.notebook, args.output, args.mode, args.seed)


if __name__ == "__main__":
    main()
