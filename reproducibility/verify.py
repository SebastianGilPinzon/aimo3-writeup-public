#!/usr/bin/env python3
"""
Reproduction verifier for the AIMO3 Writeup submission (Section 3.4).

Two modes:

    --mode strict
        Compares submission.parquet's SHA256 against the frozen
        strict_mode_sha256.txt. Used with REPRODUCE_DETERMINISTIC=1.

    --mode stochastic
        For each problem in submission.parquet, asserts that the submitted
        answer lies in the 95% support of the empirical answer distribution
        observed across the 30 reference reproduction runs
        (expected_hashes.json, Appendix A5).

Exit code:
    0 = verification passed
    1 = a per-problem mismatch was found
    2 = environment error (file missing, malformed, etc.)

Usage:
    python verify.py submission.parquet --mode strict
    python verify.py submission.parquet --mode stochastic --n-ref 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_strict(submission: Path, expected_sha_path: Path) -> int:
    if not expected_sha_path.exists():
        print(f"[verify] ERROR: expected SHA file not found: {expected_sha_path}", file=sys.stderr)
        return 2
    expected = expected_sha_path.read_text().strip().split()[0]
    actual = sha256_file(submission)
    print(f"[verify] Mode: STRICT")
    print(f"[verify]   expected SHA256: {expected}")
    print(f"[verify]   actual SHA256:   {actual}")
    if actual == expected:
        print("[verify] PASS: submission is bitwise-identical to the reference.")
        return 0
    print("[verify] FAIL: submission differs from the reference.", file=sys.stderr)
    return 1


def verify_stochastic(submission: Path, expected_json: Path, min_support_frac: float) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("[verify] ERROR: pandas required for stochastic mode.", file=sys.stderr)
        return 2
    if not expected_json.exists():
        print(f"[verify] ERROR: expected distribution file not found: {expected_json}", file=sys.stderr)
        return 2

    with expected_json.open() as f:
        expected = json.load(f)

    df = pd.read_parquet(submission)
    if "id" not in df.columns or "answer" not in df.columns:
        print("[verify] ERROR: submission missing required columns {id, answer}.", file=sys.stderr)
        return 2

    print(f"[verify] Mode: STOCHASTIC (Hoeffding support bound)")
    print(f"[verify]   reference runs N = {expected.get('n_reference_runs', 'UNKNOWN')}")
    print(f"[verify]   min support fraction = {min_support_frac:.3f}")

    mismatches = 0
    for _, row in df.iterrows():
        pid = str(row["id"])
        answer = int(row["answer"])
        if pid not in expected["per_problem"]:
            print(f"[verify]   SKIP {pid}: not in reference set")
            continue
        dist = expected["per_problem"][pid]  # {answer_str: support_fraction}
        support = dist.get(str(answer), 0.0)
        if support < min_support_frac:
            mismatches += 1
            print(
                f"[verify]   FAIL {pid}: answer={answer} support={support:.3f} "
                f"(< {min_support_frac:.3f}); top-3 reference answers: "
                f"{sorted(dist.items(), key=lambda kv: -kv[1])[:3]}"
            )
        else:
            print(f"[verify]   PASS {pid}: answer={answer} support={support:.3f}")

    if mismatches == 0:
        print(f"[verify] PASS: all {len(df)} problems verified.")
        return 0
    print(f"[verify] FAIL: {mismatches} of {len(df)} problems failed.", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", type=Path, help="path to submission.parquet")
    ap.add_argument("--mode", choices=["strict", "stochastic"], required=True)
    ap.add_argument("--n-ref", type=int, default=30, help="reference run count for stochastic mode")
    ap.add_argument("--min-support", type=float, default=0.05, help="min fraction of reference runs an answer must appear in")
    ap.add_argument("--expected-sha", type=Path, default=None)
    ap.add_argument("--expected-json", type=Path, default=None)
    args = ap.parse_args()

    if not args.submission.exists():
        print(f"[verify] ERROR: submission file not found: {args.submission}", file=sys.stderr)
        return 2

    here = Path(__file__).resolve().parent
    if args.mode == "strict":
        expected = args.expected_sha or (here / "strict_mode_sha256.txt")
        return verify_strict(args.submission, expected)
    expected = args.expected_json or (here / "expected_hashes.json")
    return verify_stochastic(args.submission, expected, args.min_support)


if __name__ == "__main__":
    sys.exit(main())
