"""
Generate all figures for the AIMO3 Writeup (Section 4/5/6).

Writes PNG files to writeup/figures/ that are referenced in the main paper.

Usage:
    python generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Figure 1: Ablation forest plot
# -----------------------------------------------------------------------------

def fig_ablation_forest() -> None:
    """
    Forest plot of the 16 falsified variants (§4.2). Shows val-199 effect
    with 95% CI per variant, colored by family, with BH-adjusted q-value.
    Clearly communicates: every variant is at or below zero effect.
    """
    # Data from §4.2 summary table
    variants = [
        # (id, name, family, delta_val_pp, ci_pp, delta_lb, q_adj)
        (17, "min_p=0.05 (v25-minp05)",       "Speed",       -2.0, 5.8, -2,  1.00),
        (16, "Code-fix retry (v7-codefix)",    "Generation",   0.0, 5.9,  None, 1.00),
        (15, "Thought-prefix (v7-thought)",    "Prompt",       0.0, 5.9,  None, 1.00),
        (14, "Evolutionary weight GA (1000)",  "Selection",    2.0, 5.8,  None, 1.00),
        (13, "Dual-prompt ensemble (v29)",     "Prompt",      -2.2, 5.8, -7,  1.00),
        (12, "PRM-weighted vote (v28-prm)",    "Selection",   -1.0, 5.9,  None, 1.00),
        (11, "RETRY on no-consensus (v19)",    "Generation",   0.0, 5.9, -10, 1.00),
        (10, "Early-stop=5",                    "Speed",       -1.0, 5.9,  None, 1.00),
        (9,  "Length downweight",               "Selection",   -2.8, 5.8,  None, 1.00),
        (8,  "Code-output vote (v25)",          "Selection",   -4.5, 5.7,  None, 1.00),
        (7,  "gpt-oss-20B second opinion (v22)","Multi-model", -1.4, 5.8,  None, 1.00),
        (6,  "RF classifier (v26)",             "Selection",    0.0, 5.9,  None, 1.00),  # overfit
        (5,  "GenSelect JUDGE (v20)",           "Selection",   -3.2, 5.8, -20, 1.00),
        (4,  "LoRA huikang (v13)",              "Generation",  -0.8, 5.6, -4,  1.00),
        (3,  "EAGLE-3 (v15)",                   "Speed",      -85.0, 2.0, -84, 0.01),  # degenerate
        (2,  "Bayesian sqrt-prior (v18)",       "Selection",   -9.0, 6.0, -38, 0.18),
    ]
    # Filter EAGLE for display scale; mark it specially
    display_variants = variants
    family_colors = {
        "Selection":  "#377eb8",
        "Generation": "#e41a1c",
        "Speed":      "#984ea3",
        "Prompt":     "#ff7f00",
        "Multi-model":"#4daf4a",
    }

    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(display_variants))
    for i, (vid, name, fam, d, ci, lb, q) in enumerate(display_variants):
        color = family_colors[fam]
        # Clip EAGLE-3's extreme Δ for display on the same axis
        d_display = max(d, -15)
        ci_display = ci if d > -15 else 1.0
        ax.errorbar(
            d_display, i,
            xerr=ci_display,
            fmt="o", color=color, ecolor=color,
            elinewidth=1.5, capsize=4, markersize=8,
            markeredgecolor="black", markeredgewidth=0.5,
        )
        if d <= -15:
            ax.annotate(
                f"(Δ = {d:.0f})", xy=(d_display, i),
                xytext=(-10, 0), textcoords="offset points",
                fontsize=8, ha="right", va="center", color=color,
            )
        if lb is not None:
            ax.annotate(
                f"LB {'+' if lb > 0 else ''}{lb}",
                xy=(d_display + ci_display + 0.5, i),
                fontsize=8, ha="left", va="center",
                color="black", alpha=0.7,
            )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{v[0]:>2d}. {v[1]}" for v in display_variants], fontsize=9)
    ax.set_xlabel("Δ validation-199 accuracy (pp) ± 95% Wilson CI", fontsize=11)
    ax.set_title(
        "Figure 1. Falsification forest: sixteen controlled variants vs. baseline (42/50).\n"
        "Zero variants exceed baseline at BH q = 0.10. Label at right shows leaderboard Δ where submitted.",
        fontsize=11, pad=15,
    )
    ax.set_xlim(-17, 10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Family legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, edgecolor="black", label=f, linewidth=0.5)
        for f, c in family_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, frameon=True)

    plt.tight_layout()
    out = HERE / "fig1_ablation_forest.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# -----------------------------------------------------------------------------
# Figure 2: MXFP4 delta-magnitude vs collapse threshold
# -----------------------------------------------------------------------------

def fig_mxfp4_collapse_boundary() -> None:
    """
    Visualize the MXFP4 quantization noise floor vs LoRA adapter delta
    magnitude. Shows at what delta magnitude the adapter is erased.
    """
    import sys
    sys.path.insert(0, str(HERE.parent / "tests"))
    from test_lora_mxfp4_collapse import detect_lora_mxfp4_collapse

    import math
    rng = np.random.default_rng(seed=42)
    d, k, r = 512, 512, 32
    base_W = rng.standard_normal((d, k)).astype(np.float32)

    delta_stds = np.logspace(-3, 0, 20)  # from 0.001 to 1.0
    snrs = []
    ratios = []
    collapsed_flags = []
    for dstd in delta_stds:
        factor_scale = float(dstd) / math.sqrt(r)
        lora_B = rng.standard_normal((d, r)).astype(np.float32) * factor_scale
        lora_A = rng.standard_normal((r, k)).astype(np.float32) * factor_scale
        v = detect_lora_mxfp4_collapse(base_W, lora_B, lora_A)
        # Re-compute SNR for plotting
        delta = lora_B @ lora_A
        merged = base_W + delta
        from test_lora_mxfp4_collapse import quantize_mxfp4
        Qb = quantize_mxfp4(base_W)
        Qm = quantize_mxfp4(merged)
        noise = np.linalg.norm(Qb - base_W)
        signal = np.linalg.norm(Qm - Qb)
        snr = signal / max(noise, 1e-12)
        snrs.append(float(snr))
        ratios.append(float(v.frobenius_ratio))
        collapsed_flags.append(v.collapsed)

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    # SNR
    colors = ["#e41a1c" if c else "#377eb8" for c in collapsed_flags]
    ax1.scatter(delta_stds, snrs, c=colors, s=60, edgecolor="black", linewidth=0.5, zorder=3)
    ax1.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="SNR = 1 (noise floor)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Per-element LoRA delta magnitude (std, relative to base)", fontsize=11)
    ax1.set_ylabel("Signal-to-noise ratio after MXFP4 re-quantization", fontsize=11)
    ax1.set_title(
        "Figure 2. MXFP4 re-quantization collapse boundary for LoRA adapters.\n"
        "Red = delta erased by quantization noise; blue = delta survives. Boundary at delta std ≈ 0.02.",
        fontsize=11, pad=15,
    )
    ax1.grid(True, which="both", alpha=0.3)

    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor="#e41a1c", edgecolor="black", label="Collapsed (SNR ≤ 1)"),
        Patch(facecolor="#377eb8", edgecolor="black", label="Survives (SNR > 1)"),
    ], loc="upper left", fontsize=9)

    plt.tight_layout()
    out = HERE / "fig2_mxfp4_collapse_boundary.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# -----------------------------------------------------------------------------
# Figure 3: Submission timeline
# -----------------------------------------------------------------------------

def fig_submission_timeline() -> None:
    """Timeline of submitted leaderboard scores across the 22 days.

    Labels are laid out with alternating above/below and an explicit
    per-point text offset to avoid overlap in the dense day-16 to day-20
    window (flagged by editor review)."""
    # (day_offset, version, public_lb, text_x_off, text_y_off)
    submissions = [
        (5,  "v7 (baseline)",    42,  0,  14),
        (10, "v15 (EAGLE-3)",     0,  0, -22),
        (12, "v18 (Bayesian)",   23,  0, -22),
        (14, "v19 (pipeline)",   32,  0,  14),
        (16, "v20 (GenSelect)",  32, -22,  24),
        (17, "v22 (FT huikang)", 40, 0,  -28),
        (18, "v25 (min_p=.05)",  40,  22, 14),
        (19, "v29 (dual+K=12)",  35,  0, -22),
        (20, "v7-lb-resubmit",   41, -10,  26),
        (22, "v7 (final, 42/42)", 42, 0, -26),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    days = [s[0] for s in submissions]
    scores = [s[2] for s in submissions]
    colors = ["#4daf4a" if s >= 42 else ("#e41a1c" if s < 30 else "#ff7f00") for s in scores]

    ax.plot(days, scores, color="gray", alpha=0.4, zorder=1)
    ax.scatter(days, scores, c=colors, s=140, edgecolor="black", linewidth=1, zorder=3)

    for d, sc, l, xoff, yoff in [(s[0], s[2], s[1], s[3], s[4]) for s in submissions]:
        ax.annotate(
            f"{l}\n({sc}/50)",
            xy=(d, sc),
            xytext=(xoff, yoff),
            textcoords="offset points",
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5) if abs(xoff) > 5 or abs(yoff) > 25 else None,
        )

    ax.axhline(42, color="#4daf4a", linestyle=":", alpha=0.6, label="Baseline 42 (our floor, public consensus)")
    ax.axhline(44, color="blue", linestyle=":", alpha=0.5, label="Best public notebook ~44")
    ax.axhline(47, color="red", linestyle=":", alpha=0.4, label="Top-3 private (unpublished) ≥47")
    ax.set_xlabel("Day of 22-day study", fontsize=11)
    ax.set_ylabel("Public leaderboard score (/50)", fontsize=11)
    ax.set_title(
        "Figure 3. Submission timeline: baseline is optimal; all departures regress.\n"
        "Final submission is byte-identical to day-5 baseline; private = public = 42.",
        fontsize=11, pad=15,
    )
    ax.set_ylim(-5, 52)
    ax.set_xlim(2, 25)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = HERE / "fig3_submission_timeline.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# -----------------------------------------------------------------------------
# Figure 4: Per-category accuracy breakdown
# -----------------------------------------------------------------------------

def fig_category_breakdown() -> None:
    """Per-category accuracy on reference set (§5.2)."""
    categories = ["Algebra", "Combinatorics", "Geometry", "Number Theory"]
    accuracies = [29/30, 28/30, 20/30, 22/30]
    n_problems = [3, 3, 2, 2]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c"]
    bars = ax.bar(categories, accuracies, color=colors, edgecolor="black", linewidth=0.8)

    for bar, acc, n in zip(bars, accuracies, n_problems):
        ax.annotate(
            f"{acc:.0%}\n(n={n})",
            xy=(bar.get_x() + bar.get_width() / 2, acc),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )

    ax.axhline(sum(accuracies) / len(accuracies), color="black", linestyle="--", alpha=0.5,
               label=f"Overall reference mean {sum(accuracies) / len(accuracies):.0%}")
    ax.set_ylabel("Baseline accuracy (30 reference runs)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Figure 4. Baseline accuracy by problem category (reference set, n=30 runs/problem).\n"
        "Geometry and Number Theory concentrate the remaining errors; 86e8e5 never solved.",
        fontsize=11, pad=15,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = HERE / "fig4_category_breakdown.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")


# -----------------------------------------------------------------------------

def main() -> None:
    fig_ablation_forest()
    fig_mxfp4_collapse_boundary()
    fig_submission_timeline()
    fig_category_breakdown()
    print("[figures] All 4 figures written to", HERE)


if __name__ == "__main__":
    main()
