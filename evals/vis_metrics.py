"""
plot_sampling_expts.py

Usage:
    python plot_sampling_expts.py <results_dir>

<results_dir> must contain subdirectories named low_only, medium_only, high_only,
each containing run folders named like:
    synllama_1b_2m_<model_tag>_on_<testset_tag>/

Reads:
  - combined_final_stats.csv         → failure rate, recon rate, similarities, avg steps
  - logs/diversity_eval.log          → product & BB diversity at each threshold
"""

import os
import re
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── helpers ──────────────────────────────────────────────────────────────────

SAMPLING_MODES = ["low_only", "medium_only", "high_only", "greedy"]
SAMPLING_LABELS = {"low_only": "Low", "medium_only": "Medium", "high_only": "High", "greedy": "Greedy"}
DIVERSITY_THRESH = 0.8


def infer_tags(run_name: str):
    """
    Extract model_tag and testset_tag from a run folder name like:
        synllama_1b_2m_91rxns_on_1k_chembl
    Returns (model_tag, testset_tag) or (None, None) on failure.
    """
    m = re.match(r"synllama_1b_2m_(.+?)_on_(.+)", run_name)
    if m:
        return m.group(1), m.group(2)
    return None, None


def parse_combined_stats(csv_path: str) -> dict:
    """Parse combined_final_stats.csv → dict of scalar metrics."""
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    return {
        "failure_rate": float(row["total_failure_rate %"]),
        "recon_rate": float(row["total_enamine_reconstruct_rate %"]),
        "morgan_sim": float(row["morgan_sim"]),
        "scf_sim": float(row["scf_sim"]),
        "pharm2d_sim": float(row["pharm2d_sim"]),
        "avg_rxn_steps": float(row["avg_rxn_steps"]),
    }


def parse_diversity_log(log_path: str, target_thresh: float = DIVERSITY_THRESH) -> dict:
    """
    Parse diversity_eval.log.
    Returns dict with keys:
        recon_rate_from_log,
        product_diversity_<thresh>, bb_diversity_<thresh>
        for the target threshold (and all thresholds stored as lists).
    """
    result = {}
    thresholds, prod_divs, bb_divs = [], [], []
    cur_thresh = None

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"={3,} Analogue Threshold ([0-9.]+)", line)
            if m:
                cur_thresh = float(m.group(1))
                continue
            if cur_thresh is not None:
                mp = re.match(r"Mean product diversity:\s*([0-9.]+)", line)
                mb = re.match(r"Mean BB diversity:\s*([0-9.]+)", line)
                mr = re.match(r"Reconstruction rate:\s*\d+/\d+\s*=\s*([0-9.]+)", line)
                if mp:
                    thresholds.append(cur_thresh)
                    prod_divs.append(float(mp.group(1)))
                if mb:
                    bb_divs.append(float(mb.group(1)))

    result["thresholds"] = thresholds
    result["prod_divs"] = prod_divs
    result["bb_divs"] = bb_divs

    # Extract values at the target threshold
    for i, t in enumerate(thresholds):
        if abs(t - target_thresh) < 1e-6:
            result[f"prod_div_{target_thresh}"] = prod_divs[i]
            result[f"bb_div_{target_thresh}"] = bb_divs[i] if i < len(bb_divs) else float("nan")
            break

    return result


def collect_all_data(base_dir: str) -> pd.DataFrame:
    """Walk the directory tree and collect all metrics into a DataFrame."""
    records = []

    for mode in SAMPLING_MODES:
        mode_dir = os.path.join(base_dir, mode)
        if not os.path.isdir(mode_dir):
            print(f"[warn] Missing directory: {mode_dir}")
            continue

        for run_name in sorted(os.listdir(mode_dir)):
            run_dir = os.path.join(mode_dir, run_name)
            if not os.path.isdir(run_dir):
                continue

            model_tag, testset_tag = infer_tags(run_name)
            if model_tag is None:
                print(f"[warn] Cannot parse run name: {run_name}")
                continue

            rec = {
                "mode": mode,
                "model": model_tag,
                "testset": testset_tag,
                "run_name": run_name,
            }

            # combined_final_stats.csv
            csv_path = os.path.join(run_dir, "combined_final_stats.csv")
            if os.path.isfile(csv_path):
                rec.update(parse_combined_stats(csv_path))
            else:
                print(f"[warn] Missing: {csv_path}")

            # diversity_eval.log
            log_path = os.path.join(run_dir, "logs", "diversity_eval.log")
            if os.path.isfile(log_path):
                div = parse_diversity_log(log_path, DIVERSITY_THRESH)
                rec["thresholds"] = div.get("thresholds", [])
                rec["prod_divs"] = div.get("prod_divs", [])
                rec["bb_divs"] = div.get("bb_divs", [])
                rec[f"prod_div_{DIVERSITY_THRESH}"] = div.get(f"prod_div_{DIVERSITY_THRESH}", float("nan"))
                rec[f"bb_div_{DIVERSITY_THRESH}"] = div.get(f"bb_div_{DIVERSITY_THRESH}", float("nan"))
            else:
                print(f"[warn] Missing: {log_path}")

            records.append(rec)

    return pd.DataFrame(records)


# ─── plotting ─────────────────────────────────────────────────────────────────

# Color palette per sampling mode
MODE_COLORS = {
    "low_only": "#4C72B0",
    "medium_only": "#DD8452",
    "high_only": "#55A868",
    "greedy": "#A855A8",
}
# Marker per testset
TESTSET_MARKERS = {}  # filled dynamically


def get_marker(testset: str) -> str:
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    if testset not in TESTSET_MARKERS:
        TESTSET_MARKERS[testset] = markers[len(TESTSET_MARKERS) % len(markers)]
    return TESTSET_MARKERS[testset]


def grouped_bar(ax, groups, values_dict, ylabel, title, colors=None):
    """
    groups: list of x-axis labels
    values_dict: {series_label: [value_per_group]}
    """
    n_groups = len(groups)
    n_series = len(values_dict)
    width = 0.7 / n_series
    x = np.arange(n_groups)

    for i, (label, vals) in enumerate(values_dict.items()):
        offset = (i - n_series / 2 + 0.5) * width
        color = colors[label] if colors else None
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                # Choose precision based on magnitude: large values (rates %) use 1dp,
                # small decimals (diversity ~0.0x) use 3dp so labels match bar heights
                fmt = f"{v:.1f}" if v >= 1.0 else f"{v:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def plot_diversity_curves(df: pd.DataFrame, testsets: list, out_dir: str):
    """One figure per testset: product & BB diversity vs threshold curves."""
    for testset in testsets:
        sub = df[df["testset"] == testset].copy()
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(f"Diversity vs Threshold — {testset}", fontsize=12, fontweight="bold")

        for _, row in sub.iterrows():
            mode = row["mode"]
            label = SAMPLING_LABELS[mode]
            color = MODE_COLORS[mode]
            thresholds = row.get("thresholds", [])
            prod_divs = row.get("prod_divs", [])
            bb_divs = row.get("bb_divs", [])
            if not thresholds:
                continue
            axes[0].plot(thresholds, prod_divs, marker="o", label=label, color=color, lw=2)
            axes[1].plot(thresholds, bb_divs, marker="s", label=label, color=color, lw=2)

        for ax, metric in zip(axes, ["Mean Product Diversity", "Mean BB Diversity"]):
            ax.axvline(DIVERSITY_THRESH, color="gray", linestyle="--", lw=1, label=f"thresh={DIVERSITY_THRESH}")
            ax.set_xlabel("Analogue Similarity Threshold")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.legend(fontsize=9)
            ax.grid(linestyle="--", alpha=0.4)

        plt.tight_layout()
        fname = os.path.join(out_dir, f"diversity_curves_{testset}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")


def make_plots(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    testsets = sorted(df["testset"].unique())
    models = sorted(df["model"].unique())

    # ── Figure 1: Reconstruction rate & Failure rate ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Reconstruction & Failure Rates", fontsize=13, fontweight="bold")

    for ax, metric, label in zip(
        axes, ["recon_rate", "failure_rate"], ["Enamine Reconstruction Rate (%)", "Failure Rate (%)"]
    ):
        groups = []
        values_by_mode = {m: [] for m in SAMPLING_MODES}

        for ts in testsets:
            for model in models:
                lbl = f"{model}\n{ts}"
                groups.append(lbl)
                for mode in SAMPLING_MODES:
                    row = df[(df["testset"] == ts) & (df["model"] == model) & (df["mode"] == mode)]
                    v = float(row[metric].values[0]) if not row.empty else float("nan")
                    values_by_mode[mode].append(v)

        series = {SAMPLING_LABELS[m]: values_by_mode[m] for m in SAMPLING_MODES}
        colors = {SAMPLING_LABELS[m]: MODE_COLORS[m] for m in SAMPLING_MODES}
        grouped_bar(ax, groups, series, label, label, colors)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recon_failure_rates.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(out_dir, 'recon_failure_rates.png')}")

    # ── Figure 2: Product & BB diversity @ thresh=0.8 ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Diversity at Threshold = {DIVERSITY_THRESH}", fontsize=13, fontweight="bold")

    for ax, metric, label in zip(
        axes,
        [f"prod_div_{DIVERSITY_THRESH}", f"bb_div_{DIVERSITY_THRESH}"],
        [f"Mean Product Diversity (thresh={DIVERSITY_THRESH})", f"Mean BB Diversity (thresh={DIVERSITY_THRESH})"],
    ):
        groups = []
        values_by_mode = {m: [] for m in SAMPLING_MODES}

        for ts in testsets:
            for model in models:
                lbl = f"{model}\n{ts}"
                groups.append(lbl)
                for mode in SAMPLING_MODES:
                    row = df[(df["testset"] == ts) & (df["model"] == model) & (df["mode"] == mode)]
                    v = float(row[metric].values[0]) if not row.empty and metric in row.columns else float("nan")
                    values_by_mode[mode].append(v)

        series = {SAMPLING_LABELS[m]: values_by_mode[m] for m in SAMPLING_MODES}
        colors = {SAMPLING_LABELS[m]: MODE_COLORS[m] for m in SAMPLING_MODES}
        grouped_bar(ax, groups, series, label, label, colors)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diversity_at_thresh.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(out_dir, 'diversity_at_thresh.png')}")

    # ── Figure 3: Diversity curves per testset ───────────────────────────────
    plot_diversity_curves(df, testsets, out_dir)

    # ── Figure 4: Summary heatmap — one per testset ──────────────────────────
    scalar_cols = [
        "failure_rate",
        "recon_rate",
        f"prod_div_{DIVERSITY_THRESH}",
        f"bb_div_{DIVERSITY_THRESH}",
        "morgan_sim",
    ]
    scalar_cols = [c for c in scalar_cols if c in df.columns]

    col_labels = {
        "failure_rate": "Failure Rate %",
        "recon_rate": "Recon Rate %",
        f"prod_div_{DIVERSITY_THRESH}": f"Prod Diversity\n(t={DIVERSITY_THRESH})",
        f"bb_div_{DIVERSITY_THRESH}": f"BB Diversity\n(t={DIVERSITY_THRESH})",
        "morgan_sim": "Morgan Sim",
    }

    for testset in testsets:
        sub = df[df["testset"] == testset].copy()
        if sub.empty:
            continue

        pivot_rows = []
        for _, row in sub.iterrows():
            pivot_rows.append(
                {
                    "run": f"{SAMPLING_LABELS[row['mode']]} | {row['model']}",
                    **{c: row[c] for c in scalar_cols},
                }
            )
        heatmap_df = pd.DataFrame(pivot_rows).set_index("run")

        # Normalise per column for colour mapping
        normed = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min() + 1e-9)

        n_cols = len(scalar_cols)
        n_rows = len(pivot_rows)
        fig, ax = plt.subplots(figsize=(n_cols * 2.0 + 1, n_rows * 0.7 + 1.5))
        im = ax.imshow(normed.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([col_labels.get(c, c) for c in scalar_cols], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(heatmap_df.index, fontsize=9)
        ax.set_title(f"Summary Heatmap — {testset} (column-normalised)", fontsize=11, fontweight="bold")

        for i in range(n_rows):
            for j, col in enumerate(scalar_cols):
                val = heatmap_df.iloc[i][col]
                txt = f"{val:.1f}" if val >= 1.0 else f"{val:.3f}"
                if np.isnan(val):
                    txt = "–"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="black" if 0.3 < normed.values[i, j] < 0.75 else "white",
                )

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Normalised value")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"summary_heatmap_{testset}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    global DIVERSITY_THRESH
    parser = argparse.ArgumentParser(description="Plot SynLlama sampling experiment results.")
    parser.add_argument(
        "results_dir",
        help="Path to directory containing low_only/, medium_only/, high_only/ subdirs",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for plots (default: <results_dir>/plots)",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=DIVERSITY_THRESH,
        help=f"Analogue similarity threshold for diversity comparison (default: {DIVERSITY_THRESH})",
    )
    args = parser.parse_args()

    DIVERSITY_THRESH = args.thresh

    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")

    print(f"Collecting data from: {args.results_dir}")
    df = collect_all_data(args.results_dir)

    if df.empty:
        print("No data found. Check the directory structure.")
        sys.exit(1)

    print(f"\nFound {len(df)} runs:")
    print(df[["mode", "model", "testset"]].to_string(index=False))
    print()

    make_plots(df, out_dir)
    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
