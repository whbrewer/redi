#!/usr/bin/env python3
"""
Plot REDI parallel speedup benchmark results — publication-quality figure.

Four subplots (one per domain), each showing measured speedup vs ideal linear
speedup as a function of worker count.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/parallel/plot_speedup_benchmark.py [--results-dir DIR] [--out FILE]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style — matches plot_io_benchmark.py palette
# ---------------------------------------------------------------------------

DOMAIN_META = {
    "climax": {
        "title": "Climate\n(ClimaX/CMCC, 32 NC files)",
        "color": "#2ca02c",   # green
        "marker": "o",
        "file": "speedup_climax.json",
    },
    "xgc1": {
        "title": "Fusion\n(XGC1, 32 snapshots, 177 GiB)",
        "color": "#9467bd",   # purple
        "marker": "s",
        "file": "speedup_xgc1.json",
    },
    "openfold": {
        "title": "Life Sciences\n(OpenFold, 128 proteins)",
        "color": "#d62728",   # red
        "marker": "^",
        "file": "speedup_openfold.json",
    },
    "oc22": {
        "title": "Materials\n(OC22, 32 LMDB shards, 47 GiB)",
        "color": "#ff7f0e",   # orange
        "marker": "D",
        "file": "speedup_oc22.json",
    },
}

DOMAIN_ORDER = ["climax", "openfold", "oc22", "xgc1"]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict:
    data = {}
    for key, meta in DOMAIN_META.items():
        path = results_dir / meta["file"]
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {key}")
            continue
        with open(path) as f:
            raw = json.load(f)
        # Normalise: top-level "runs" list with parallel_efficiency key
        entries = raw.get("runs", raw if isinstance(raw, list) else [])
        rows = []
        for e in entries:
            rows.append({
                "workers":    e["workers"],
                "time_s":     e.get("total_s", e.get("time_s", 0)),
                "speedup":    e["speedup"],
                "efficiency": e.get("parallel_efficiency", e.get("efficiency", 1.0)),
            })
        data[key] = sorted(rows, key=lambda e: e["workers"])
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(data: dict, out_path: Path):
    domains_present = [d for d in DOMAIN_ORDER if d in data]
    n = len(domains_present)

    fig, axes = plt.subplots(1, n, figsize=(14, 4.2), sharey=False)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.18, wspace=0.38)

    for ax, domain in zip(axes, domains_present):
        meta    = DOMAIN_META[domain]
        entries = data[domain]

        workers  = [e["workers"]    for e in entries]
        speedups = [e["speedup"]    for e in entries]
        effs     = [e["efficiency"] for e in entries]

        max_w = max(workers)

        # Ideal speedup line
        ideal_x = [workers[0], max_w]
        ideal_y = [workers[0], max_w]
        ax.plot(ideal_x, ideal_y,
                color="#aaaaaa", linewidth=1.2, linestyle="--",
                label="Ideal", zorder=1)

        # Measured speedup
        ax.plot(workers, speedups,
                color=meta["color"], marker=meta["marker"],
                linewidth=1.8, markersize=6, label="Measured", zorder=2)

        # Annotate efficiency at each point
        for w, sp, eff in zip(workers, speedups, effs):
            ax.annotate(f"{eff:.2f}",
                        xy=(w, sp),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=6.5, color=meta["color"])

        ax.set_xscale("log", base=2)
        ax.set_xlim(0.8, max_w * 1.3)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticks(workers)

        ax.set_xlabel("Workers", fontsize=9)
        ax.set_ylabel("Speedup", fontsize=9)
        ax.set_title(meta["title"], fontsize=9, fontweight="bold", pad=6)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)

        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85,
                  handlelength=1.8, borderpad=0.5)

    fig.suptitle(
        "REDI Parallel Speedup — Measured vs Ideal (efficiency annotated)",
        fontsize=11, fontweight="bold", y=0.97,
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="benchmarks/parallel/results",
        help="Directory containing speedup_*.json files",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/parallel/results/speedup_benchmark.pdf",
        help="Output file (.pdf or .png)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_results(results_dir)
    print(f"  Loaded: {list(data.keys())}")

    print("Plotting...")
    plot(data, out_path)


if __name__ == "__main__":
    main()
