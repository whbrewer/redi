#!/usr/bin/env python3
"""
Plot REDI parallel speedup benchmark results — publication-quality figure.

Four subplots (one per domain), each showing measured speedup vs ideal linear
speedup as a function of worker/node count.

XGC1 (Fusion) subplot shows both strong scaling (speedup, left axis) and
weak scaling (efficiency, right axis).

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
        "title": "Climate\n(ClimaX/AWI-ESM, 800 NetCDF Files)",
        "color": "#2ca02c",   # green
        "marker": "o",
        "file": "speedup_climax-awi.json",
        "xlabel": "Workers",
    },
    "xgc1": {
        "title": "Fusion\n(XGC1, 1168 snapshots, 7.9 TB)",
        "color": "#9467bd",   # purple
        "marker": "s",
        "file": "speedup_xgc1.json",
        "weak_file": "speedup_xgc1_weak.json",
        "xlabel": "Nodes (56 processes/node)",
    },
    "openfold": {
        "title": "Life Sciences\n(OpenFold/OPS, 4,096 mmCIF files)",
        "color": "#d62728",   # red
        "marker": "^",
        "file": "speedup_openfold.json",
        "xlabel": "Workers",
    },
    "oc22": {
        "title": "Materials\n(HydraGNN/OC22, 160k LMDB structures)",
        "color": "#ff7f0e",   # orange
        "marker": "D",
        "file": "speedup_oc22.json",
        "xlabel": "Workers",
    },
}

DOMAIN_ORDER = ["climax", "openfold", "oc22", "xgc1"]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def _parse_runs(raw):
    entries = raw.get("runs", raw if isinstance(raw, list) else [])
    rows = []
    for e in entries:
        rows.append({
            "workers":    e["workers"],
            "processes":  e.get("processes"),
            "speedup":    e.get("speedup"),
            "efficiency": e.get("parallel_efficiency", e.get("efficiency", 1.0)),
        })
    return sorted(rows, key=lambda e: e["workers"])


def load_results(results_dir: Path) -> dict:
    data = {}
    for key, meta in DOMAIN_META.items():
        path = results_dir / meta["file"]
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {key}")
            continue
        with open(path) as f:
            data[key] = {"strong": _parse_runs(json.load(f))}

        weak_file = meta.get("weak_file")
        if weak_file:
            wpath = results_dir / weak_file
            if wpath.exists():
                with open(wpath) as f:
                    data[key]["weak"] = _parse_runs(json.load(f))
    return data


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_standard(ax, domain, entries):
    """Speedup vs workers with ideal reference line and efficiency annotations."""
    meta     = DOMAIN_META[domain]
    workers  = [e["workers"]    for e in entries]
    speedups = [e["speedup"]    for e in entries]
    effs     = [e["efficiency"] for e in entries]
    max_w    = max(workers)
    min_w    = min(workers)

    ax.plot(workers, speedups,
            color=meta["color"], marker=meta["marker"],
            linewidth=1.8, markersize=6, label="Strong scaling", zorder=1)

    # Ideal line for every data point
    ax.plot(workers, [w / workers[0] for w in workers],
            color="#aaaaaa", linewidth=1.2, linestyle="--",
            label="Ideal", zorder=2)

    for w, sp, eff in zip(workers, speedups, effs):
        ax.annotate(f"{eff:.2f}",
                    xy=(w, sp),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=6.5, color=meta["color"])

    ax.set_xscale("log", base=2)
    ax.set_xlim(min_w * 0.8, max_w * 1.3)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks(workers)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel(meta.get("xlabel", "Workers"), fontsize=9)
    ax.set_ylabel("Speedup", fontsize=9)


def _plot_with_weak(ax, domain, strong_entries, weak_entries):
    """Strong scaling speedup (left axis) + weak scaling efficiency (right axis)."""
    meta = DOMAIN_META[domain]

    _plot_standard(ax, domain, strong_entries)

    # Weak scaling on right y-axis
    ax2 = ax.twinx()
    w_workers = [e["workers"]    for e in weak_entries]
    w_effs    = [e["efficiency"] for e in weak_entries]

    weak_color = "#1f77b4"  # blue
    ax2.plot(w_workers, w_effs,
             color=weak_color, marker="o", linestyle="--",
             linewidth=1.4, markersize=5, label="Weak scaling (eff.)", zorder=3)
    ax2.axhline(1.0, color="#aaaaaa", linewidth=0.8, linestyle=":")
    ax2.set_ylim(0, 1.25)
    ax2.set_ylabel("Parallel Efficiency", fontsize=9, color=weak_color)
    ax2.tick_params(axis="y", labelsize=8, labelcolor=weak_color)

    # Combine legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left",
              framealpha=0.85, handlelength=1.6, borderpad=0.5)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot(data: dict, out_path: Path):
    domains_present = [d for d in DOMAIN_ORDER if d in data]
    n = len(domains_present)

    fig, axes = plt.subplots(1, n, figsize=(14, 4.2), sharey=False)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.06, right=0.95, top=0.84, bottom=0.18, wspace=0.48)

    for ax, domain in zip(axes, domains_present):
        meta    = DOMAIN_META[domain]
        strong  = data[domain]["strong"]
        weak    = data[domain].get("weak")

        if weak:
            _plot_with_weak(ax, domain, strong, weak)
        else:
            _plot_standard(ax, domain, strong)
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85,
                      handlelength=1.8, borderpad=0.5)

        ax.set_title(meta["title"], fontsize=9, fontweight="bold", pad=6)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)

    #fig.suptitle(
    #    "REDI Parallel Speedup — Measured vs Ideal (efficiency annotated)",
    #    fontsize=11, fontweight="bold", y=0.97,
    #)

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
        default="benchmarks/parallel/results/speedup_benchmark.png",
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
