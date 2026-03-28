#!/usr/bin/env python3
"""
Plot REDI I/O benchmark results — publication-quality figure.

Four subplots (one per domain), each showing read throughput (samples/sec)
vs number of DataLoader workers for every format benchmarked.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/plot_io_benchmark.py [--results-dir DIR] [--out FILE]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLORS = {
    "native":        "#7f7f7f",   # grey  — native / baseline format
    "npz":           "#d62728",   # red   — NPZ
    "lmdb":          "#ff7f0e",   # orange
    "lmdb_sharded":  "#ff7f0e",
    "hdf5":          "#1f77b4",   # blue
    "zarr":          "#2ca02c",   # green
    "adios2":        "#9467bd",   # purple
}

LABELS = {
    "native":        "Native (mmCIF+A3M)",
    "npz":           "NPZ",
    "lmdb":          "LMDB",
    "lmdb_sharded":  "LMDB (sharded)",
    "hdf5":          "HDF5 (REDI)",
    "zarr":          "Zarr (REDI)",
    "adios2":        "ADIOS2 (native)",
}

MARKERS = {
    "native":        "s",
    "npz":           "^",
    "lmdb":          "D",
    "lmdb_sharded":  "D",
    "hdf5":          "o",
    "zarr":          "o",
    "adios2":        "P",
}

DOMAIN_TITLES = {
    "life_sciences": "Life Sciences\n(OpenFold, per-protein NPZ)",
    "materials":     "Materials\n(OC22, 8.2M graphs)",
    "climate":       "Climate\n(ClimaX/TaiESM1, 181×360 grid)",
    "fusion":        "Fusion\n(XGC1, plasma timesteps)",
}

DOMAIN_ORDER = ["climate", "life_sciences", "materials", "fusion"]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict:
    data = {}
    paths = {
        "life_sciences": results_dir / "life_sciences" / "benchmark_life_sciences.json",
        "materials":     results_dir / "materials"     / "benchmark_materials.json",
        "climate":       results_dir / "climate"       / "benchmark_climate.json",
        "fusion":        results_dir / "fusion"        / "benchmark_fusion.json",
    }
    for domain, path in paths.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {domain}")
            continue
        with open(path) as f:
            raw = json.load(f)
        # life_sciences nests under "results" -> "read"
        results = raw.get("results", raw)
        read = results.get("read", results)
        # convert string keys to int
        data[domain] = {fmt: {int(k): v for k, v in wc.items()}
                        for fmt, wc in read.items()
                        if isinstance(wc, dict)}
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(data: dict, out_path: Path):
    n_domains = len(DOMAIN_ORDER)
    fig, axes = plt.subplots(1, n_domains, figsize=(14, 4.2), sharey=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.18, wspace=0.38)

    for ax, domain in zip(axes, DOMAIN_ORDER):
        if domain not in data:
            ax.set_visible(False)
            continue

        domain_data = data[domain]
        worker_counts = sorted({w for fmtd in domain_data.values() for w in fmtd})

        for fmt, wc_data in domain_data.items():
            wcs   = sorted(wc_data)
            vals  = [wc_data[w] for w in wcs]
            # mark timeouts (NaN) with open markers
            valid = [not (isinstance(v, float) and math.isnan(v)) for v in vals]
            y     = [v if ok else np.nan for v, ok in zip(vals, valid)]

            color  = COLORS.get(fmt, "#333333")
            marker = MARKERS.get(fmt, "o")
            label  = LABELS.get(fmt, fmt.upper())

            ax.plot(wcs, y, color=color, marker=marker,
                    linewidth=1.8, markersize=6, label=label)

            # annotate timeout points
            for w, v, ok in zip(wcs, vals, valid):
                if not ok:
                    ax.annotate("T/O", xy=(w, ax.get_ylim()[0]),
                                xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", fontsize=6,
                                color=color)

        ax.set_xscale("log", base=2)
        ax.set_xlim(0.8, max(worker_counts) * 1.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticks(worker_counts)

        # y-axis: use K suffix if values exceed 1000
        max_val = max((v for wc in domain_data.values() for v in wc.values()
                       if isinstance(v, float) and not math.isnan(v)), default=1)
        if max_val >= 1000:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else str(int(x))))

        ax.set_xlabel("DataLoader workers", fontsize=9)
        ax.set_ylabel("Samples / sec", fontsize=9)
        ax.set_title(DOMAIN_TITLES.get(domain, domain), fontsize=9, fontweight="bold", pad=6)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)
        ax.set_ylim(bottom=0)

        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85,
                  handlelength=1.8, borderpad=0.5)

    #fig.suptitle(
    #    "REDI I/O Benchmark — Read Throughput vs DataLoader Workers",
    #    fontsize=11, fontweight="bold", y=0.97,
    #)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/lustre/orion/scratch/whbrewer/stf053/openfold_io_bench",
                        help="Directory containing per-domain result subdirs")
    parser.add_argument("--out", default="benchmarks/io/results/io_benchmark.png",
                        help="Output file (.pdf or .png)")
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
