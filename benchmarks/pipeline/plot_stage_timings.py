#!/usr/bin/env python3
"""
Plot REDI pipeline stage timings as a horizontal stacked bar chart.

Reads four JSON result files (one per case) and produces a publication-quality
figure showing time (x-axis) broken into ingest / preprocess / transform /
structure / output stages for each case (y-axis).

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/plot_stage_timings.py [--results-dir DIR] [--normalize MODE] [--out FILE]

    --normalize  raw        wall-clock seconds (default)
                 pct        fraction of total (percentage stacked)
                 per_gib    seconds per GiB of input
                 per_sample seconds per 1 000 samples
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Defaults ───────────────────────────────────────────────────────────────────
RESULTS_DIR = str(Path(__file__).parent / "results")
OUT_BASE    = str(Path(__file__).parent / "results" / "stage_timings")

CASES = [
    ("bench_xgc1.json",      "XGC1\n(fusion)"),
    ("bench_oc22.json",      "OC22 / HydraGNN\n(materials)"),
    ("bench_openfold_quick.json",  "OpenFold\n(life sciences)"),
    ("bench_climax.json",    "ClimaX / TaiESM1\n(climate)"),
]

STAGES   = ["ingest", "preprocess", "transform", "structure", "output"]
COLORS   = {
    "ingest":     "#4C72B0",
    "preprocess": "#DD8452",
    "transform":  "#55A868",
    "structure":  "#C44E52",
    "output":     "#8172B2",
}
STAGE_LABELS = {
    "ingest":     "Ingest",
    "preprocess": "Preprocess",
    "transform":  "Transform",
    "structure":  "Structure",
    "output":     "Output",
}

# Per-case abbreviated labels rendered inside each segment.
# Keyed by JSON filename (basename).
SEGMENT_LABELS = {
    "bench_climax.json": {
        "ingest":     "Ingest",
        "preprocess": "Regrid",
        "transform":  "Normalize",
        "structure":  "Structure",
        "output":     "Write NPZ",
    },
    "bench_openfold_quick.json": {
        "ingest":     "Scan",
        "preprocess": "Parse",
        "transform":  "Encode",
        "structure":  "Assemble",
        "output":     "Serialize",
    },
    "bench_oc22.json": {
        "ingest":     "LMDB Read",
        "preprocess": "Filter",
        "transform":  "Graph Enc.",
        "structure":  "PyG Build",
        "output":     "Write BP",
    },
    "bench_xgc1.json": {
        "ingest":     "BP Read",
        "preprocess": "Mesh Proj.",
        "transform":  "",
        "structure":  "Graph",
        "output":     "Write PT",
    },
}

LABEL_THRESHOLD_PCT = 8.0   # minimum segment width as % of total bar to show label


# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> list:
    """Load available JSON result files; return list of (fname, label, data) tuples."""
    entries = []
    for fname, label in CASES:
        fpath = Path(results_dir) / fname
        if not fpath.exists():
            print(f"  [skip] {fname} not found — run the benchmark first")
            continue
        with open(fpath) as f:
            data = json.load(f)
        entries.append((fname, label, data))
    return entries


def get_normalised_values(data: dict, mode: str) -> dict:
    """Return stage values in the requested normalisation mode."""
    stages_ms = data.get("stages_ms", {})
    total_ms  = data.get("total_ms", sum(stages_ms.values()))

    if mode == "raw":
        return {s: stages_ms.get(s, 0) / 1000.0 for s in STAGES}

    if mode == "pct":
        stage_total = sum(stages_ms.get(s, 0) for s in STAGES)
        return {s: 100.0 * stages_ms.get(s, 0) / stage_total if stage_total else 0
                for s in STAGES}

    if mode == "per_gib":
        gib = data.get("total_input_gib", None)
        if gib is None or gib == 0:
            # Fallback: just use raw
            return {s: stages_ms.get(s, 0) / 1000.0 for s in STAGES}
        return {s: stages_ms.get(s, 0) / 1000.0 / gib for s in STAGES}

    if mode == "per_sample":
        # Per 1 000 samples (proteins / structures / grid-pts)
        n = (data.get("n_proteins") or data.get("n_valid") or
             data.get("n_files") or 1)
        return {s: 1000.0 * stages_ms.get(s, 0) / 1000.0 / n for s in STAGES}

    raise ValueError(f"Unknown normalisation mode: {mode}")


def x_label(mode: str) -> str:
    return {
        "raw":        "Wall-clock time (s)",
        "pct":        "Fraction of total time (%)",
        "per_gib":    "Time per GiB of input (s / GiB)",
        "per_sample": "Time per 1 000 samples (s)",
    }[mode]


def make_plot(entries: list, mode: str, out_base: str):
    n = len(entries)
    if n == 0:
        print("No result files found — nothing to plot.")
        return

    fig_h = max(3.0, 1.2 * n)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    bar_h = 0.55
    y_positions = np.arange(n)

    # Normalise values
    all_vals = [get_normalised_values(data, mode) for _, _, data in entries]

    # Build stacked bars and add in-segment labels
    left = np.zeros(n)
    for stage in STAGES:
        widths = np.array([v[stage] for v in all_vals])
        ax.barh(
            y_positions, widths, left=left,
            height=bar_h, color=COLORS[stage], label=STAGE_LABELS[stage],
            edgecolor="white", linewidth=0.5,
        )
        # In-segment labels
        for i, (fname, lbl, data) in enumerate(entries):
            w = widths[i]
            total_bar = sum(all_vals[i][s] for s in STAGES)
            if total_bar > 0 and (w / total_bar * 100) >= LABEL_THRESHOLD_PCT:
                seg_label = SEGMENT_LABELS.get(fname, {}).get(stage, "")
                if seg_label:
                    ax.text(
                        left[i] + w / 2, y_positions[i],
                        seg_label, va="center", ha="center",
                        fontsize=7, color="white", fontweight="normal",
                        clip_on=True,
                    )
        left += widths

    # Annotations
    for i, (fname, lbl, data) in enumerate(entries):
        total = sum(all_vals[i][s] for s in STAGES)
        unit = {
            "raw":        "s",
            "pct":        "%",
            "per_gib":    "s/GiB",
            "per_sample": "s/1k",
        }[mode]

        if mode == "pct":
            # Show actual wall-clock total just above each bar, centred
            total_s = data.get("total_s", data.get("total_ms", 0) / 1000)
            label_s = f"{total_s:.0f} s" if total_s >= 10 else f"{total_s:.1f} s"
            ax.text(50, y_positions[i] + bar_h * 0.55,
                    label_s, va="bottom", ha="center", fontsize=8.5,
                    fontweight="bold", color="#333333")
        else:
            ax.text(total + 0.01 * left.max(), y_positions[i],
                    f"{total:.1f} {unit}", va="center", ha="left", fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([lbl for _, lbl, _ in entries], fontsize=10)
    ax.set_xlabel(x_label(mode), fontsize=11)
    ax.set_title("REDI Pipeline Stage Timings", fontsize=13, fontweight="bold")

    # Legend — horizontal, centred below the bars
    handles = [mpatches.Patch(color=COLORS[s], label=STAGE_LABELS[s]) for s in STAGES]
    if mode == "pct":
        # Place legend inside the axes in the gap below the bottom bar
        ax.legend(handles=handles, loc="lower center",
                  bbox_to_anchor=(0.5, 0.01), ncol=len(STAGES),
                  fontsize=9, framealpha=0.85, borderpad=0.4)
    else:
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # pct mode: larger bottom margin to create space for the in-axes legend
    ax.margins(y=0.28 if mode == "pct" else 0.15)

    # Add sample-size annotations below each bar
    for i, (fname, lbl, data) in enumerate(entries):
        n_str = ""
        if "n_proteins" in data:
            n_str = f"n={data['n_proteins']:,} proteins"
        elif "n_valid" in data:
            n_str = f"n={data['n_valid']:,} structs"
        elif "n_files" in data and "total_input_gib" in data:
            n_str = f"n={data['n_files']} files  ({data['total_input_gib']:.0f} GiB)"
        if n_str:
            ax.text(0, y_positions[i] - bar_h * 0.6, n_str,
                    va="top", ha="left", fontsize=7, color="#666666")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{out_base}_{mode}.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--normalize", default="raw",
                        choices=["raw", "pct", "per_gib", "per_sample"],
                        help="Normalisation mode (default: raw)")
    parser.add_argument("--out", default=OUT_BASE,
                        help="Output file base (extensions .pdf and .png added)")
    parser.add_argument("--all-modes", action="store_true",
                        help="Generate plots for all four normalisation modes")
    args = parser.parse_args()

    entries = load_results(args.results_dir)
    if not entries:
        print("No JSON result files found.  Submit the SLURM jobs first.")
        return

    print(f"\nLoaded {len(entries)} result(s):")
    for fname, lbl, data in entries:
        print(f"  {lbl.strip()}: total={data.get('total_s','?')}s")
    print()

    modes = ["raw", "pct", "per_gib", "per_sample"] if args.all_modes else [args.normalize]
    for mode in modes:
        print(f"Plotting mode={mode} …")
        make_plot(entries, mode, args.out)


if __name__ == "__main__":
    main()
