#!/usr/bin/env python3
"""
Generate SC26 paper tables from validated results.

Table V:   Multi-domain validation + AIDRIN quality metrics
Table VI:  Per-stage timing across domains

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/gen_paper_tables.py
"""

import json
from pathlib import Path

VR = Path("validate_results")
OUT_TEX_V   = VR / "table_v_validate_aidrin.tex"
OUT_TEX_VI  = VR / "table_vi_timing.tex"
OUT_TXT     = VR / "paper_tables.txt"


# ─── Data ─────────────────────────────────────────────────────────────────────

ROWS_V = [
    # (use_case, domain, validate_json, aidrin_json, corr_key, mae_key, mae_unit, note)
    ("ClimaX",   "Climate",    "taiesm1_2m_temp_all_validate.json", "aidrin_climate.json",   None, None, "K",    None),
    ("HydraGNN", "Materials",  "aflow_validate.json",               "aidrin_aflow.json",     None, None, "eV",   None),
    ("OpenFold", "Life Sci.",  "openfold_validate.json",            "aidrin_openfold.json",  None, None, "—",    None),
    ("sPHENIX",  "Nuc. Phys.","sphenix_validate.json",             "aidrin_sphenix.json",   None, None, "ADC",  None),
    ("ExaDigiT*","HPC",        "frontier_validate.json",            "aidrin_full_frontier.json", None, None, "s", "Frontier representative"),
    ("XGC1†",    "Fusion",     None,                                None,                    None, None, "—",    "Deferred"),
]

TIMING_ROWS = [
    ("ClimaX",    "Climate",    "bench_climate_timing.json",   "14601 steps",   "grid-pts/s"),
    ("HydraGNN",  "Materials",  "bench_aflow_timing.json",     "9 structs",     "structs/s"),
    ("OpenFold",  "Life Sci.",  "bench_openfold_timing.json",  "1 seq",         "seq/s"),
    ("sPHENIX",   "Nuc. Phys.","bench_sphenix_timing.json",   "1310 events",   "events/s"),
    ("ExaDigiT†", "HPC",        None,                          "10 systems",    "—"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _scalar(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    return d


def load_validate(fname):
    if not fname:
        return None
    p = VR / fname
    if not p.exists():
        return None
    d = json.load(open(p))
    # Handle list (climate multi-chunk summary)
    if isinstance(d, list):
        d = d[0]
    return d


def load_aidrin(fname):
    if not fname:
        return None
    p = VR / fname
    if not p.exists():
        return None
    return json.load(open(p))


def get_sim(d):
    if d is None:
        return None, None
    sim = d.get("similarity", {})
    corr = sim.get("corr")
    mae  = sim.get("mae")
    if corr is None:
        corr = d.get("corr")
        mae  = d.get("mae")
    return corr, mae


def get_aidrin_deltas(d):
    """Return (Δcomplete, Δdup, Δoutlier, Δimbalance) or (None,...) if missing."""
    if d is None:
        return None, None, None, None
    raw = d.get("raw", {})
    ai  = d.get("ai_ready", {})

    def get(section, *keys):
        rv = _scalar(raw, section, *keys)
        av = _scalar(ai,  section, *keys)
        return rv, av

    rc, ac = get("completeness", "completeness", "Overall Completeness")
    rd, ad = get("duplicity", "duplicity", "Duplicity scores", "Overall duplicity of the dataset")
    ro, ao = get("outliers", "outliers", "Outlier scores", "Overall outlier score")
    ri, ai_ci = get("class_imbalance", "class_imbalance", "Imbalance degree", "Imbalance Degree score")

    dc  = (ac - rc) if rc is not None and ac is not None else None
    dd  = (ad - rd) if rd is not None and ad is not None else None
    do  = (ao - ro) if ro is not None and ao is not None else None
    di  = (ai_ci - ri) if ri is not None and ai_ci is not None else None
    return dc, dd, do, di


def fmt(val, prec=3, sign=False):
    if val is None:
        return "—"
    if sign:
        return f"{val:+.{prec}f}"
    return f"{val:.{prec}f}"


def arrow(delta, direction="↓"):
    """direction: ↓ means lower is better → negative delta is good."""
    if delta is None:
        return ""
    if abs(delta) < 0.001:
        return ""
    if direction == "↓":
        return r"\improvement" if delta < 0 else r"\degradation"
    else:  # ↑ = higher is better
        return r"\improvement" if delta > 0 else r"\degradation"


# ─── Table V ──────────────────────────────────────────────────────────────────

def build_table_v_txt():
    lines = []
    W = 100
    lines.append("=" * W)
    lines.append("TABLE V — MULTI-DOMAIN VALIDATION + AIDRIN DATA QUALITY")
    lines.append("=" * W)
    h = f"{'Use Case':<12} {'Domain':<12} {'Corr':>7} {'MAE':>8} {'ΔComplete':>10} {'ΔDup':>7} {'ΔOutlier':>10} {'ΔImbalance':>12}"
    lines.append(h)
    lines.append("-" * W)

    for (uc, domain, vf, af, _, _, mae_unit, note) in ROWS_V:
        vd = load_validate(vf)
        ad = load_aidrin(af)
        corr, mae = get_sim(vd)
        dc, dd, do, di = get_aidrin_deltas(ad)

        if note == "Deferred":
            lines.append(f"{uc:<12} {domain:<12} {'—':>7} {'—':>8} {'—':>10} {'—':>7} {'—':>10} {'—':>12}  (deferred)")
            continue

        corr_str = fmt(corr, 4) if corr is not None else "—"
        mae_str  = f"{fmt(mae,4)} {mae_unit}" if mae is not None else "—"
        dc_str   = fmt(dc, 3, sign=True) if dc is not None else "—"
        dd_str   = fmt(dd, 3, sign=True) if dd is not None else "—"
        do_str   = fmt(do, 3, sign=True) if do is not None else "—"
        di_str   = fmt(di, 3, sign=True) if di is not None else "N/A"

        row = f"{uc:<12} {domain:<12} {corr_str:>7} {mae_str:>12} {dc_str:>10} {dd_str:>7} {do_str:>10} {di_str:>12}"
        lines.append(row)

    lines.append("=" * W)
    lines.append("  * HPC row = Frontier representative; see Table IV for all 10 systems")
    lines.append("  † XGC1 deferred pending GT from colleague")
    lines.append("  Δ = AI-ready minus raw (positive = increase, negative = decrease)")
    lines.append("  Outlier/Imbalance: negative Δ = improvement after REDI processing")
    return "\n".join(lines)


def build_table_v_latex():
    rows = []
    for (uc, domain, vf, af, _, _, mae_unit, note) in ROWS_V:
        vd = load_validate(vf)
        ad = load_aidrin(af)
        corr, mae = get_sim(vd)
        dc, dd, do, di = get_aidrin_deltas(ad)

        if note == "Deferred":
            rows.append(f"  {uc} & {domain} & — & — & — & — & — & — \\\\")
            continue

        corr_str = fmt(corr, 3) if corr is not None else "—"
        mae_str  = fmt(mae, 4) if mae is not None else "—"

        def tex_delta(d, direction="↓"):
            if d is None: return "—"
            val_str = f"{d:+.3f}"
            if abs(d) > 0.001:
                if (direction == "↓" and d < 0) or (direction == "↑" and d > 0):
                    return r"\textbf{" + val_str + "}"
            return val_str

        dc_str = tex_delta(dc, "↑")
        dd_str = tex_delta(dd, "↓")
        do_str = tex_delta(do, "↓")
        di_str = tex_delta(di, "↓") if di is not None else "N/A"

        rows.append(f"  {uc} & {domain} & {corr_str} & {mae_str} & {dc_str} & {dd_str} & {do_str} & {di_str} \\\\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Multi-domain validation and AIDRIN data quality assessment.
Corr and MAE measure fidelity between REDI output and ground-truth reference data.
$\Delta$ columns show the change (AI-ready $-$ raw) for each AIDRIN metric;
bold indicates improvement.
$\dagger$ XGC1 deferred pending GT from colleague.
*HPC row is Frontier; full 10-system breakdown in Table~IV.}}
\label{{tab:multidomain}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
\textbf{{Use Case}} & \textbf{{Domain}} & \textbf{{Corr}} & \textbf{{MAE}}
  & \textbf{{$\Delta$Complete}} & \textbf{{$\Delta$Dup}} & \textbf{{$\Delta$Outlier}} & \textbf{{$\Delta$Imbalance}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# ─── Table VI (Timing) ────────────────────────────────────────────────────────

def load_timing(fname):
    if not fname:
        return None
    p = VR / fname
    if not p.exists():
        return None
    return json.load(open(p))


def build_table_vi_txt():
    lines = []
    W = 110
    lines.append("=" * W)
    lines.append("TABLE VI — REDI PIPELINE TIMING ACROSS DOMAINS")
    lines.append("=" * W)
    h = f"{'Use Case':<12} {'Domain':<12} {'Size':<16} {'Ingest':>8} {'Preproc':>8} {'Trans':>8} {'Struct':>8} {'Output':>8} {'Total':>8} {'Throughput':>18}"
    lines.append(h)
    lines.append("-" * W)

    for (uc, domain, tf, size_str, thru_unit) in TIMING_ROWS:
        td = load_timing(tf)
        if td is None:
            lines.append(f"{uc:<12} {domain:<12} {size_str:<16} {'[see Table IV]':>62}")
            continue

        stages = td.get("stages_ms", {})
        total_s = td.get("total_s", td.get("total_ms", 0) / 1000)
        thru = (td.get("throughput_grid_pts_per_s") or td.get("throughput_structs_per_s") or
                td.get("throughput_seqs_per_s") or td.get("throughput_events_per_s"))

        def ms(k): return f"{stages.get(k, 0):,}"
        thru_str = f"{thru:,.0f} {thru_unit}" if thru else "—"
        row = (f"{uc:<12} {domain:<12} {size_str:<16} "
               f"{ms('ingest'):>8} {ms('preprocess'):>8} {ms('transform'):>8} "
               f"{ms('structure'):>8} {ms('output'):>8} {total_s:>7.1f}s {thru_str:>18}")
        lines.append(row)

    lines.append("=" * W)
    lines.append("  All times in ms (except Total in seconds). HPC timing in Table IV.")
    lines.append("  ClimaX throughput in grid-pts/s; others as labeled.")
    return "\n".join(lines)


def build_table_vi_latex():
    rows = []
    for (uc, domain, tf, size_str, thru_unit) in TIMING_ROWS:
        td = load_timing(tf)
        if td is None:
            rows.append(f"  {uc} & {domain} & {size_str} & \\multicolumn{{6}}{{c}}{{see Table~IV}} \\\\")
            continue
        stages = td.get("stages_ms", {})
        total_s = td.get("total_s", td.get("total_ms", 0) / 1000)
        thru = (td.get("throughput_grid_pts_per_s") or td.get("throughput_structs_per_s") or
                td.get("throughput_seqs_per_s") or td.get("throughput_events_per_s"))

        def ms(k):
            v = stages.get(k, 0)
            return f"{v:,}" if v > 0 else "—"

        thru_str = f"{thru:,.0f}" if thru and thru > 100 else (f"{thru:.2f}" if thru else "—")
        rows.append(
            f"  {uc} & {domain} & {size_str} & {ms('ingest')} & {ms('preprocess')} & "
            f"{ms('transform')} & {ms('structure')} & {ms('output')} & {total_s:.1f} & {thru_str} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{REDI pipeline wall-clock timing by domain.
Stage times in milliseconds; Total in seconds.
Throughput units: ClimaX = grid-pts/s; HydraGNN = structs/s;
OpenFold = seq/s; sPHENIX = events/s.
HPC timing reported separately in Table~IV.}}
\label{{tab:timing}}
\begin{{tabular}}{{llrrrrrrrr}}
\toprule
\textbf{{Use Case}} & \textbf{{Domain}} & \textbf{{Size}}
  & \textbf{{Ingest}} & \textbf{{Preproc}} & \textbf{{Trans}}
  & \textbf{{Struct}} & \textbf{{Output}} & \textbf{{Total (s)}} & \textbf{{Throughput}} \\
& & & \multicolumn{{5}}{{c}}{{(ms)}} & & \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    txt_v  = build_table_v_txt()
    txt_vi = build_table_vi_txt()
    tex_v  = build_table_v_latex()
    tex_vi = build_table_vi_latex()

    # Print to stdout
    print(txt_v)
    print()
    print(txt_vi)

    # Save LaTeX
    OUT_TEX_V.write_text(tex_v)
    OUT_TEX_VI.write_text(tex_vi)
    print(f"\nLaTeX Table V  → {OUT_TEX_V}")
    print(f"LaTeX Table VI → {OUT_TEX_VI}")

    # Save combined text
    OUT_TXT.write_text(txt_v + "\n\n" + txt_vi + "\n")
    print(f"Text tables    → {OUT_TXT}")


if __name__ == "__main__":
    main()
