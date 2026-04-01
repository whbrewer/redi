#!/usr/bin/env python3
"""
XGC1 ITER fusion pipeline stage timing benchmark (xgc1 plugin).

Runs REDI using the redi-xgc1 plugin (--domain xgc1) on the
n560fr_ITER_PFPO_W_Ne case, limited to N timesteps.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/pipeline/runners/run_xgc1_bench.py [--n-timesteps N] [--out FILE]
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

# ── Data locations ─────────────────────────────────────────────────────────────
XGC1_ROOT   = "/lustre/orion/world-shared/fus184/fusiond-seed-xgc1-data"
CASE_FILTER = "n560fr_ITER_PFPO_W_Ne"
CFG_TPL     = str(Path(__file__).parent.parent / "configs" / "bench_xgc1.yaml")
OUT_JSON    = str(Path(__file__).parent.parent / "results" / "bench_xgc1.json")

# ── Step → stage mapping (xgc1 plugin step names) ──────────────────────────────
# build_xgc1_mesh and process_xgc1_timesteps are redistributed using substep
# ratios (read_mesh→ingest, build_graph→preprocess, read_fields→ingest,
# assemble→structure, write→output).
STAGE_MAP = {
    "load_xgc1_cases":        "ingest",
    "build_xgc1_mesh":        "structure",          # redistributed: read_mesh→ingest, build_graph→preprocess
    "process_xgc1_timesteps": "output",             # redistributed: read_fields→ingest, assemble→structure, write→output
    "write_checksums":        "output",
    "save_metadata":          "output",
}

STEP_RE = re.compile(r"(\w+)\s+.*?\((\d+)ms\)")


def parse_timings(output: str) -> dict:
    timings = {}
    for line in output.splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STEP_RE.search(clean)
        if m:
            step, ms = m.group(1), int(m.group(2))
            if ms > 0:
                timings[step] = timings.get(step, 0) + ms
    return timings


def parse_substeps(output: str) -> dict:
    """Parse substep timing lines printed by the xgc1 plugin steps."""
    substeps: dict = {}

    # ProcessXGC1Timesteps: "read_fields : NNN ms/timestep"
    ts_re   = re.compile(r"(read_fields|assemble|write)\s*:\s*([\d.]+)\s*ms/timestep")
    # BuildXGC1Mesh: "read_mesh  : NNN ms" / "build_graph: NNN ms"
    mesh_re = re.compile(r"(read_mesh|build_graph)\s*:\s*([\d.]+)\s*ms\b")

    for line in output.splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        for pat in (ts_re, mesh_re):
            m = pat.search(clean)
            if m:
                substeps[m.group(1)] = float(m.group(2))

    return substeps


def aggregate_stages(step_timings: dict, substeps: dict) -> dict:
    stages = {"ingest": 0, "preprocess": 0, "transform": 0, "structure": 0, "output": 0}

    for step, ms in step_timings.items():
        stage = STAGE_MAP.get(step)
        if stage:
            stages[stage] += ms

    # Redistribute process_xgc1_timesteps wall time:
    #   read_fields → ingest    (ADIOS BP read of timestep data)
    #   assemble    → structure (feature matrix assembly → particle-mesh graph)
    #   write       → output    (shards)
    proc_ms  = step_timings.get("process_xgc1_timesteps", 0)
    ts_keys  = ["read_fields", "assemble", "write"]
    ts_sub   = {k: substeps.get(k, 0) for k in ts_keys}
    ts_total = sum(ts_sub.values())
    if proc_ms and ts_total > 0:
        ts_stage = {"read_fields": "ingest", "assemble": "structure", "write": "output"}
        stages["output"] -= proc_ms
        for k in ts_keys:
            stages[ts_stage[k]] += int(proc_ms * ts_sub[k] / ts_total)

    # Redistribute build_xgc1_mesh wall time:
    #   read_mesh   → ingest    (ADIOS BP read of mesh file)
    #   build_graph → preprocess (mesh projection: 2D → 3D toroidal)
    mesh_ms    = step_timings.get("build_xgc1_mesh", 0)
    mesh_read  = substeps.get("read_mesh", 0)
    mesh_bld   = substeps.get("build_graph", 0)
    mesh_total = mesh_read + mesh_bld
    if mesh_ms and mesh_total > 0:
        stages["structure"] -= mesh_ms
        stages["ingest"]    += int(mesh_ms * mesh_read / mesh_total)
        stages["preprocess"] += int(mesh_ms * mesh_bld  / mesh_total)

    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-timesteps", type=int, default=5,
                        help="Max timesteps per case (default: 5)")
    parser.add_argument("--out", default=OUT_JSON, help="Output JSON path")
    args = parser.parse_args()

    # Check data size
    case_dir = Path(XGC1_ROOT) / CASE_FILTER
    bp_dirs  = sorted(case_dir.glob("xgc.3d.*.bp"))[:args.n_timesteps]
    total_gib = 0.0
    for bp in bp_dirs:
        try:
            total_gib += sum(f.stat().st_size for f in bp.iterdir() if f.is_file()) / (1024**3)
        except Exception:
            pass

    print(f"\nXGC1 Plugin Stage Timing Benchmark")
    print(f"  Root:       {XGC1_ROOT}")
    print(f"  Case:       {CASE_FILTER}")
    print(f"  Timesteps:  {args.n_timesteps}  ({total_gib:.1f} GiB)")
    print()

    with open(CFG_TPL) as f:
        cfg = yaml.safe_load(f)
    cfg["case_filter"]    = CASE_FILTER
    cfg["max_timesteps"]  = args.n_timesteps

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["output_dir"] = tmpdir
        tmp_cfg = "/tmp/redi_xgc1_bench.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg, f)

        print(f"  Running: redi run {XGC1_ROOT} --domain xgc1 -c {tmp_cfg}", flush=True)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", XGC1_ROOT, "--domain", "xgc1", "-c", tmp_cfg],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        total_ms = int((time.time() - t0) * 1000)
        Path(tmp_cfg).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  WARNING: redi exited {result.returncode}")
        print(result.stdout[-3000:])

    step_t   = parse_timings(result.stdout)
    substeps = parse_substeps(result.stdout)
    stage_t  = aggregate_stages(step_t, substeps)

    throughput_gib_s = total_gib / (total_ms / 1000) if total_ms > 0 else 0

    result_dict = {
        "case":              "xgc1_iter",
        "domain":            "xgc1",
        "n_files":           len(bp_dirs),
        "files":             [bp.name for bp in bp_dirs],
        "total_input_gib":   round(total_gib, 2),
        "stages_ms":         stage_t,
        "total_ms":          total_ms,
        "total_s":           round(total_ms / 1000, 1),
        "throughput_gib_per_s": round(throughput_gib_s, 3),
        "throughput_label":  "GiB/s",
        "step_timings_ms":   step_t,
    }
    if substeps:
        result_dict["substep_timings_ms_per_timestep"] = substeps

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nResults → {args.out}")

    W = 65
    print("\n" + "=" * W)
    print(f"XGC1 PLUGIN TIMING SUMMARY  ({len(bp_dirs)} timesteps, {total_gib:.1f} GiB)")
    print("=" * W)
    print(f"  {'Stage':<16}  {'s':>8}  {'%':>6}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*6}")
    for stage in ["ingest", "preprocess", "transform", "structure", "output"]:
        ms  = stage_t[stage]
        pct = 100 * ms / total_ms if total_ms else 0
        print(f"  {stage:<16}  {ms/1000:>8.1f}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<16}  {total_ms/1000:>8.1f}  100.0%")
    print(f"\n  Throughput: {throughput_gib_s:.3f} GiB/s")
    if substeps:
        print(f"\n  Timestep substep breakdown:")
        for k in ["read_fields", "assemble", "write"]:
            if k in substeps:
                print(f"    {k:<14s}: {substeps[k]:7.1f} ms/timestep")
    print("=" * W)


if __name__ == "__main__":
    main()
