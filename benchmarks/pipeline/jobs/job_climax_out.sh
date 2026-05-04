#!/bin/bash
# ── REDI Output Run: ClimaX / TaiESM1 ─────────────────────────────────────────
# Runs the full REDI pipeline on all TaiESM1 2m_temperature NetCDF chunks and
# writes DATA_CARD.md, metadata.json, and NPZ output to the shared Lustre path.
#SBATCH -A stf053
#SBATCH -J redi_out_climax
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_climax_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_climax_%j.out

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"
NC_DIR="/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/TaiESM1/raw/2m_temperature"
OUT_DIR="/lustre/orion/proj-shared/stf249/redi/outputs/climax"
CFG="$REDI_DIR/benchmarks/pipeline/configs/bench_climax.yaml"

echo "=== REDI ClimaX Output Run  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Output:  $OUT_DIR"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/pipeline/logs "$OUT_DIR"

# Process each NC file individually to avoid OOM from xr.concat on all 17 files
for nc_file in "$NC_DIR"/*.nc; do
    fname=$(basename "$nc_file" .nc)
    file_out="$OUT_DIR/$fname"
    mkdir -p "$file_out"
    echo "  Processing: $nc_file"
    redi run "$nc_file" \
        --domain climate \
        -c "$CFG" \
        -o "$file_out" \
        -v
done

# Generate top-level metadata.json aggregating all per-file runs
echo "  Generating top-level metadata.json ..."
python3 - <<'PYEOF'
import json, os, glob
from pathlib import Path

out_dir = os.environ["OUT_DIR"]
sub_dirs = sorted(glob.glob(f"{out_dir}/2m_temperature_*/"))

per_file = []
all_errors = []
time_starts, time_ends = [], []
total_timesteps = 0

for d in sub_dirs:
    mf = Path(d) / "metadata.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        per_file.append({"subdir": Path(d).name, **{k: v for k, v in m.items()
                         if k not in ("saved_files",)}})
        all_errors.extend(m.get("errors", []))
        tr = m.get("time_range", {})
        if tr.get("start"): time_starts.append(tr["start"])
        if tr.get("end"):   time_ends.append(tr["end"])
        total_timesteps += tr.get("steps", 0)

first = per_file[0] if per_file else {}
summary = {
    "dataset":          "TaiESM1 2m_temperature",
    "domain":           "climate",
    "n_files":          len(per_file),
    "total_timesteps":  total_timesteps,
    "time_range":       {"start": min(time_starts) if time_starts else None,
                         "end":   max(time_ends)   if time_ends   else None},
    "input_shape_per_file":  "14601×192×288",
    "output_shape_per_file": "14601×128×256",
    "regridded":        first.get("regridded"),
    "normalization":    "z-score per variable",
    "output_format":    first.get("output_format"),
    "completed_stages": first.get("completed_stages"),
    "errors":           all_errors,
    "per_file":         per_file,
}
out = Path(out_dir) / "metadata.json"
out.write_text(json.dumps(summary, indent=2, default=str))
print(f"  Wrote {out}")
PYEOF

echo "=== Done  $(date) ==="
echo "Output files:"
ls -lh "$OUT_DIR" 2>/dev/null || true
