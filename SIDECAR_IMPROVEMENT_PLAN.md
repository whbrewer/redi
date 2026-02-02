# REDI Sidecar Improvement Plan

**Date:** 2026-02-02
**Goal:** Transform `redi sidecar` from a static workflow parser into a useful observability layer for AI practitioners.

## Context

From `/home/w1b/drai/logs/climax_vs_redi`:
- ClimaX Snakemake pipeline is already excellent for its purpose (cross-year normalization, pressure levels, unit conversions, sharding)
- REDI shouldn't try to replace it
- The opportunity: be the **observability layer** that documents what the pipeline actually does

## Current State (Problems)

The current sidecar implementation:
- Only parses workflow file syntax (Snakefile text)
- Extracts rule names, inputs, outputs via regex
- Maps rules to REDI stages via keyword matching ("download" → ingest)
- Generates a basic JSON report

**What an AI practitioner actually needs:**
- Data shape/schema changes per step → Not tracked
- Intermediate data statistics → Not captured
- Normalization parameters used → Not recorded
- Train/val/test split ratios → Unknown
- Data leakage detection → None
- Feature engineering details → Not documented
- Quality metrics per stage → Missing
- Actual runtime observation → Not implemented

## The Value Proposition

**For ClimaX (or any existing pipeline):** "Run your workflow unchanged, get a comprehensive Data Card documenting exactly what happened to your data."

---

## Phase 1: Runtime Observation (not just static parsing)

**Current:** Parses Snakefile text, guesses what rules do
**Proposed:** Hooks into Snakemake execution to capture actual data flow

```bash
# Instead of just parsing:
redi sidecar --workflow Snakefile

# Actually observe execution:
snakemake --snakefile Snakefile 2>&1 | redi sidecar --observe

# Or wrap execution:
redi sidecar --exec "snakemake --snakefile Snakefile"
```

**What to capture:**

| Event | Data Captured |
|-------|---------------|
| Rule starts | Input file paths, sizes, checksums |
| Rule completes | Output file paths, sizes, timing |
| File created | Shape, dtype, statistics (min/max/mean/std) |
| File read | Same + validate against expected |

---

## Phase 2: Intermediate Data Snapshots

**The key insight:** ClimaX computes `normalize_mean.npz` and `normalize_std.npz` but doesn't document them. REDI sidecar should.

For each intermediate file:
1. **Detect file type** (NetCDF, NPZ, etc.)
2. **Compute statistics:**
   - Shape, dtype, memory footprint
   - Per-variable: min, max, mean, std, null count
   - For climate: lat/lon bounds, time range, pressure levels
3. **Track provenance:** which rule created it, from which inputs

```python
@dataclass
class DataSnapshot:
    path: str
    rule_name: str
    timestamp: datetime
    shape: tuple
    dtype: str
    variables: Dict[str, VariableStats]
    provenance: List[str]  # parent files
```

---

## Phase 3: Transformation Parameter Capture

**The ClimaX normalization problem:** Uses law of total variance across years, stores in separate files. An AI practitioner needs to know:
- What mean/std values were used?
- Were they computed before or after splitting?
- What's the target distribution?

**Sidecar should automatically detect and record:**

| Transformation | Parameters to Capture |
|---------------|----------------------|
| Normalization | mean, std, method (z-score/minmax), computed_on |
| Regridding | source_resolution, target_resolution, method |
| Unit conversion | conversion_factor, formula |
| Temporal sharding | shard_size, time_boundaries |
| Train/test split | split_ratio, split_method (random/temporal) |

---

## Phase 4: Data Quality Checks (at each stage)

**Automatically flag:**
- Distribution shift between stages (KL divergence)
- Missing values introduced
- Outliers created by transformations
- Shape mismatches
- NaN/Inf values
- Potential data leakage (stats computed before split)

```python
class QualityCheck:
    def check_leakage(self, snapshots: List[DataSnapshot]) -> List[Warning]:
        """Detect if normalization stats were computed on full data before split"""

    def check_distribution_shift(self, before: DataSnapshot, after: DataSnapshot) -> float:
        """KL divergence between distributions"""
```

---

## Phase 5: Comprehensive Data Card Generation

**Current output:** Basic JSON with rule names
**Proposed output:** Full Data Card with:

```markdown
# Data Card: TaiESM1 Climate Data

## Provenance
- **Source:** ESGF (AS-RCEC/TaiESM1/historical)
- **Downloaded:** 2026-01-27
- **Pipeline:** Snakemake (4 rules, 3 stages)

## Transformations Applied
| Stage | Rule | Input Shape | Output Shape | Parameters |
|-------|------|-------------|--------------|------------|
| Download | download | - | (730, 13, 128, 256) | wget from ESGF |
| Regrid | regrid | (730, 13, 128, 256) | (730, 13, 128, 256) | bilinear, 1.40625° |
| Finalize | delete | - | - | rename .tmp → .nc |

## Normalization
- **Method:** Z-score (cross-year aggregated)
- **Mean:** normalize_mean.npz (per-variable)
- **Std:** normalize_std.npz (per-variable)
- **Computed on:** Full training set (no leakage detected)

## Variables
| Variable | Shape | Min | Max | Mean | Std |
|----------|-------|-----|-----|------|-----|
| temperature_850 | (730, 128, 256) | 210.3 | 315.2 | 273.1 | 18.4 |
| ...

## Quality Checks
- ✓ No missing values
- ✓ No NaN/Inf introduced
- ✓ Distribution preserved within 2σ
- ⚠ Large outlier in temperature_50 (value: 450K)

## Reproducibility
- **Snakefile hash:** abc123...
- **Config hash:** def456...
- **Dependencies:** xesmf==0.8.2, xarray==2024.1.0
```

---

## Phase 6: Integration Points

**Make it zero-friction:**

### 1. Snakemake plugin
```python
# In Snakefile
from redi.sidecar import observe

onstart: observe.start()
onsuccess: observe.finish()
onerror: observe.error()
```

### 2. Log parser (for existing workflows)
```bash
snakemake 2>&1 | tee snakemake.log
redi sidecar --from-log snakemake.log --data-dir ./data/
```

### 3. Watch mode (FS events)
```bash
redi sidecar --watch ./data/ --workflow Snakefile
# Monitors file creations, correlates with rules
```

---

## Implementation Priority

| Priority | Feature | Impact | Effort |
|----------|---------|--------|--------|
| 1 | Data snapshot capture | High | Medium |
| 2 | Transformation parameter extraction | High | Medium |
| 3 | Quality checks | High | Low |
| 4 | Rich Data Card generation | High | Low |
| 5 | Snakemake log parsing | Medium | Medium |
| 6 | Runtime hooks | Medium | High |
| 7 | Watch mode | Low | High |

---

## Why This Matters

For an AI practitioner using ClimaX:

**Before:** "I ran the Snakemake pipeline. The data is in `./output/`. I think it's normalized somehow."

**After:** "I ran the pipeline with REDI sidecar. I have a Data Card showing:
- Exact normalization parameters (mean=273.1K, std=18.4K for temperature_850)
- No data leakage detected (normalization computed after temporal split)
- Distribution is preserved (KL divergence < 0.01)
- Ready for training with these stats files"

---

## Implementation Notes

### Starting Point: Data Snapshot Module

Create `redi/sidecar/snapshot.py`:
- `DataSnapshot` dataclass
- `VariableStats` dataclass
- `capture_snapshot(path: Path) -> DataSnapshot` - auto-detect format and compute stats
- Support for: NetCDF, NPZ, NPY, Parquet, CSV, HDF5, Zarr

### Key Files to Create/Modify

1. `redi/sidecar/__init__.py` - Module init
2. `redi/sidecar/snapshot.py` - Data snapshot capture
3. `redi/sidecar/quality.py` - Quality checks
4. `redi/sidecar/datacard.py` - Data Card generation
5. `redi/sidecar/snakemake.py` - Snakemake integration
6. `redi/cli.py` - Update CLI to use new sidecar functionality

### Test Data

Use the TaiESM1 workflow at:
`/home/w1b/drai/domains/climate/ClimaX/snakemake_configs/TaiESM1/`

This has real intermediate files we can use to test snapshot capture.
