# SC26 Reproducibility Guide

Reproduces all REDI results from the paper.
All commands assume the REDI repository root as the working directory.

```bash
source /opt/venvs/drai/bin/activate
cd /home/w1b/drai/apps/redi
```

---

## 1. ClimaX — Climate (TaiESM1 CMIP6)

**Data:** `/opt/drai/datasets/CMIP6/TaiESM1/raw/`
**Ground truth:** `/opt/drai/datasets/CMIP6/TaiESM1/1.40625deg/`

### Run pipeline (single chunk)
```bash
redi run /opt/drai/datasets/CMIP6/TaiESM1/raw/2m_temperature/2m_temperature_185001010000-186001010000_raw.nc \
    -c examples/climate_config.yaml \
    -o ./output/taiesm1_climax
```

### Run + validate all 2m_temperature chunks
```bash
bash scripts/validate_climate_2m_all.sh
# Results: validate_results/taiesm1_2m_temp_<chunk>_validate.json
#          validate_results/taiesm1_2m_temp_all_validate.json
```

### Run + validate 3D variables (temperature, u-wind, v-wind)
```bash
bash scripts/validate_climate_3d.sh
# Results: validate_results/taiesm1_temperature_validate.json
#          validate_results/taiesm1_u_component_of_wind_validate.json
#          validate_results/taiesm1_v_component_of_wind_validate.json
```

### Timing benchmark
```bash
python scripts/bench_climate_timing.py
# Results: validate_results/bench_climate_timing.json
```

### AIDRIN data quality
```bash
python scripts/run_aidrin_climate.py
# Results: validate_results/aidrin_climate.json
```

---

## 2. OpenFold — Proteomics (200 PDB structures)

**Data:** `/opt/drai/openfold_data/`
(200 proteins: `fasta/`, `pdb/`, `alignments/` — see `download_openfold_minimal_dataset.sh`)

### Run pipeline
```bash
redi run /opt/drai/openfold_data/ --domain life_sciences -o ./output/openfold_ready
# or using the config file:
redi run /opt/drai/openfold_data/ -c examples/openfold_config.yaml
```

### Validate against OpenFold's own pipeline
```bash
python scripts/validate_openfold.py \
    --data_dir /opt/drai/openfold_data \
    --redi_dir ./output/openfold_ready \
    --out validate_results/openfold_validate.json
# Results: validate_results/openfold_validate.json
# Expected: 11/11 features match for all proteins with valid MSA
```

### AIDRIN data quality
```bash
python scripts/run_aidrin_openfold.py
# Results: validate_results/aidrin_openfold.json
```

---

## 3. HydraGNN — Materials Science (AFLOW DFT)

**Data:** `/opt/drai/aflow/aflow_ml_ready.csv`

### Run pipeline
```bash
redi run /opt/drai/aflow/aflow_ml_ready.csv --domain materials -o ./output/aflow_ready
# or using the config file:
redi run -c examples/aflow_config.yaml
```

### Validate
```bash
python scripts/validate_materials.py
# Results: validate_results/aflow_validate.json
```

### AIDRIN data quality
```bash
python scripts/run_aidrin_materials.py
# Results: validate_results/aidrin_materials.json
```

---

## 4. sPHENIX — Nuclear Physics (TPC detector hits)

**Data:** `/opt/drai/sphenix/outer/`
(31,440 wedge files in 24 subdirs; canonical `train.txt` / `test.txt` manifests included)

### Run pipeline
```bash
redi run /opt/drai/sphenix/outer/ --domain nuclear_physics -o ./output/sphenix_ready
# or using the config file:
redi run /opt/drai/sphenix/outer/ -c examples/sphenix_config.yaml
```

Output is directly compatible with NeuralCompression_v3's `DatasetTPC` loader.
Set `log: false` in the model YAML — REDI already applies log2(ADC+1).

### Validate
```bash
python scripts/validate_sphenix.py
# Results: validate_results/sphenix_validate.json
```

### AIDRIN data quality
```bash
python scripts/run_aidrin_sphenix.py
# Results: validate_results/aidrin_sphenix.json
```

### Timing benchmark
```bash
python scripts/bench_sphenix_timing.py
# Results: validate_results/bench_sphenix_timing.json
```

---

## 5. XGC1 — Fusion (KSTAR particle-mesh)

**Data:** `/opt/drai/xgc-fusion-data/KSTART/`

### Run pipeline
```bash
redi run -c examples/fusion/xgc_kstar.yaml
# or with 3D fields:
redi run -c examples/fusion/xgc_kstar_f3d.yaml
```

---

## 6. ExaDigiT — HPC Telemetry (10 systems)

**Data:** `/opt/data/<system>/` (see individual configs in `examples/hpc/`)
**Requires:** RAPS venv for ground-truth generation

```bash
source /opt/venvs/exadigit/bin/activate   # for raps
source /opt/venvs/drai/bin/activate       # for redi
```

### Run + validate all 10 systems
```bash
bash scripts/run_hpc_validate.sh
# Results: validate_results/<system>_validate.json
#          validate_results/summary.json
```

### Run a single system (e.g. Frontier)
```bash
redi run -c examples/hpc/frontier.yaml
redi validate output/frontier_raps \
    --ground-truth validate_results/frontier_raps_gt/snapshot.npz \
    --style json > validate_results/frontier_validate.json
```

### Timing benchmark
```bash
python scripts/gen_hpc_timing.py
# Results: validate_results/bench_redi_timing.json
#          validate_results/bench_raps_timing2.json
```

### AIDRIN data quality (Frontier)
```bash
python scripts/run_aidrin_hpc.py
# Results: validate_results/aidrin_hpc_analysis.json
```

### All systems supported
`frontier` · `aurora` · `adastra_mi250` · `fugaku` · `marconi100` ·
`mit_supercloud` · `bluewaters` · `lassen` · `philly` · `gcloud_v2`

---

## Summary of result files

| Use Case | Validate | AIDRIN | Timing |
|---|---|---|---|
| ClimaX | `taiesm1_*_validate.json` | `aidrin_climate.json` | `bench_climate_timing.json` |
| OpenFold | `openfold_validate.json` | `aidrin_openfold.json` | `bench_openfold_timing.json` |
| HydraGNN | `aflow_validate.json` | `aidrin_materials.json` | `bench_aflow_timing.json` |
| sPHENIX | `sphenix_validate.json` | `aidrin_sphenix.json` | `bench_sphenix_timing.json` |
| XGC1 | — | — | — |
| ExaDigiT | `<system>_validate.json` | `aidrin_hpc_analysis.json` | `bench_redi_timing.json` |

All result files are written to `validate_results/`.
