# REDI CLI Refactoring - Session Context

**Date:** 2026-01-20

## Background Discussion

### Original Vision for REDI Modes

REDI was conceived to have different operational modes:

1. **Orchestrator mode** - For domains without established pipelines, REDI owns and runs the full pipeline
2. **Sidecar mode** - For domains with established workflows (like ClimaX's Snakemake), REDI observes and provides consistent data readiness tracking without replacing the existing workflow
3. **Discovery mode** - For unknown datasets, REDI inspects, infers characteristics, and proposes pipelines

### The Problem

During implementation, we realized:
- `sidecar` as implemented just parsed Snakefiles statically - not very useful
- `discover` and `analyze` (the old `--analyze` flag) were confusingly similar
- Having three modes plus the old `--analyze` was too complex

### Simplified Two-Command Model (AGREED)

```
redi analyze <target>    # Understand what you have
redi run <data>          # Process data through pipeline
```

Where `analyze` is smart about input type:

```bash
# Analyze a dataset → Data Card, readiness level, recommendations
redi analyze data.csv
redi analyze /path/to/dataset/

# Analyze a workflow → stage mapping, gap analysis, REDI suggestions
redi analyze Snakefile
redi analyze pipeline.nf
```

### Key Design Decisions

1. **No separate `discover`** - it's just `analyze` on unknown data
2. **No separate `sidecar`** - it's just `analyze` on a workflow file
3. **YAML auto-detection** - no `-c` flag needed, auto-detect `.yaml`/`.yml` files
4. **Backward compatible** - `redi data.csv` still works (defaults to `run`)

### Workflow Analysis Output (Proposed)

```
▸ Workflow Analysis: Snakefile

  Detected stages:
    ✓ ingest: download
    ✓ preprocess: regrid
    ✗ transform: (none detected)
    ✗ structure: (none detected)
    ✓ output: all

  Gaps identified:
    • No normalization step detected
    • No train/val/test split

  Suggestions:
    • Add REDI normalize step after regrid
    • Add REDI split step before final output
```

## What Was Implemented (NEEDS REVISION)

The CLI was refactored with three subcommands (run, sidecar, discover), but this needs to be simplified to just two (analyze, run).

### Files Created/Modified

```
redi/
├── redi/cli.py                              # NEEDS REVISION - simplify to analyze+run
├── README.md                                # NEEDS REVISION - update for new model
└── examples/
    ├── sidecar_config.yaml                  # Rename to analyze_workflow_config.yaml?
    ├── sidecar_climax_example.yaml          # Rename to analyze_climax_example.yaml?
    ├── discover_config.yaml                 # Rename to analyze_data_config.yaml?
    └── discover_hep_example.yaml            # Rename to analyze_hep_example.yaml?
```

## TODO for Next Session

### 1. Refactor CLI to Two-Command Model

```python
# Main commands
redi analyze <target>   # data file, directory, or workflow file
redi run <data>         # process data

# Auto-detect input type in analyze:
# - .yaml/.yml → load as config
# - Snakefile, *.smk, *.nf → workflow analysis
# - data files/directories → Data Card generation
```

### 2. Implement Smart Input Detection in `analyze`

```python
def detect_input_type(path):
    """Detect if input is a workflow, data file, or config."""
    if path.suffix in ['.yaml', '.yml']:
        return 'config'
    elif path.name.lower() in ['snakefile'] or path.suffix in ['.smk', '.nf']:
        return 'workflow'
    else:
        return 'data'
```

### 3. YAML Auto-Detection for `run`

Remove need for `-c` flag:
```bash
redi run config.yaml           # auto-detected as config
redi run data.csv              # auto-detected as data
redi run data.csv config.yaml  # data + config
```

### 4. Rename/Update Example Files

| Old Name | New Name |
|----------|----------|
| sidecar_config.yaml | analyze_workflow_config.yaml |
| sidecar_climax_example.yaml | analyze_climax_workflow.yaml |
| discover_config.yaml | analyze_data_config.yaml |
| discover_hep_example.yaml | analyze_hep_data.yaml |

### 5. Update README.md

Simplify to show two-command model with smart input detection.

## Reference: ClimaX Snakefile Location

The ClimaX Snakefile used for testing:
```
/home/w1b/drai/climate/ClimaX/snakemake_configs/TaiESM1/Snakefile
```

Rules: download → regrid → delete → all

## Reference: Example Unknown Dataset

Zenodo HEP dataset discussed for discovery use case:
https://zenodo.org/records/17525207

- sPHENIX TPC detector data (high-energy physics)
- .npy files, metadata CSV
- 90% sparse data
- Good test case for `analyze` on unknown domain
