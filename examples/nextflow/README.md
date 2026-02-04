# REDI Sidecar Nextflow Examples

This directory contains example Nextflow workflows and sample data to demonstrate
the REDI sidecar functionality for Nextflow pipelines.

## Quick Start

```bash
# Activate the REDI environment
source /opt/venvs/drai/bin/activate

# Run sidecar on the simple workflow
redi sidecar --workflow simple_workflow.nf -o ./output

# Run sidecar with data scanning
redi sidecar --workflow simple_workflow.nf --data-dir ./data -o ./output

# Run sidecar on the bioinformatics workflow
redi sidecar --workflow bioinformatics_pipeline.nf --data-dir ./data -o ./output
```

## Example Workflows

### 1. simple_workflow.nf

A minimal Nextflow DSL2 workflow demonstrating basic concepts:
- Single process with input/output
- Parameter definitions
- Basic script execution

### 2. bioinformatics_pipeline.nf

A realistic bioinformatics pipeline with:
- Multiple processes (QC, alignment, quantification, reporting)
- nf-core style container definitions
- Tuple inputs/outputs with emit directives
- Workflow composition

### 3. climate_processing.nf

A climate data processing pipeline with:
- NetCDF file handling
- Regridding and normalization
- Train/val/test splitting
- Multiple output formats

## Sample Data

The `data/` directory contains sample files for testing:

- `sequences.fasta` - Sample protein sequences
- `sample_data.csv` - Tabular data with numeric features
- `normalize_mean.npz` - Example normalization statistics
- `train/`, `val/`, `test/` - Example split directories

## Output

After running sidecar, check the `output/` directory for:

- `sidecar_report.json` - Detailed JSON report with:
  - Process definitions
  - Stage mappings
  - Data snapshots (if --data-dir used)
  - Transformation parameters

- `DATA_CARD.md` - Human-readable Data Card with:
  - Workflow overview
  - Process table
  - Parameter documentation
  - Quality summary

## What REDI Sidecar Captures

For Nextflow workflows, REDI sidecar extracts:

| Feature | Description |
|---------|-------------|
| Processes | Name, inputs, outputs, containers, scripts |
| Workflows | Named workflows, process calls, take/emit |
| Includes | Module imports with paths |
| Parameters | params.* definitions |
| Stage Mapping | Maps processes to REDI stages (ingest/preprocess/transform/structure/output) |

When `--data-dir` is provided, it also captures:
- Data file statistics (shape, dtype, min/max/mean/std)
- Quality checks (NaN, outliers, distribution)
- Transformation parameters (normalization, splits, regrid)

## Tips

1. **Watch Mode**: Monitor workflow execution in real-time:
   ```bash
   redi sidecar --workflow pipeline.nf --data-dir ./work --watch
   ```

2. **JSON Output**: For programmatic use:
   ```bash
   redi sidecar --workflow pipeline.nf --style json
   ```

3. **nf-core Pipelines**: Works with nf-core modules and pipelines:
   ```bash
   redi sidecar --workflow main.nf --data-dir ./results
   ```
