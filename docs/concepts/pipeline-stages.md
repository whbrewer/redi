# Pipeline Stages

The REDI pipeline consists of five sequential stages. You can run a subset of
stages with the `--stages` flag; prerequisites are included automatically:

```bash
redi run data.csv --stages ingest,preprocess   # runs ingest then preprocess
redi run data.csv --stages output              # runs all five stages
```

## Stage overview

| # | Stage | Description |
|---|---|---|
| 1 | `ingest` | Load data from disk. Detect file format (CSV, Parquet, NetCDF, HDF5, …), extract metadata, and build the initial `redi.datacard.DataCard`. |
| 2 | `preprocess` | Clean the dataset: handle missing values, remove duplicates, detect and flag outliers, normalise types. Advances the dataset to `CLEANED` level. |
| 3 | `transform` | Normalise numeric features, one-hot encode categoricals, apply domain-specific feature engineering (e.g. physics-derived quantities, spectral features, graph construction). Advances to `FEATURE_ENGINEERED`. |
| 4 | `structure` | Convert processed data into the target representation for ML training: tensors, graphs, sequences, or tabular arrays. Applies train/val/test splitting and optional sharding. |
| 5 | `output` | Serialise the structured dataset to the selected output format (NPZ, HDF5, or Parquet) and write the final `redi.datacard.DataCard` as `data_card.json`. Advances the dataset to `FULLY_READY`. |

## Module reference

Each stage is implemented as a module under `redi/pipeline/`:

* [`redi.pipeline.ingest`](../../redi/pipeline/ingest.py)
* [`redi.pipeline.preprocess`](../../redi/pipeline/preprocess.py)
* [`redi.pipeline.transform`](../../redi/pipeline/transform.py)
* [`redi.pipeline.structure`](../../redi/pipeline/structure.py)
* [`redi.pipeline.output`](../../redi/pipeline/output.py)

The pipeline orchestrator ([`redi.pipeline`](../../redi/pipeline/__init__.py)) selects the appropriate
domain-specific processor for each stage and routes data through the five steps.

## Output formats

The `--format` flag accepts: `npz`, `parquet`, `csv`, `hdf5`, `zarr`.

Full end-to-end pipeline implementations exist for **NPZ**, **HDF5**, and
**Parquet**. The `csv` and `zarr` options are accepted by the CLI but may
not be fully implemented for all domain processors.
