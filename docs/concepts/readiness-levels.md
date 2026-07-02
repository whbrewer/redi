# Data Readiness Levels

REDI uses a five-level taxonomy to characterize the AI-readiness of a dataset, defined
in [`redi.datacard.MaturityLevel`](../../redi/datacard.py).

| Level | Name | Description |
|---|---|---|
| 1 | `RAW` | Unprocessed data as originally collected. No cleaning, normalization, or quality checks have been applied. |
| 2 | `CLEANED` | Missing values handled, duplicates removed, obvious outliers addressed, and basic quality checks passed. |
| 3 | `LABELED` | Target variables defined and attached. The dataset has a clear supervised learning objective. |
| 4 | `FEATURE_ENGINEERED` | Domain-specific features extracted and encoded (e.g., categorical encoding, physics-derived quantities, graph representations). |
| 5 | `FULLY_READY` | Fully AI-ready: data is sharded, normalized, saved in an optimized binary format (NPZ, HDF5, Parquet), and split into train/val/test sets. |

> **Note:** The enum member for level 5 is `FULLY_READY` (not `FULLY_AI_READY`).
> See [`redi.datacard.MaturityLevel`](../../redi/datacard.py) for the authoritative definition.

## Reaching level 5

Running the full REDI pipeline (all five stages) on a supported domain will
typically advance the dataset to level 5. You can inspect the current level
by reading the `data_card.json` written to the output directory after each
pipeline run.
