"""
Hugging Face Hub exporter for REDI.

Exports REDI-processed datasets to Hugging Face Hub, making them
discoverable and downloadable via the datasets library.

Requirements:
    pip install datasets huggingface_hub

Example:
    from redi.catalog.huggingface import HuggingFaceExporter

    exporter = HuggingFaceExporter()
    exporter.configure(
        repo_id="your-org/dataset-name",
        token="hf_...",
        private=True,
    )

    result = exporter.export("output/metadata.json")
    print(f"Dataset URL: {result.entity_url}")
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    CatalogExporter,
    CatalogExportResult,
    CatalogType,
    REDIMetadata,
)

logger = logging.getLogger(__name__)


class HuggingFaceExporter(CatalogExporter):
    """Export REDI datasets to Hugging Face Hub.

    Creates a dataset repository on Hugging Face Hub with:
    - Train/val/test splits as separate subsets
    - Dataset card generated from REDI metadata
    - Automatic Parquet conversion for efficient storage
    - Metadata tags for discoverability

    Attributes:
        catalog_type: CatalogType.HUGGINGFACE
    """

    catalog_type = CatalogType.HUGGINGFACE

    def __init__(self):
        """Initialize the Hugging Face exporter."""
        super().__init__()

    def validate_connection(self) -> bool:
        """Validate Hugging Face Hub authentication.

        Returns:
            True if token is valid and can access the Hub
        """
        try:
            from huggingface_hub import HfApi

            token = self.config.token
            if not token:
                return False

            api = HfApi(token=token)
            # Try to get user info to validate token
            user_info = api.whoami()
            self._log(f"Authenticated as: {user_info.get('name', 'unknown')}")
            return True

        except Exception as e:
            self._log(f"Authentication failed: {e}", "error")
            return False

    def export(self, metadata_path: str) -> CatalogExportResult:
        """Export REDI dataset to Hugging Face Hub.

        Args:
            metadata_path: Path to REDI metadata.json file

        Returns:
            CatalogExportResult with Hub URL and dataset info
        """
        try:
            # Check dependencies
            try:
                from datasets import Dataset, DatasetDict
                from huggingface_hub import HfApi
            except ImportError:
                return self._create_error_result(
                    "Hugging Face libraries not installed. Install with:\n"
                    "  pip install datasets huggingface_hub"
                )

            # Load REDI metadata
            redi_meta = REDIMetadata.from_file(metadata_path)
            self._log(f"Loaded metadata for: {redi_meta.name}")

            # Get configuration
            repo_id = self.config.extra.get("repo_id")
            if not repo_id:
                # Generate repo_id from dataset name
                repo_id = redi_meta.name.lower().replace(" ", "-")
                self._log(f"Using generated repo_id: {repo_id}")

            token = self.config.token
            if not token:
                import os
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

            if not token:
                return self._create_error_result(
                    "Hugging Face token not configured. Set HF_TOKEN env var or use --catalog-token"
                )

            # Auto-prefix username if repo_id doesn't contain one
            if "/" not in repo_id:
                try:
                    api = HfApi(token=token)
                    username = api.whoami()["name"]
                    repo_id = f"{username}/{repo_id}"
                    self._log(f"Auto-prefixed username: {repo_id}")
                except Exception as e:
                    self._log(f"Could not auto-detect username: {e}")
                    return self._create_error_result(
                        f"repo_id must be in format 'username/dataset-name'. Got: {repo_id}"
                    )

            private = self.config.extra.get("private", True)

            # Find output directory from metadata
            output_dir = Path(metadata_path).parent

            # Load data files and create DatasetDict
            dataset_dict = self._load_redi_outputs(output_dir, redi_meta)

            if not dataset_dict:
                return self._create_error_result(
                    f"No data files found in {output_dir}"
                )

            # Generate dataset card
            dataset_card = self._generate_dataset_card(redi_meta, repo_id)

            # Push to Hub
            self._log(f"Pushing to Hub: {repo_id}")
            dataset_dict.push_to_hub(
                repo_id,
                token=token,
                private=private,
            )

            # Upload dataset card (README.md)
            api = HfApi(token=token)
            api.upload_file(
                path_or_fileobj=dataset_card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )

            # Build result
            entity_url = f"https://huggingface.co/datasets/{repo_id}"

            return CatalogExportResult(
                success=True,
                catalog_type="huggingface",
                entity_id=repo_id,
                entity_url=entity_url,
                message=f"Successfully pushed dataset to {entity_url}",
                metadata={
                    "repo_id": repo_id,
                    "private": private,
                    "splits": list(dataset_dict.keys()),
                    "num_rows": {k: len(v) for k, v in dataset_dict.items()},
                },
            )

        except Exception as e:
            self._log(f"Export failed: {e}", "error")
            return self._create_error_result(f"Export failed: {str(e)}")

    def _load_redi_outputs(self, output_dir: Path, redi_meta: REDIMetadata) -> Optional[Dict]:
        """Load REDI output files into a DatasetDict.

        Args:
            output_dir: Directory containing REDI outputs
            redi_meta: REDI metadata

        Returns:
            DatasetDict with train/val/test splits, or None if no files found
        """
        from datasets import Dataset, DatasetDict
        import numpy as np

        datasets = {}
        output_format = redi_meta.output_format or "npz"

        # Try to load each split
        for split in ["train", "val", "test"]:
            data = None

            # Try different formats
            if output_format == "npz":
                npz_path = output_dir / f"{split}.npz"
                if npz_path.exists():
                    data = self._load_npz(npz_path)

            elif output_format == "parquet":
                parquet_path = output_dir / f"{split}.parquet"
                if parquet_path.exists():
                    data = self._load_parquet(parquet_path)

            elif output_format == "csv":
                csv_path = output_dir / f"{split}.csv"
                if csv_path.exists():
                    data = self._load_csv(csv_path)

            elif output_format == "hdf5":
                h5_path = output_dir / "data.h5"
                if h5_path.exists():
                    data = self._load_hdf5_split(h5_path, split)

            # Also try common patterns if format-specific didn't work
            if data is None:
                for ext, loader in [
                    (".parquet", self._load_parquet),
                    (".csv", self._load_csv),
                    (".npz", self._load_npz),
                ]:
                    path = output_dir / f"{split}{ext}"
                    if path.exists():
                        data = loader(path)
                        break

            if data is not None:
                datasets[split] = Dataset.from_dict(data)
                self._log(f"Loaded {split}: {len(datasets[split])} rows")

        if not datasets:
            return None

        return DatasetDict(datasets)

    def _load_npz(self, path: Path) -> Optional[Dict]:
        """Load NPZ file into dict format."""
        import numpy as np

        try:
            data = np.load(path, allow_pickle=True)
            result = {}

            if "X" in data:
                X = data["X"]
                # Get feature names if available
                feature_names = None
                if "feature_names" in data:
                    feature_names = list(data["feature_names"])
                elif "variable_names" in data:
                    feature_names = list(data["variable_names"])

                if feature_names and len(feature_names) == X.shape[1]:
                    for i, name in enumerate(feature_names):
                        result[name] = X[:, i].tolist()
                else:
                    # Use generic column names
                    for i in range(X.shape[1]):
                        result[f"feature_{i}"] = X[:, i].tolist()

            if "y" in data:
                result["target"] = data["y"].tolist()

            return result if result else None

        except Exception as e:
            self._log(f"Error loading NPZ {path}: {e}", "warning")
            return None

    def _load_parquet(self, path: Path) -> Optional[Dict]:
        """Load Parquet file into dict format."""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict(orient="list")
        except Exception as e:
            self._log(f"Error loading Parquet {path}: {e}", "warning")
            return None

    def _load_csv(self, path: Path) -> Optional[Dict]:
        """Load CSV file into dict format."""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict(orient="list")
        except Exception as e:
            self._log(f"Error loading CSV {path}: {e}", "warning")
            return None

    def _load_hdf5_split(self, path: Path, split: str) -> Optional[Dict]:
        """Load a split from HDF5 file."""
        try:
            import h5py
            import json

            with h5py.File(path, "r") as f:
                if split not in f:
                    return None

                grp = f[split]
                result = {}

                if "X" in grp:
                    X = grp["X"][:]
                    feature_names = None
                    if "feature_names" in grp.attrs:
                        feature_names = json.loads(grp.attrs["feature_names"])

                    if feature_names and len(feature_names) == X.shape[1]:
                        for i, name in enumerate(feature_names):
                            result[name] = X[:, i].tolist()
                    else:
                        for i in range(X.shape[1]):
                            result[f"feature_{i}"] = X[:, i].tolist()

                if "y" in grp:
                    result["target"] = grp["y"][:].tolist()

                return result if result else None

        except Exception as e:
            self._log(f"Error loading HDF5 {path}: {e}", "warning")
            return None

    def _generate_dataset_card(self, redi_meta: REDIMetadata, repo_id: str) -> str:
        """Generate a Hugging Face dataset card (README.md).

        Args:
            redi_meta: REDI metadata
            repo_id: Repository ID

        Returns:
            Markdown string for dataset card
        """
        # Build YAML frontmatter
        tags = [
            "redi",
            f"domain:{redi_meta.domain}",
            "ai-ready",
        ]

        yaml_header = f"""---
license: cc-by-4.0
task_categories:
  - tabular-classification
  - tabular-regression
tags:
{chr(10).join(f'  - {tag}' for tag in tags)}
size_categories:
  - {self._get_size_category(redi_meta.num_rows)}
---
"""

        # Build markdown content
        content = f"""# {redi_meta.name}

{redi_meta.description}

## Dataset Description

- **Domain:** {redi_meta.domain}
- **Source:** `{redi_meta.input_file}`
- **Processed by:** [REDI](https://github.com/your-org/redi) (Readiness Engine for Data Integration)

## Dataset Structure

### Data Splits

| Split | Samples |
|-------|---------|
| train | {redi_meta.train_size:,} |
| validation | {redi_meta.val_size:,} |
| test | {redi_meta.test_size:,} |

### Features

"""
        # Add feature list
        if redi_meta.feature_columns:
            content += f"This dataset contains {len(redi_meta.feature_columns)} features:\n\n"
            for col in redi_meta.feature_columns[:20]:
                content += f"- `{col}`\n"
            if len(redi_meta.feature_columns) > 20:
                content += f"- ... and {len(redi_meta.feature_columns) - 20} more\n"
        else:
            content += f"This dataset contains {redi_meta.num_columns} features.\n"

        if redi_meta.target_column:
            content += f"\n**Target variable:** `{redi_meta.target_column}`\n"

        content += """
## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access splits
train_data = dataset["train"]
val_data = dataset["val"]
test_data = dataset["test"]

# Convert to pandas
df = train_data.to_pandas()
```

## Processing Details

This dataset was processed using REDI (Readiness Engine for Data Integration),
which transforms raw scientific data into AI-ready formats.

### Preprocessing Applied

""".format(repo_id=repo_id)

        if redi_meta.completed_stages:
            content += "- " + "\n- ".join(redi_meta.completed_stages) + "\n"
        else:
            content += "- Standard REDI pipeline\n"

        if redi_meta.normalization_stats:
            content += "\n### Normalization\n\n"
            content += "Features have been normalized using z-score normalization.\n"

        content += """
## Citation

If you use this dataset, please cite:

```bibtex
@misc{redi_dataset,
  title={""" + redi_meta.name + """},
  author={REDI},
  year={2025},
  publisher={Hugging Face},
  howpublished={\\url{https://huggingface.co/datasets/""" + repo_id + """}}
}
```
"""

        return yaml_header + content

    def _get_size_category(self, num_rows: int) -> str:
        """Get HuggingFace size category string."""
        if num_rows < 1000:
            return "n<1K"
        elif num_rows < 10000:
            return "1K<n<10K"
        elif num_rows < 100000:
            return "10K<n<100K"
        elif num_rows < 1000000:
            return "100K<n<1M"
        elif num_rows < 10000000:
            return "1M<n<10M"
        else:
            return "n>10M"


def export_to_huggingface(
    metadata_path: str,
    repo_id: str,
    token: str = None,
    private: bool = True,
    **kwargs,
) -> CatalogExportResult:
    """Convenience function to export to Hugging Face Hub.

    Args:
        metadata_path: Path to REDI metadata.json
        repo_id: Hugging Face repository ID (e.g., "your-org/dataset-name")
        token: Hugging Face token (or set HF_TOKEN env var)
        private: Whether to make the dataset private
        **kwargs: Additional configuration options

    Returns:
        CatalogExportResult

    Example:
        result = export_to_huggingface(
            "output/metadata.json",
            repo_id="my-org/aurora-hpc",
            private=True,
        )
    """
    exporter = HuggingFaceExporter()
    exporter.configure(
        token=token or "",
        repo_id=repo_id,
        private=private,
        **kwargs,
    )
    return exporter.export(metadata_path)
