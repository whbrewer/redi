"""
REDI sPHENIX Nuclear Physics Domain Module

Preprocesses sPHENIX TPC detector wedge data for AI-based compression training.

Input layout (the canonical Zenodo dataset)::

    outer/
        train.txt          # relative .npy paths (25,152 wedge files)
        test.txt           # relative .npy paths  (6,288 wedge files)
        <wedge_dir>/       # e.g. 12-2_0-0/
            *.npy          # uint16 arrays, shape (192, 249, 16)
                           # axes: azimuth x beam x layer

Output layout (compatible with NeuralCompression_v3 DatasetTPC)::

    <output_dir>/
        train/
            <wedge>.npy    # float32, log2(x+1)-transformed
            ...
        test/
            <wedge>.npy
            ...
        train.txt          # relative paths -- ready for DatasetTPC
        test.txt
        checksums.sha256
        metadata.json

NeuralCompression_v3 usage::

    from neuralcompress_v3.datasets.dataset_tpc import DatasetTPC
    ds = DatasetTPC(data_root='<output_dir>', split='train', dimension=2)

Set ``log: false`` in the model YAML config — REDI already applies log2(ADC+1).
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ADC noise floor: values strictly below this threshold are zero-suppressed
# in the sPHENIX simulation (matches the dataset description).
TPC_ZERO_SUPPRESS_THRESHOLD = 64


def get_sphenix_pipeline():
    """Return the complete sPHENIX TPC preprocessing pipeline."""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext
    from ..pipeline.output import WriteChecksums, SaveMetadata

    # ------------------------------------------------------------------
    # INGEST
    # ------------------------------------------------------------------

    class LoadSPHENIXManifest(PipelineStep):
        """
        Read the canonical train.txt / test.txt manifests and record the list
        of absolute .npy paths for each split.

        ctx.data after this step:
            {
                "train_files": [Path, ...],   # 25,152 entries
                "test_files":  [Path, ...],   #  6,288 entries
            }
        """

        name = "load_sphenix_manifest"
        stage = PipelineStage.INGEST
        description = "Read sPHENIX train/test manifests"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            base_dir = Path(ctx.input_path)
            if not base_dir.is_dir():
                ctx.add_error(f"sPHENIX input must be a directory, got: {base_dir}")
                return ctx

            splits = {}
            for split in ("train", "test"):
                manifest = base_dir / f"{split}.txt"
                if not manifest.exists():
                    ctx.add_error(f"Missing manifest: {manifest}")
                    return ctx

                lines = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
                paths = [base_dir / ln for ln in lines]
                missing = [p for p in paths if not p.exists()]
                if missing:
                    logger.warning(f"{split}: {len(missing)} files in manifest not found on disk")
                splits[split] = [p for p in paths if p.exists()]

            ctx.data = {
                "train_files": splits["train"],
                "test_files": splits["test"],
            }
            ctx.metadata["num_train_wedges"] = len(splits["train"])
            ctx.metadata["num_test_wedges"] = len(splits["test"])
            ctx.metadata["num_wedges_total"] = len(splits["train"]) + len(splits["test"])
            logger.info(
                f"sPHENIX manifest: {len(splits['train'])} train + "
                f"{len(splits['test'])} test wedge files"
            )
            return ctx

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    class SaveTPCNpy(PipelineStep):
        """
        Stream each wedge .npy through the preprocessing transform and write
        one output .npy per wedge.  Also writes train.txt / test.txt manifests
        compatible with NeuralCompression_v3's DatasetTPC loader.

        Transform per array (uint16 → float32):
          1. Zero-suppression: ADC < 64 → 0.0
          2. log2(ADC + 1)   (NeuralCompression model's standard normalisation)

        Set `log: false` in your NeuralCompression YAML — REDI already applies it.
        """

        name = "save_tpc_npy"
        stage = PipelineStage.OUTPUT
        description = "Save per-wedge .npy files + DatasetTPC manifests"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            import numpy as np

            output_dir = ctx.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            saved_files: List[str] = []
            manifest_lines = {"train": [], "test": []}

            for split in ("train", "test"):
                split_dir = output_dir / split
                split_dir.mkdir(exist_ok=True)

                files: List[Path] = ctx.data.get(f"{split}_files", [])
                logger.info(f"Writing {len(files)} {split} wedge files …")

                for i, src in enumerate(files):
                    arr = np.load(src).astype(np.float32)

                    # Zero-suppress noise floor
                    arr[arr < TPC_ZERO_SUPPRESS_THRESHOLD] = 0.0

                    # log2(ADC + 1)  →  [0, ~10]
                    arr = np.log2(arr + 1.0)

                    # Preserve wedge identity in the filename
                    out_name = f"{src.parent.name}_{src.name}"
                    out_path = split_dir / out_name
                    np.save(out_path, arr)

                    rel = f"{split}/{out_name}"
                    manifest_lines[split].append(rel)
                    saved_files.append(str(out_path))

                    if (i + 1) % 2000 == 0:
                        logger.info(f"  {split}: {i + 1}/{len(files)}")

                manifest_path = output_dir / f"{split}.txt"
                manifest_path.write_text("\n".join(manifest_lines[split]) + "\n")
                logger.info(f"Wrote {manifest_path.name} ({len(manifest_lines[split])} entries)")

            ctx.metadata["saved_files"] = saved_files
            ctx.metadata["output_format"] = "tpc_npy"
            ctx.metadata["transform"] = "zero_suppress(threshold=64) + log2(ADC+1)"
            logger.info(f"Saved {len(saved_files)} TPC wedge files to {output_dir}")
            return ctx

    # ------------------------------------------------------------------

    return [
        LoadSPHENIXManifest(),
        SaveTPCNpy(),
        WriteChecksums(),
        SaveMetadata(),
    ]
