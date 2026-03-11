"""
DRAI Materials Science Domain Module

Domain-specific preprocessing for materials data.
Supports AFLOW, Materials Project, and structure files.

Tools: pymatgen, matminer, ase, torch_geometric
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = ["pymatgen", "matminer", "ase", "torch_geometric"]

# Common materials properties
MATERIALS_PROPERTIES = {
    "electronic": ["band_gap", "bandgap", "fermi_energy", "dos"],
    "thermodynamic": ["formation_energy", "enthalpy", "entropy", "heat_capacity"],
    "mechanical": ["bulk_modulus", "shear_modulus", "elastic_tensor"],
    "structural": ["volume", "density", "lattice", "spacegroup"],
}

# Magpie feature prefixes
MAGPIE_PREFIXES = [
    "MagpieData", "Magpie", "ElementProperty"
]


def get_ingest_steps():
    """Get materials-specific ingest steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext, missing_dependency_error
    from ..pipeline.ingest import DetectFormat

    class LoadMaterialsCSV(PipelineStep):
        """Load materials CSV with composition parsing"""

        name = "load_materials_csv"
        stage = PipelineStage.INGEST
        description = "Load materials CSV with composition parsing"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") == "csv" and ctx.domain == "materials"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                # Handle multi-file input
                paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

                if len(paths) == 1:
                    df = pd.read_csv(paths[0])
                else:
                    dfs = []
                    for i, path in enumerate(paths):
                        logger.debug(f"Loading file {i+1}/{len(paths)}: {path.name}")
                        dfs.append(pd.read_csv(path))
                    df = pd.concat(dfs, axis=0, ignore_index=True)
                    logger.info(f"Concatenated {len(paths)} files")

                ctx.data = df

                # Identify composition column
                comp_keywords = ["formula", "composition", "compound", "material"]
                for col in df.columns:
                    if any(kw in col.lower() for kw in comp_keywords):
                        ctx.metadata["composition_column"] = col
                        break

                # Identify target columns
                target_keywords = ["band_gap", "bandgap", "formation_energy", "energy"]
                for col in df.columns:
                    if any(kw in col.lower() for kw in target_keywords):
                        ctx.metadata.setdefault("target_columns", []).append(col)

                # Check for existing features (e.g., Magpie)
                magpie_cols = [c for c in df.columns if any(p in c for p in MAGPIE_PREFIXES)]
                if magpie_cols:
                    ctx.metadata["existing_features"] = "magpie"
                    ctx.metadata["n_magpie_features"] = len(magpie_cols)

                ctx.metadata["num_rows"] = len(df)
                ctx.metadata["num_columns"] = len(df.columns)
                logger.info(f"Loaded materials data: {len(df)} samples")

            except Exception as e:
                ctx.add_error(f"Error loading materials CSV: {str(e)}")

            return ctx

    class LoadStructureFile(PipelineStep):
        """Load crystal structure file (CIF, POSCAR)"""

        name = "load_structure"
        stage = PipelineStage.INGEST
        description = "Load crystal structure using pymatgen"

        def validate(self, ctx: PipelineContext) -> bool:
            return ctx.metadata.get("file_format") in ["cif", "poscar", "vasp"]

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                from pymatgen.core import Structure

                # Handle multi-file input
                paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

                if len(paths) == 1:
                    structure = Structure.from_file(paths[0])
                    ctx.data = structure
                    ctx.metadata["formula"] = structure.composition.reduced_formula
                    ctx.metadata["n_atoms"] = len(structure)
                    ctx.metadata["spacegroup"] = structure.get_space_group_info()
                else:
                    # Multiple structure files - load into list
                    structures = []
                    for i, path in enumerate(paths):
                        logger.debug(f"Loading structure {i+1}/{len(paths)}: {path.name}")
                        structures.append(Structure.from_file(path))
                    ctx.data = structures
                    ctx.metadata["n_structures"] = len(structures)
                    # Use first structure for metadata
                    structure = structures[0]
                    ctx.metadata["formula"] = structure.composition.reduced_formula
                    ctx.metadata["n_atoms"] = len(structure)
                    ctx.metadata["spacegroup"] = structure.get_space_group_info()
                    logger.info(f"Loaded {len(structures)} structure files")

                ctx.metadata["lattice"] = {
                    "a": structure.lattice.a,
                    "b": structure.lattice.b,
                    "c": structure.lattice.c,
                    "alpha": structure.lattice.alpha,
                    "beta": structure.lattice.beta,
                    "gamma": structure.lattice.gamma,
                    "volume": structure.volume
                }

                logger.info(f"Loaded structure: {ctx.metadata['formula']}")

            except ImportError:
                ctx.add_error(missing_dependency_error("pymatgen", "materials"))
            except Exception as e:
                ctx.add_error(f"Error loading structure: {str(e)}")

            return ctx

    return [DetectFormat(), LoadMaterialsCSV(), LoadStructureFile()]


def get_preprocess_steps():
    """Get materials-specific preprocess steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class CleanMaterialsData(PipelineStep):
        """Clean materials data (remove invalid compositions, handle missing)"""

        name = "clean_materials"
        stage = PipelineStage.PREPROCESS
        description = "Clean materials data"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                initial_count = len(df)

                # Remove rows with missing target values
                target_cols = ctx.metadata.get("target_columns", [])
                for col in target_cols:
                    if col in df.columns:
                        df = df.dropna(subset=[col])

                # Validate compositions if column exists
                comp_col = ctx.metadata.get("composition_column")
                if comp_col and comp_col in df.columns:
                    try:
                        from pymatgen.core import Composition

                        def is_valid_composition(formula):
                            try:
                                Composition(formula)
                                return True
                            except Exception:
                                return False

                        valid_mask = df[comp_col].apply(is_valid_composition)
                        df = df[valid_mask]
                    except ImportError:
                        pass

                ctx.data = df
                ctx.metadata["samples_removed"] = initial_count - len(df)
                logger.info(f"Cleaned data: {initial_count} -> {len(df)} samples")

            except Exception as e:
                ctx.add_error(f"Error cleaning materials data: {str(e)}")

            return ctx

    return [CleanMaterialsData()]


def get_transform_steps():
    """Get materials-specific transform steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class FeaturizeCompositions(PipelineStep):
        """Generate composition-based features using matminer"""

        name = "featurize_compositions"
        stage = PipelineStage.TRANSFORM
        description = "Generate Magpie/composition features"

        def __init__(self, featurizers: Optional[List[str]] = None):
            super().__init__()
            self.featurizers = featurizers or ["magpie"]

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                from pymatgen.core import Composition
                from matminer.featurizers.composition import ElementProperty

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                comp_col = ctx.metadata.get("composition_column")

                if not comp_col or comp_col not in df.columns:
                    logger.warning("No composition column found for featurization")
                    return ctx

                # Skip if already featurized
                if ctx.metadata.get("existing_features") == "magpie":
                    logger.info("Magpie features already present, skipping")
                    return ctx

                # Parse compositions
                df["composition_obj"] = df[comp_col].apply(
                    lambda x: Composition(x) if pd.notna(x) else None
                )

                # Apply Magpie featurizer
                if "magpie" in self.featurizers:
                    featurizer = ElementProperty.from_preset("magpie")
                    feature_labels = featurizer.feature_labels()

                    features = []
                    for comp in df["composition_obj"]:
                        if comp is not None:
                            try:
                                features.append(featurizer.featurize(comp))
                            except Exception:
                                features.append([None] * len(feature_labels))
                        else:
                            features.append([None] * len(feature_labels))

                    feature_df = pd.DataFrame(features, columns=feature_labels)
                    df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)

                # Clean up
                df = df.drop(columns=["composition_obj"], errors='ignore')
                ctx.data = df

                ctx.metadata["features_added"] = feature_labels
                logger.info(f"Added {len(feature_labels)} composition features")

            except ImportError as e:
                package = getattr(e, "name", None) or "pymatgen/matminer"
                ctx.add_error(missing_dependency_error(package, "materials"))
            except Exception as e:
                ctx.add_error(f"Error featurizing: {str(e)}")

            return ctx

    class NormalizeMaterialsFeatures(PipelineStep):
        """Normalize numerical features"""

        name = "normalize_materials"
        stage = PipelineStage.TRANSFORM
        description = "Normalize materials features"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data

                # Get numeric columns (exclude target and composition)
                target_cols = ctx.metadata.get("target_columns", [])
                comp_col = ctx.metadata.get("composition_column", "")

                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                feature_cols = [c for c in numeric_cols if c not in target_cols]

                # Z-score normalization
                stats = {}
                for col in feature_cols:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std
                    stats[col] = {"mean": float(mean), "std": float(std)}

                ctx.data = df
                ctx.metadata["normalization_stats"] = stats
                logger.info(f"Normalized {len(feature_cols)} features")

            except Exception as e:
                ctx.add_error(f"Error normalizing: {str(e)}")

            return ctx

    return [FeaturizeCompositions(), NormalizeMaterialsFeatures()]


def get_structure_steps():
    """Get materials-specific structure steps"""
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

    class CreateGraphFromStructure(PipelineStep):
        """Create graph representation of crystal structure"""

        name = "structure_to_graph"
        stage = PipelineStage.STRUCTURE
        description = "Convert crystal structure to graph"

        def __init__(self, cutoff: float = 5.0):
            super().__init__()
            self.cutoff = cutoff

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import numpy as np

                # Check if we have a pymatgen Structure
                try:
                    from pymatgen.core import Structure
                    if not isinstance(ctx.data, Structure):
                        return ctx
                except ImportError:
                    return ctx

                structure = ctx.data

                # Get node features (atomic numbers)
                atomic_numbers = np.array([site.specie.Z for site in structure])

                # Get edges from neighbor list
                neighbors = structure.get_all_neighbors(self.cutoff)
                edge_src = []
                edge_dst = []
                edge_attr = []  # distances

                for i, site_neighbors in enumerate(neighbors):
                    for neighbor in site_neighbors:
                        j = neighbor.index
                        dist = neighbor.nn_distance
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_attr.append(dist)

                ctx.data = {
                    "x": atomic_numbers.reshape(-1, 1),  # Node features
                    "edge_index": np.array([edge_src, edge_dst]),
                    "edge_attr": np.array(edge_attr).reshape(-1, 1),
                    "num_nodes": len(structure)
                }

                ctx.metadata["graph"] = {
                    "num_nodes": len(structure),
                    "num_edges": len(edge_src),
                    "cutoff": self.cutoff
                }
                logger.info(f"Created graph: {len(structure)} nodes, {len(edge_src)} edges")

            except Exception as e:
                ctx.add_error(f"Error creating graph: {str(e)}")

            return ctx

    class ToNumpyArrays(PipelineStep):
        """Convert DataFrame to numpy arrays for tabular ML"""

        name = "materials_to_numpy"
        stage = PipelineStage.STRUCTURE
        description = "Convert materials features to numpy arrays"

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(ctx.data, pd.DataFrame):
                    return ctx

                df = ctx.data
                target_cols = ctx.metadata.get("target_columns", [])
                comp_col = ctx.metadata.get("composition_column", "")

                # Get feature columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                feature_cols = [c for c in numeric_cols if c not in target_cols]

                X = df[feature_cols].values.astype(np.float32)

                result = {"X": X, "feature_names": feature_cols}

                # Add targets
                for target_col in target_cols:
                    if target_col in df.columns:
                        result["y"] = df[target_col].values.astype(np.float32)
                        break

                ctx.data = result
                ctx.metadata["array_shape"] = X.shape
                logger.info(f"Converted to arrays: X shape {X.shape}")

            except Exception as e:
                ctx.add_error(f"Error converting to numpy: {str(e)}")

            return ctx

    return [CreateGraphFromStructure(), ToNumpyArrays()]


def get_output_steps():
    """Get materials-specific output steps"""
    # Use standard output steps
    from ..pipeline.output import get_output_steps
    return get_output_steps(output_format="npz")


def get_full_pipeline():
    """Get the complete materials preprocessing pipeline"""
    return (
        get_ingest_steps() +
        get_preprocess_steps() +
        get_transform_steps() +
        get_structure_steps() +
        get_output_steps()
    )
