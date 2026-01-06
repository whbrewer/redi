#!/usr/bin/env python3
"""
REDI CLI - Data Readiness for AI Command Line Interface

Unified entry point for running REDI pipelines.

Usage:
    python cli.py <input_file> [options]
    python cli.py analyze <input_file>
    python cli.py run <input_file> --domain <domain> --output <output_dir>

Examples:
    # Auto-detect domain and run full pipeline
    python cli.py hpc/ANL-ALCF-DJC-AURORA.csv

    # Specify domain explicitly
    python cli.py data.nc --domain climate --output ./processed

    # Run specific stages only
    python cli.py data.csv --stages ingest,preprocess

    # Generate data card only
    python cli.py analyze data.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redi")


def detect_domain(input_path: Path, columns: Optional[List[str]] = None) -> str:
    """Auto-detect domain from file and content"""
    from .domains import detect_domain as _detect_domain, DOMAIN_EXTENSIONS

    # Check extension first
    ext = input_path.suffix.lower()
    for domain, extensions in DOMAIN_EXTENSIONS.items():
        if ext in extensions:
            return domain.value

    # For tabular files, check column names
    if ext in [".csv", ".tsv", ".parquet", ".json"]:
        if columns is None:
            try:
                import pandas as pd
                if ext == ".parquet":
                    df = pd.read_parquet(input_path, columns=None)
                    columns = list(df.columns)
                else:
                    # Just read header
                    df = pd.read_csv(input_path, nrows=0)
                    columns = list(df.columns)
            except Exception:
                columns = []

        detected = _detect_domain(columns=columns, file_extension=ext)
        return detected.value

    return "general"


def run_pipeline(
    input_path: str,
    output_dir: str = "./output",
    domain: Optional[str] = None,
    stages: Optional[List[str]] = None,
    config: Optional[dict] = None
):
    """Run the REDI pipeline"""
    from .pipeline import Pipeline, PipelineStage, create_context
    from .pipeline.ingest import get_ingest_steps
    from .pipeline.preprocess import get_preprocess_steps
    from .pipeline.transform import get_transform_steps
    from .pipeline.structure import get_structure_steps
    from .pipeline.output import get_output_steps

    input_path = Path(input_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return None

    # Auto-detect domain if not specified
    if domain is None:
        domain = detect_domain(input_path)
        logger.info(f"Auto-detected domain: {domain}")

    # Create context
    ctx = create_context(str(input_path), output_dir, domain, **(config or {}))

    # Build pipeline
    pipeline = Pipeline(domain)

    # Determine which stages to run
    all_stages = ["ingest", "preprocess", "transform", "structure", "output"]
    if stages is None:
        stages = all_stages

    # Add steps for each stage
    if "ingest" in stages:
        for step in get_ingest_steps():
            pipeline.add_step(step)

    if "preprocess" in stages:
        for step in get_preprocess_steps(domain):
            pipeline.add_step(step)

    if "transform" in stages:
        for step in get_transform_steps(domain):
            pipeline.add_step(step)

    if "structure" in stages:
        for step in get_structure_steps(domain):
            pipeline.add_step(step)

    if "output" in stages:
        for step in get_output_steps():
            pipeline.add_step(step)

    # Try to add domain-specific steps
    try:
        if domain == "climate":
            from .domains.climate import get_full_pipeline
        elif domain == "materials":
            from .domains.materials import get_full_pipeline
        elif domain == "life_sciences":
            from .domains.life_sciences import get_full_pipeline
        elif domain == "fusion":
            from .domains.fusion import get_full_pipeline
        elif domain == "hpc":
            from .domains.hpc import get_full_pipeline
        else:
            get_full_pipeline = None

        if get_full_pipeline:
            logger.info(f"Using domain-specific pipeline for: {domain}")
            pipeline = Pipeline(domain)
            for step in get_full_pipeline():
                if stages is None or step.stage.value in stages:
                    pipeline.add_step(step)
    except ImportError as e:
        logger.warning(f"Could not load domain module: {e}")

    # Execute pipeline
    logger.info(f"Running pipeline: {len(pipeline.steps)} steps")
    ctx = pipeline.execute(ctx)

    # Report results
    if ctx.errors:
        logger.warning(f"Pipeline completed with {len(ctx.errors)} errors:")
        for error in ctx.errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("Pipeline completed successfully")

    logger.info(f"Completed stages: {[s.value for s in ctx.completed_stages]}")

    return ctx


def analyze_dataset(input_path: str) -> str:
    """Run data card analysis only"""
    try:
        from .analyze import generate_report
        return generate_report(input_path)
    except ImportError as e:
        logger.error(f"Could not import analyze module: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="REDI - Data Readiness for AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv                      # Auto-detect and run full pipeline
  %(prog)s data.csv --domain hpc         # Specify domain
  %(prog)s data.csv -o ./processed       # Custom output directory
  %(prog)s data.csv --stages ingest,preprocess  # Run specific stages
  %(prog)s --analyze data.csv            # Generate data card only
        """
    )

    parser.add_argument("input", help="Input file path")
    parser.add_argument("-d", "--domain", choices=["climate", "materials", "life_sciences", "fusion", "hpc", "general"],
                       help="Scientific domain (auto-detected if not specified)")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("-s", "--stages", help="Comma-separated list of stages to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-a", "--analyze", action="store_true", help="Only analyze and generate data card")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle analyze mode
    if args.analyze:
        result = analyze_dataset(args.input)
        print(result)
        return

    # Run pipeline
    stages = args.stages.split(",") if args.stages else None
    ctx = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        domain=args.domain,
        stages=stages
    )

    if ctx:
        print("\n" + "=" * 60)
        print("REDI Pipeline Results")
        print("=" * 60)
        print(f"Input: {ctx.input_path}")
        print(f"Domain: {ctx.domain}")
        print(f"Output: {ctx.output_dir}")
        print(f"Stages completed: {[s.value for s in ctx.completed_stages]}")

        if ctx.metadata.get("saved_files"):
            print(f"\nSaved files:")
            for f in ctx.metadata["saved_files"]:
                print(f"  - {f}")

        if ctx.errors:
            print(f"\nErrors ({len(ctx.errors)}):")
            for e in ctx.errors:
                print(f"  - {e}")

        print("=" * 60)


if __name__ == "__main__":
    main()
