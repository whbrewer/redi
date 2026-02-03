#!/usr/bin/env python3
"""
REDI CLI - Readiness Engine for Data Integration

Unified entry point for running REDI in different modes:
  - run (orchestrator): Execute the full pipeline on data
  - sidecar: Observe and report on existing workflows
  - discover: Analyze unknown datasets and propose pipelines
"""

import argparse
import glob
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

# Setup logging - quieter by default
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
logger = logging.getLogger("redi")


def print_banner():
    """Print the REDI banner"""
    print(f"""{Colors.BOLD}{Colors.CYAN}
 ██████╗ ███████╗██████╗ ██╗
 ██╔══██╗██╔════╝██╔══██╗██║
 ██████╔╝█████╗  ██║  ██║██║
 ██╔══██╗██╔══╝  ██║  ██║██║
 ██║  ██║███████╗██████╔╝██║
 ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝
{Colors.END}{Colors.DIM} Readiness Engine for Data Integration{Colors.END}
""")


def print_step(icon: str, name: str, status: str, detail: str = "", timing: str = ""):
    """Print a pipeline step with status"""
    status_colors = {
        "running": Colors.YELLOW,
        "success": Colors.GREEN,
        "skipped": Colors.DIM,
        "error": Colors.RED,
    }
    color = status_colors.get(status, Colors.END)

    timing_str = f" {Colors.DIM}({timing}){Colors.END}" if timing else ""
    detail_str = f" {Colors.DIM}→ {detail}{Colors.END}" if detail else ""

    print(f"  {icon} {color}{name}{Colors.END}{detail_str}{timing_str}")


def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▸ {title}{Colors.END}")


def print_metric(label: str, value: str, indent: int = 4):
    """Print a metric"""
    print(f"{' ' * indent}{Colors.DIM}{label}:{Colors.END} {Colors.BOLD}{value}{Colors.END}")


def detect_domain(input_path: Path, columns: Optional[List[str]] = None) -> str:
    """Auto-detect domain from file and content"""
    from .domains import detect_domain as _detect_domain, DOMAIN_EXTENSIONS

    ext = input_path.suffix.lower()
    for domain, extensions in DOMAIN_EXTENSIONS.items():
        if ext in extensions:
            return domain.value

    if ext in [".csv", ".tsv", ".parquet", ".json"]:
        if columns is None:
            try:
                import pandas as pd
                if ext == ".parquet":
                    df = pd.read_parquet(input_path, columns=None)
                    columns = list(df.columns)
                else:
                    df = pd.read_csv(input_path, nrows=0)
                    columns = list(df.columns)
            except Exception:
                columns = []

        detected = _detect_domain(columns=columns, file_extension=ext)
        return detected.value

    return "general"


def expand_input_paths(input_args: List[str]) -> List[Path]:
    """Expand input arguments to a list of file paths.

    Supports:
    - Single file paths
    - Multiple file paths
    - Glob patterns (*.nc, data_*.csv, etc.)

    Args:
        input_args: List of input paths or patterns

    Returns:
        Sorted list of Path objects
    """
    expanded_paths = []

    for arg in input_args:
        # Check if it's a glob pattern
        if '*' in arg or '?' in arg or '[' in arg:
            # Expand glob pattern
            matches = glob.glob(arg, recursive=True)
            if matches:
                expanded_paths.extend(Path(m) for m in matches)
            else:
                # No matches - keep original (will fail validation later)
                expanded_paths.append(Path(arg))
        else:
            expanded_paths.append(Path(arg))

    # Sort by filename for consistent ordering (important for time-series)
    expanded_paths = sorted(set(expanded_paths), key=lambda p: p.name)

    return expanded_paths


# =============================================================================
# ORCHESTRATOR MODE (redi run)
# =============================================================================

def run_pipeline(
    input_path: Optional[str] = None,
    output_dir: str = "./output",
    domain: Optional[str] = None,
    stages: Optional[List[str]] = None,
    config: Optional[dict] = None,
    verbose: bool = False,
    style: str = "rich",
    input_paths: Optional[List[str]] = None,
    array_mode: bool = False,
    normalize_method: Optional[str] = None,
    verify: bool = False
):
    """Run the REDI pipeline with engaging visual output

    Args:
        input_path: Single input file path (for backward compatibility)
        output_dir: Output directory
        domain: Scientific domain (auto-detected if not specified)
        stages: List of stages to run
        config: Additional configuration
        verbose: Enable verbose output
        style: Output style (rich, simple, json)
        input_paths: List of input file paths for multi-file processing
        array_mode: If True, optimize pipeline for array data (skip DataFrame steps)
        normalize_method: Normalization method for array data ("minmax" or "zscore")
        verify: If True, run verification checks after each pipeline step
    """
    from .pipeline import Pipeline, PipelineStage, create_context

    # Handle input paths
    if input_paths:
        # Multi-file mode - paths already expanded
        paths = [Path(p) for p in input_paths]
    elif input_path:
        # Single file mode (backward compatible)
        paths = [Path(input_path)]
    else:
        if style == "rich":
            print(f"{Colors.RED}✗ No input files specified{Colors.END}")
        else:
            print("Error: No input files specified")
        return None

    # Validate all files exist
    missing_files = [p for p in paths if not p.exists()]
    if missing_files:
        if style == "rich":
            print(f"{Colors.RED}✗ Input file(s) not found:{Colors.END}")
            for f in missing_files[:5]:
                print(f"    {f}")
            if len(missing_files) > 5:
                print(f"    ... and {len(missing_files) - 5} more")
        else:
            print(f"Error: Input file(s) not found: {missing_files[0]}")
        return None

    # Use first file as primary for backward compatibility
    primary_path = paths[0]

    # Print banner (rich mode only)
    if style == "rich":
        print_banner()

    # Show input info
    if len(paths) == 1:
        # Single file
        file_size = primary_path.stat().st_size
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"

        if style == "rich":
            print_section("Input")
            print_metric("File", str(primary_path))
            print_metric("Size", size_str)
            print_metric("Format", primary_path.suffix.upper().replace(".", ""))
        elif style == "simple":
            print(f"Input: {primary_path} ({size_str})")
    else:
        # Multiple files
        total_size = sum(p.stat().st_size for p in paths)
        size_str = f"{total_size / 1024:.1f} KB" if total_size < 1024*1024 else f"{total_size / (1024*1024):.1f} MB"
        if total_size >= 1024*1024*1024:
            size_str = f"{total_size / (1024*1024*1024):.2f} GB"

        if style == "rich":
            print_section("Input")
            print_metric("Files", f"{len(paths)} files")
            print_metric("Total Size", size_str)
            print_metric("Format", primary_path.suffix.upper().replace(".", ""))
            print_metric("First", paths[0].name)
            print_metric("Last", paths[-1].name)
        elif style == "simple":
            print(f"Input: {len(paths)} files ({size_str})")

    # Auto-detect domain
    domain_was_auto = domain is None
    if domain is None:
        domain = detect_domain(primary_path)

    domain_icons = {
        "climate": "🌍",
        "materials": "⚛️ ",
        "life_sciences": "🧬",
        "fusion": "☀️ ",
        "hpc": "🖥️ ",
        "general": "📊",
    }
    icon = domain_icons.get(domain, "📊")

    if style == "rich":
        print_section("Domain Detection")
        if domain == "general" and domain_was_auto:
            print(f"    {Colors.YELLOW}⚠ Could not auto-detect domain{Colors.END}")
            print(f"    {Colors.DIM}Tip: Use -d/--domain to specify: climate, materials, life_sciences, fusion, hpc{Colors.END}")
        else:
            print(f"    {icon} Detected: {Colors.BOLD}{Colors.CYAN}{domain.upper()}{Colors.END}")
    elif style == "simple":
        if domain == "general" and domain_was_auto:
            print(f"Warning: Could not auto-detect domain. Use -d to specify.")
        print(f"Domain: {domain}")

    # Create context
    safe_config = dict(config or {})
    for reserved_key in ("input_path", "input", "output_dir", "domain", "input_paths"):
        safe_config.pop(reserved_key, None)

    # Pass all paths to context
    ctx = create_context(
        input_path=str(primary_path),
        output_dir=output_dir,
        domain=domain,
        input_paths=[str(p) for p in paths],
        **safe_config
    )

    # Enable step verification if requested
    ctx.verify_steps = verify

    # Load domain-specific pipeline
    if style == "rich":
        print_section("Loading Pipeline")

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
            pipeline = Pipeline(domain)
            steps = get_full_pipeline()

            # Expand stages to include prerequisites (pipeline stages are sequential)
            all_stages = ["ingest", "preprocess", "transform", "structure", "output"]
            if stages is not None:
                max_idx = max(all_stages.index(s) for s in stages if s in all_stages)
                stages = all_stages[:max_idx + 1]

            for step in steps:
                if stages is None or step.stage.value in stages:
                    pipeline.add_step(step)
            if style == "rich":
                print(f"    ✓ Loaded {Colors.BOLD}{len(pipeline.steps)}{Colors.END} steps for {domain} domain")
            elif style == "simple":
                print(f"Pipeline: {len(pipeline.steps)} steps")
        else:
            # Fall back to generic pipeline
            from .pipeline.ingest import get_ingest_steps
            from .pipeline.preprocess import get_preprocess_steps, NormalizeArray, ComputeArrayStats
            from .pipeline.transform import get_transform_steps
            from .pipeline.structure import get_structure_steps
            from .pipeline.output import get_output_steps

            pipeline = Pipeline(domain)
            all_stages = ["ingest", "preprocess", "transform", "structure", "output"]
            if stages is None:
                stages = all_stages
            else:
                # Expand stages to include prerequisites (pipeline stages are sequential)
                # e.g., "-s preprocess" becomes ["ingest", "preprocess"]
                max_idx = max(all_stages.index(s) for s in stages if s in all_stages)
                stages = all_stages[:max_idx + 1]

            if "ingest" in stages:
                for step in get_ingest_steps():
                    pipeline.add_step(step)

            if "preprocess" in stages:
                if array_mode:
                    # Use array-specific preprocessing steps
                    pipeline.add_step(ComputeArrayStats())
                    if normalize_method:
                        pipeline.add_step(NormalizeArray(method=normalize_method))
                else:
                    for step in get_preprocess_steps(domain):
                        pipeline.add_step(step)

            if "transform" in stages:
                if not array_mode:  # Skip DataFrame transforms in array mode
                    for step in get_transform_steps(domain):
                        pipeline.add_step(step)

            if "structure" in stages:
                if not array_mode:  # Skip DataFrame structure in array mode
                    for step in get_structure_steps(domain):
                        pipeline.add_step(step)

            if "output" in stages:
                for step in get_output_steps():
                    pipeline.add_step(step)

            mode_str = " (array mode)" if array_mode else ""
            if style == "rich":
                print(f"    ✓ Loaded {Colors.BOLD}{len(pipeline.steps)}{Colors.END} generic steps{mode_str}")
            elif style == "simple":
                print(f"Pipeline: {len(pipeline.steps)} steps{mode_str}")

    except ImportError as e:
        if style == "rich":
            print(f"    {Colors.YELLOW}⚠ Could not load domain module: {e}{Colors.END}")
        else:
            print(f"Error: Could not load domain module: {e}")
        return None

    # Execute pipeline with visual feedback
    if style == "rich":
        print_section("Processing Pipeline")
    elif style == "simple":
        print("Processing...")

    total_start = time.time()
    step_results = []
    prev_error_count = 0

    for i, step in enumerate(pipeline.steps, 1):
        step_start = time.time()

        # Show running status (rich mode)
        if style == "rich":
            if verbose:
                # In verbose mode, print on own line so logs don't get overwritten
                print(f"  ⏳ {Colors.YELLOW}{step.name}{Colors.END}: {step.description}")
            else:
                print(f"  ⏳ {Colors.YELLOW}{step.name}{Colors.END} ...", end="", flush=True)

        # Validate
        if not step.validate(ctx):
            step_time = time.time() - step_start
            if style == "rich":
                if verbose:
                    print(f"     {Colors.DIM}↳ skipped (validation failed){Colors.END} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                else:
                    print(f"\r  ⏭️  {Colors.DIM}{step.name}{Colors.END} → skipped (validation) {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
            elif style == "simple":
                print(f"  [SKIP] {step.name}")
            step_results.append(("skipped", step.name, "", step_time))
            continue

        # Execute
        try:
            old_metadata = dict(ctx.metadata)
            ctx = step.execute(ctx)
            step_time = time.time() - step_start

            # Determine what changed
            detail = ""
            if ctx.metadata.get("records_loaded") and "records_loaded" not in old_metadata:
                detail = f"{ctx.metadata['records_loaded']:,} records"
            elif ctx.metadata.get("array_shape") and "array_shape" not in old_metadata:
                detail = f"shape {ctx.metadata['array_shape']}"
            elif ctx.metadata.get("features_added"):
                detail = f"+{len(ctx.metadata['features_added'])} features"
            elif ctx.metadata.get("engineered_features") and "engineered_features" not in old_metadata:
                detail = f"{len(ctx.metadata['engineered_features'])} features"
            elif ctx.metadata.get("saved_files") and "saved_files" not in old_metadata:
                detail = f"{len(ctx.metadata['saved_files'])} files"
            elif ctx.metadata.get("split") and "split" not in old_metadata:
                split = ctx.metadata["split"]
                detail = f"train={split.get('train_size', '?')}, val={split.get('val_size', '?')}, test={split.get('test_size', '?')}"

            # Check for errors added during this step
            new_errors = len(ctx.errors) > prev_error_count
            prev_error_count = len(ctx.errors)

            if new_errors:
                error_msg = ctx.errors[-1]
                # Check for critical errors that should stop the pipeline
                is_critical = any(phrase in error_msg.lower() for phrase in [
                    "unknown file format",
                    "file not found",
                    "no data loaded",
                    "all files failed",
                    "missing dependency",
                    "install with:"
                ])

                if style == "rich":
                    if is_critical:
                        if verbose:
                            print(f"     {Colors.RED}↳ error: {error_msg}{Colors.END} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                        else:
                            print(f"\r  ❌ {Colors.RED}{step.name}{Colors.END} → {error_msg[:40]}... {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                    else:
                        if verbose:
                            print(f"     {Colors.YELLOW}↳ warning: {error_msg}{Colors.END} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                        else:
                            print(f"\r  ⚠️  {Colors.YELLOW}{step.name}{Colors.END} → {error_msg[:40]}... {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                elif style == "simple":
                    print(f"  [{'FAIL' if is_critical else 'WARN'}] {step.name}: {error_msg[:60]}")

                if is_critical:
                    step_results.append(("error", step.name, error_msg, step_time))
                    if style == "rich":
                        print(f"\n    {Colors.RED}Pipeline stopped due to critical error.{Colors.END}")
                    break
                else:
                    step_results.append(("warning", step.name, error_msg, step_time))
            else:
                if style == "rich":
                    detail_str = f" → {detail}" if detail else ""
                    if verbose:
                        print(f"     {Colors.GREEN}↳ done{Colors.END}{detail_str} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                    else:
                        print(f"\r  ✅ {Colors.GREEN}{step.name}{Colors.END}{detail_str} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                elif style == "simple":
                    detail_str = f": {detail}" if detail else ""
                    print(f"  [OK] {step.name}{detail_str}")
                step_results.append(("success", step.name, detail, step_time))

                # Run verification if enabled
                if ctx.verify_steps:
                    verify_result = step.verify(ctx)
                    ctx.verification_results.append(verify_result)
                    if not verify_result.passed:
                        if style == "rich":
                            print(f"       {Colors.YELLOW}⚠ Verification issues:{Colors.END}")
                            for check in verify_result.checks:
                                if not check["passed"]:
                                    print(f"         {Colors.DIM}└─{Colors.END} {check['name']}: {check.get('message', 'FAILED')}")
                        elif style == "simple":
                            print(f"    [VERIFY WARN] {step.name}")

        except Exception as e:
            step_time = time.time() - step_start
            ctx.add_error(f"Error in step {step.name}: {str(e)}")
            prev_error_count = len(ctx.errors)
            if style == "rich":
                if verbose:
                    print(f"     {Colors.RED}↳ error: {str(e)}{Colors.END} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
                else:
                    print(f"\r  ❌ {Colors.RED}{step.name}{Colors.END} → {str(e)[:50]} {Colors.DIM}({step_time*1000:.0f}ms){Colors.END}")
            elif style == "simple":
                print(f"  [FAIL] {step.name}: {str(e)[:60]}")
            step_results.append(("error", step.name, str(e), step_time))

    total_time = time.time() - total_start

    # Summary
    success_count = sum(1 for r in step_results if r[0] == "success")
    warning_count = sum(1 for r in step_results if r[0] == "warning")
    error_count = sum(1 for r in step_results if r[0] == "error")
    skipped_count = sum(1 for r in step_results if r[0] == "skipped")

    if style == "rich":
        print_section("Results")

        status_parts = []
        if success_count:
            status_parts.append(f"{Colors.GREEN}✓ {success_count} succeeded{Colors.END}")
        if warning_count:
            status_parts.append(f"{Colors.YELLOW}⚠ {warning_count} warnings{Colors.END}")
        if error_count:
            status_parts.append(f"{Colors.RED}✗ {error_count} failed{Colors.END}")
        if skipped_count:
            status_parts.append(f"{Colors.DIM}⏭ {skipped_count} skipped{Colors.END}")

        print(f"    {' │ '.join(status_parts)}")
        print(f"    {Colors.DIM}Total time: {total_time:.2f}s{Colors.END}")

        # Verification results
        if ctx.verify_steps and ctx.verification_results:
            print_section("Verification")
            all_passed = all(r.passed for r in ctx.verification_results)
            if all_passed:
                print(f"    {Colors.GREEN}✓ All {len(ctx.verification_results)} verification checks passed{Colors.END}")
            else:
                failed = [r for r in ctx.verification_results if not r.passed]
                print(f"    {Colors.YELLOW}⚠ {len(failed)}/{len(ctx.verification_results)} steps have verification issues{Colors.END}")

            for result in ctx.verification_results:
                if result.passed:
                    print(f"    {Colors.GREEN}✓{Colors.END} {result.step_name}")
                else:
                    print(f"    {Colors.RED}✗{Colors.END} {result.step_name}")
                    for check in result.checks:
                        if not check["passed"]:
                            msg = check.get("message", "FAILED")
                            if check.get("expected") is not None:
                                msg = f"expected {check['expected']}, got {check['actual']}"
                            print(f"      {Colors.DIM}└─{Colors.END} {check['name']}: {msg}")

        # Output files
        if ctx.metadata.get("saved_files"):
            print_section("Output Files")
            for f in ctx.metadata["saved_files"]:
                fpath = Path(f)
                if fpath.exists():
                    fsize = fpath.stat().st_size
                    size_str = f"{fsize / 1024:.1f} KB" if fsize < 1024*1024 else f"{fsize / (1024*1024):.2f} MB"
                    print(f"    📄 {Colors.BOLD}{fpath.name}{Colors.END} {Colors.DIM}({size_str}){Colors.END}")
                else:
                    print(f"    📄 {fpath.name}")

        # Data summary
        if ctx.metadata.get("array_shape"):
            print_section("Data Summary")
            print_metric("Shape", str(ctx.metadata["array_shape"]))
            if ctx.metadata.get("split"):
                split = ctx.metadata["split"]
                print_metric("Train", str(split.get("train_size", "N/A")))
                print_metric("Val", str(split.get("val_size", "N/A")))
                print_metric("Test", str(split.get("test_size", "N/A")))

        # Generate and save Data Card
        print_section("Data Card")
        try:
            from .analyze import generate_report, analyze_dataset, PipelineRecommender

            # Generate full report for file (use primary path for multi-file)
            report = generate_report(str(primary_path))
            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            datacard_path = ctx.output_dir / "DATA_CARD.md"
            datacard_path.write_text(report)
            if "saved_files" not in ctx.metadata:
                ctx.metadata["saved_files"] = []
            ctx.metadata["saved_files"].append(str(datacard_path))
            print(f"    📋 {Colors.BOLD}DATA_CARD.md{Colors.END} {Colors.DIM}(saved){Colors.END}")

            # Save metadata.json for downstream tooling
            metadata_path = ctx.output_dir / "metadata.json"
            metadata_dict = {
                "input_file": str(primary_path),
                "domain": domain,
                "completed_stages": [s.value for s in ctx.completed_stages] if ctx.completed_stages else [],
                "errors": ctx.errors,
                **{k: v for k, v in ctx.metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
            }
            import json as json_module
            with open(metadata_path, 'w') as f:
                json_module.dump(metadata_dict, f, indent=2, default=str)
            ctx.metadata["saved_files"].append(str(metadata_path))
            print(f"    📋 {Colors.BOLD}metadata.json{Colors.END} {Colors.DIM}(saved){Colors.END}")

            # Display recommendations in terminal
            data_card = analyze_dataset(str(primary_path))
            recommender = PipelineRecommender(data_card)
            rec = recommender.get_recommendations()

            # Track which recommended steps have been completed in this pipeline run
            completed_steps = set()
            for status, step_name, _, _ in step_results:
                if status == "success":
                    # Map pipeline step names to recommendation step names
                    step_lower = step_name.lower()
                    if "regrid" in step_lower:
                        completed_steps.add("regrid")
                    if "normalize" in step_lower:
                        completed_steps.add("normalize")
                    if "shard" in step_lower or "save" in step_lower or "output" in step_lower:
                        completed_steps.add("shard")
                    if "load" in step_lower or "ingest" in step_lower:
                        completed_steps.add("download")
                    if "featurize" in step_lower or "engineer" in step_lower:
                        completed_steps.add("featurize")
                    if "encode" in step_lower:
                        completed_steps.add("encode")
                    if "label" in step_lower:
                        completed_steps.add("label")

            print_section("Pipeline Recommendations")
            print(f"    {Colors.BOLD}Current Level:{Colors.END} {rec['current_level']}")
            print(f"    {Colors.BOLD}Target Level:{Colors.END} {rec['target_level']}")

            # Show completed steps
            completed_rec_steps = [step for step in rec['needed_steps'] if step['name'] in completed_steps]
            if completed_rec_steps:
                print()
                print(f"    {Colors.BOLD}Completed This Run:{Colors.END}")
                for step in completed_rec_steps:
                    print(f"      {Colors.GREEN}✓{Colors.END} {step['name']}: {step['description']}")

            # Show pending steps
            pending_steps = [step for step in rec['needed_steps']
                           if step['status'] == 'pending' and step['name'] not in completed_steps]
            if pending_steps:
                print()
                print(f"    {Colors.BOLD}Next Steps:{Colors.END}")
                for step in pending_steps:
                    print(f"      {Colors.YELLOW}○{Colors.END} {step['name']}: {step['description']}")

            print()
            print(f"    {Colors.BOLD}Recommended Tools:{Colors.END} {', '.join(rec['recommended_tools'])}")
            print(f"    {Colors.BOLD}Output Formats:{Colors.END} {', '.join(rec['output_formats'])}")

        except Exception as e:
            print(f"    {Colors.YELLOW}⚠ Could not generate Data Card: {e}{Colors.END}")

        # Final status
        print()
        if error_count == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ Pipeline completed successfully!{Colors.END}")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Pipeline completed with {error_count} error(s){Colors.END}")
        print()

    elif style == "simple":
        print(f"Done: {success_count} ok, {warning_count} warnings, {error_count} errors ({total_time:.2f}s)")

        # Verification results for simple style
        if ctx.verify_steps and ctx.verification_results:
            all_passed = all(r.passed for r in ctx.verification_results)
            if all_passed:
                print(f"Verification: {len(ctx.verification_results)} checks passed")
            else:
                for result in ctx.verification_results:
                    if not result.passed:
                        print(f"Verification FAILED: {result.step_name}")
                        for check in result.checks:
                            if not check["passed"]:
                                print(f"  - {check['name']}: {check.get('message', 'FAILED')}")

        # Save metadata.json and DATA_CARD.md for simple/json modes too
        try:
            from .analyze import generate_report
            ctx.output_dir.mkdir(parents=True, exist_ok=True)

            # Save DATA_CARD.md
            report = generate_report(str(primary_path))
            datacard_path = ctx.output_dir / "DATA_CARD.md"
            datacard_path.write_text(report)
            if "saved_files" not in ctx.metadata:
                ctx.metadata["saved_files"] = []
            ctx.metadata["saved_files"].append(str(datacard_path))

            # Save metadata.json
            metadata_path = ctx.output_dir / "metadata.json"
            import json as json_module
            metadata_dict = {
                "input_file": str(primary_path),
                "domain": domain,
                "completed_stages": [s.value for s in ctx.completed_stages] if ctx.completed_stages else [],
                "errors": ctx.errors,
                **{k: v for k, v in ctx.metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
            }
            with open(metadata_path, 'w') as f:
                json_module.dump(metadata_dict, f, indent=2, default=str)
            ctx.metadata["saved_files"].append(str(metadata_path))
        except Exception:
            pass  # Silent in simple mode

        if ctx.metadata.get("saved_files"):
            for f in ctx.metadata["saved_files"]:
                print(f"  Output: {f}")

    elif style == "json":
        # Save metadata.json and DATA_CARD.md for json mode too
        try:
            from .analyze import generate_report
            ctx.output_dir.mkdir(parents=True, exist_ok=True)

            report = generate_report(str(primary_path))
            datacard_path = ctx.output_dir / "DATA_CARD.md"
            datacard_path.write_text(report)
            if "saved_files" not in ctx.metadata:
                ctx.metadata["saved_files"] = []
            ctx.metadata["saved_files"].append(str(datacard_path))

            metadata_path = ctx.output_dir / "metadata.json"
            import json as json_module
            metadata_dict = {
                "input_file": str(primary_path),
                "domain": domain,
                "completed_stages": [s.value for s in ctx.completed_stages] if ctx.completed_stages else [],
                "errors": ctx.errors,
                **{k: v for k, v in ctx.metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
            }
            with open(metadata_path, 'w') as f:
                json_module.dump(metadata_dict, f, indent=2, default=str)
            ctx.metadata["saved_files"].append(str(metadata_path))
        except Exception:
            pass

        import json
        result = {
            "input": str(input_path),
            "domain": domain,
            "steps": [{"status": r[0], "name": r[1], "detail": r[2], "time_ms": r[3]*1000} for r in step_results],
            "summary": {"success": success_count, "warnings": warning_count, "errors": error_count, "skipped": skipped_count},
            "total_time_s": total_time,
            "output_files": ctx.metadata.get("saved_files", []),
            "errors": ctx.errors
        }
        print(json.dumps(result, indent=2))

    return ctx


# =============================================================================
# SIDECAR MODE (redi sidecar)
# =============================================================================

def run_sidecar(
    workflow_file: str,
    workflow_type: str = "auto",
    output_dir: str = "./output",
    watch: bool = False,
    style: str = "rich",
    config: Optional[dict] = None,
    data_dir: Optional[str] = None,
):
    """Run REDI in sidecar mode - observe existing workflows.

    Args:
        workflow_file: Path to workflow file (Snakefile, Nextflow, etc.)
        workflow_type: Type of workflow (auto, snakemake, nextflow, bash)
        output_dir: Output directory for Data Card and reports
        watch: If True, watch for changes and re-run
        style: Output style (rich, simple, json)
        config: Optional YAML configuration dict
        data_dir: Optional path to data directory to scan for intermediate files
    """
    config = config or {}
    workflow_path = Path(workflow_file)
    data_path = Path(data_dir) if data_dir else None

    if not workflow_path.exists():
        if style == "rich":
            print(f"{Colors.RED}✗ Workflow file not found: {workflow_file}{Colors.END}")
        else:
            print(f"Error: Workflow file not found: {workflow_file}")
        return None

    if style == "rich":
        print_banner()
        print_section("Sidecar Mode")
        print_metric("Workflow", str(workflow_path))

    # Auto-detect workflow type
    if workflow_type == "auto":
        if workflow_path.name.lower() in ("snakefile", "snakefile.smk") or workflow_path.suffix == ".smk":
            workflow_type = "snakemake"
        elif workflow_path.suffix in (".nf", ".nextflow"):
            workflow_type = "nextflow"
        elif workflow_path.suffix in (".sh", ".bash"):
            workflow_type = "bash"
        else:
            workflow_type = "unknown"

    if style == "rich":
        print_metric("Type", workflow_type)
        print_metric("Watch", "enabled" if watch else "disabled")
        if data_path:
            print_metric("Data Dir", str(data_path))

    # Parse workflow to understand structure
    if workflow_type == "snakemake":
        return _sidecar_snakemake(workflow_path, output_dir, watch, style, data_path)
    elif workflow_type == "nextflow":
        return _sidecar_nextflow(workflow_path, output_dir, watch, style)
    else:
        if style == "rich":
            print(f"\n    {Colors.YELLOW}⚠ Workflow type '{workflow_type}' not yet supported{Colors.END}")
            print(f"    {Colors.DIM}Supported types: snakemake, nextflow{Colors.END}")
        else:
            print(f"Error: Workflow type '{workflow_type}' not supported")
        return None


def _sidecar_snakemake(workflow_path: Path, output_dir: str, watch: bool, style: str, data_dir: Optional[Path] = None):
    """Sidecar for Snakemake workflows."""
    if style == "rich":
        print_section("Parsing Snakemake Workflow")

    # Parse the Snakefile to extract rules
    rules = _parse_snakefile(workflow_path)

    if style == "rich":
        print(f"    Found {Colors.BOLD}{len(rules)}{Colors.END} rules:")
        for rule in rules:
            print(f"      • {rule['name']}: {rule.get('description', 'no description')}")

    # Map rules to REDI pipeline stages
    stage_mapping = _map_rules_to_stages(rules)

    if style == "rich":
        print_section("Stage Mapping")
        for stage, mapped_rules in stage_mapping.items():
            if mapped_rules:
                print(f"    {stage}: {', '.join(mapped_rules)}")

    # Parse Snakemake logs for execution history
    log_report = None
    from .sidecar import find_latest_log, parse_snakemake_log

    latest_log = find_latest_log(workflow_path.parent)
    if latest_log:
        if style == "rich":
            print_section("Parsing Execution Logs")
            print(f"    Log: {Colors.DIM}{latest_log.name}{Colors.END}")

        log_report = parse_snakemake_log(latest_log)

        if style == "rich":
            print(f"    Jobs: {Colors.BOLD}{log_report.total_jobs}{Colors.END} total, "
                  f"{Colors.GREEN}{log_report.completed_jobs}{Colors.END} completed, "
                  f"{Colors.RED}{log_report.failed_jobs}{Colors.END} failed")
            print(f"    Executions parsed: {Colors.BOLD}{len(log_report.executions)}{Colors.END}")
            print(f"    File provenance: {Colors.BOLD}{len(log_report.file_provenance)}{Colors.END} files tracked")

            if log_report.errors:
                print(f"    {Colors.RED}Errors: {len(log_report.errors)}{Colors.END}")
    else:
        if style == "rich":
            print_section("Execution Logs")
            print(f"    {Colors.DIM}No Snakemake logs found in .snakemake/log/{Colors.END}")

    # Scan data files if data_dir provided
    snapshots = []
    quality_reports = []

    if data_dir and data_dir.exists():
        if style == "rich":
            print_section("Scanning Data Files")

        from .sidecar import capture_snapshot, check_snapshot, detect_transformations

        # Find all data files
        data_extensions = {'.nc', '.nc4', '.npz', '.npy', '.parquet', '.h5', '.hdf5', '.zarr'}
        data_files = []
        for ext in data_extensions:
            data_files.extend(data_dir.rglob(f'*{ext}'))

        # Sort by modification time (oldest first = earlier in pipeline)
        data_files = sorted(data_files, key=lambda f: f.stat().st_mtime)

        if style == "rich":
            print(f"    Found {Colors.BOLD}{len(data_files)}{Colors.END} data files")

        # Capture snapshots (limit to first 20 for speed)
        for f in data_files[:20]:
            try:
                if style == "rich":
                    print(f"    📊 Scanning {f.name}...", end="", flush=True)

                snapshot = capture_snapshot(f)
                snapshots.append(snapshot)

                # Run quality checks
                qr = check_snapshot(snapshot)
                quality_reports.append(qr)

                if style == "rich":
                    if qr.passed:
                        print(f"\r    ✅ {f.name}: {len(snapshot.variables)} vars, {qr.summary()}")
                    else:
                        print(f"\r    ⚠️  {f.name}: {len(snapshot.variables)} vars, {qr.summary()}")

            except Exception as e:
                if style == "rich":
                    print(f"\r    ❌ {f.name}: {str(e)[:50]}")

        if len(data_files) > 20:
            if style == "rich":
                print(f"    {Colors.DIM}... and {len(data_files) - 20} more files{Colors.END}")

    # Detect transformation parameters
    transform_report = None
    if data_dir and data_dir.exists():
        if style == "rich":
            print_section("Detecting Transformation Parameters")

        transform_report = detect_transformations(data_dir, snapshots)

        if transform_report.normalization:
            norm = transform_report.normalization
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Normalization detected: {Colors.BOLD}{norm.method}{Colors.END}")
                if norm.mean_file:
                    print(f"      Mean file: {Path(norm.mean_file).name}")
                if norm.std_file:
                    print(f"      Std file: {Path(norm.std_file).name}")
                if norm.mean_values:
                    print(f"      Variables: {len(norm.mean_values)}")
                    for var, mean in list(norm.mean_values.items())[:3]:
                        std = norm.std_values.get(var, 'N/A')
                        if isinstance(std, (int, float)):
                            print(f"        {var}: mean={mean:.4g}, std={std:.4g}")
                        else:
                            print(f"        {var}: mean={mean:.4g}")
                    if len(norm.mean_values) > 3:
                        print(f"        {Colors.DIM}... and {len(norm.mean_values) - 3} more{Colors.END}")

        if transform_report.splits:
            splits = transform_report.splits
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Train/val/test split detected: {Colors.BOLD}{splits.method}{Colors.END}")
                if splits.train_size:
                    print(f"      Train: {splits.train_size:,} samples")
                if splits.val_size:
                    print(f"      Val: {splits.val_size:,} samples")
                if splits.test_size:
                    print(f"      Test: {splits.test_size:,} samples")

        if transform_report.regrid:
            regrid = transform_report.regrid
            if style == "rich":
                print(f"    {Colors.GREEN}✓{Colors.END} Regridding detected: {Colors.BOLD}{regrid.method}{Colors.END}")
                if regrid.target_resolution:
                    print(f"      Target resolution: {regrid.target_resolution[0]}° x {regrid.target_resolution[1]}°")

        if not transform_report.normalization and not transform_report.splits and not transform_report.regrid:
            if style == "rich":
                print(f"    {Colors.DIM}No transformation parameters detected{Colors.END}")

    # Check for data quality issues in transformation report
    transform_issues = []
    if transform_report:
        from .sidecar import check_transformation_report
        transform_issues = check_transformation_report(transform_report)

        if transform_issues:
            if style == "rich":
                print_section("Transformation Quality Checks")
                for issue in transform_issues:
                    if issue.severity.value == 'error':
                        icon = f"{Colors.RED}✗{Colors.END}"
                    elif issue.severity.value == 'warning':
                        icon = f"{Colors.YELLOW}⚠{Colors.END}"
                    else:
                        icon = f"{Colors.DIM}ℹ{Colors.END}"
                    print(f"    {icon} {Colors.BOLD}{issue.check_name}:{Colors.END} {issue.message}")

    # Create output directory and save analysis
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate sidecar report
    report = {
        "workflow_file": str(workflow_path),
        "workflow_type": "snakemake",
        "rules": rules,
        "stage_mapping": stage_mapping,
        "mode": "sidecar",
    }

    # Add data snapshots if we scanned
    if snapshots:
        report["data_snapshots"] = [s.to_dict() for s in snapshots]
        report["quality_summary"] = {
            "files_scanned": len(snapshots),
            "passed": sum(1 for qr in quality_reports if qr.passed),
            "with_warnings": sum(1 for qr in quality_reports if qr.has_warnings),
            "failed": sum(1 for qr in quality_reports if not qr.passed),
            "issues": [issue.to_dict() for qr in quality_reports for issue in qr.issues]
        }

    # Add transformation parameters if detected
    if transform_report:
        report["transformations"] = transform_report.to_dict()

    # Add transformation quality issues
    if transform_issues:
        report["transformation_quality"] = [issue.to_dict() for issue in transform_issues]

    # Add execution log info if parsed
    if log_report:
        report["execution_log"] = {
            "log_file": str(latest_log),
            "job_stats": log_report.job_stats,
            "total_jobs": log_report.total_jobs,
            "completed_jobs": log_report.completed_jobs,
            "failed_jobs": log_report.failed_jobs,
            "executions_count": len(log_report.executions),
            "file_provenance": log_report.file_provenance,
            "errors": log_report.errors[:5] if log_report.errors else [],
        }

    import json
    report_path = out_path / "sidecar_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Generate Data Card markdown if we have snapshots
    if snapshots:
        datacard_path = out_path / "DATA_CARD.md"
        datacard_content = _generate_sidecar_datacard(
            workflow_path, rules, stage_mapping, snapshots, quality_reports,
            transform_report, log_report
        )
        datacard_path.write_text(datacard_content)

        if style == "rich":
            print_section("Output")
            print(f"    📋 {Colors.BOLD}sidecar_report.json{Colors.END} {Colors.DIM}(saved){Colors.END}")
            print(f"    📋 {Colors.BOLD}DATA_CARD.md{Colors.END} {Colors.DIM}(saved){Colors.END}")
    else:
        if style == "rich":
            print_section("Output")
            print(f"    📋 {Colors.BOLD}sidecar_report.json{Colors.END} {Colors.DIM}(saved){Colors.END}")

    if style == "rich":
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Sidecar analysis complete{Colors.END}")
        if not data_dir:
            print(f"{Colors.DIM}  Tip: Use --data-dir to scan intermediate data files{Colors.END}")
        print()

    return report


def _generate_sidecar_datacard(workflow_path: Path, rules: list, stage_mapping: dict,
                                snapshots: list, quality_reports: list,
                                transform_report=None, log_report=None) -> str:
    """Generate a Data Card markdown from sidecar analysis."""
    lines = [
        f"# Data Card: {workflow_path.parent.name}",
        "",
        f"**Generated by:** REDI Sidecar",
        f"**Workflow:** `{workflow_path.name}`",
        f"**Files Analyzed:** {len(snapshots)}",
    ]

    # Add execution summary if log was parsed
    if log_report:
        lines.extend([
            f"**Jobs:** {log_report.total_jobs} total, {log_report.completed_jobs} completed, {log_report.failed_jobs} failed",
        ])

    lines.extend([
        "",
        "## Pipeline Structure",
        "",
    ])

    # Rules table
    lines.append("| Rule | Description | Stage |")
    lines.append("|------|-------------|-------|")
    for rule in rules:
        stage = next((s for s, r in stage_mapping.items() if rule['name'] in r), '-')
        lines.append(f"| {rule['name']} | {rule.get('description', '-')} | {stage} |")

    lines.append("")

    # File provenance from execution log
    if log_report and log_report.file_provenance:
        lines.extend([
            "## File Provenance",
            "",
            "Files tracked from execution log:",
            "",
            "| Output File | Created By |",
            "|-------------|------------|",
        ])
        # Group by rule
        rule_outputs: dict = {}
        for output_file, rule_name in log_report.file_provenance.items():
            if rule_name not in rule_outputs:
                rule_outputs[rule_name] = []
            rule_outputs[rule_name].append(Path(output_file).name)

        for rule_name, outputs in list(rule_outputs.items())[:10]:
            for output in outputs[:3]:
                lines.append(f"| `{output}` | {rule_name} |")
            if len(outputs) > 3:
                lines.append(f"| *... {len(outputs) - 3} more* | {rule_name} |")

        if len(rule_outputs) > 10:
            lines.append(f"| *... {len(log_report.file_provenance) - sum(min(len(o), 3) for o in list(rule_outputs.values())[:10])} more files* | |")

        lines.append("")

    # Data snapshots summary
    lines.extend([
        "## Data Files",
        "",
        "| File | Format | Size | Variables | Quality |",
        "|------|--------|------|-----------|---------|",
    ])

    for snapshot, qr in zip(snapshots, quality_reports):
        size_mb = snapshot.size_bytes / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{snapshot.size_bytes / 1024:.1f} KB"
        quality = "✓" if qr.passed else "⚠" if qr.has_warnings else "✗"
        lines.append(f"| {snapshot.filename} | {snapshot.format} | {size_str} | {len(snapshot.variables)} | {quality} |")

    lines.append("")

    # Variable statistics
    lines.extend([
        "## Variables Summary",
        "",
    ])

    # Collect all unique variables across snapshots
    all_vars = {}
    for snapshot in snapshots:
        for name, var in snapshot.variables.items():
            if name not in all_vars:
                all_vars[name] = var

    if all_vars:
        lines.append("| Variable | Shape | Dtype | Min | Max | Mean | Std |")
        lines.append("|----------|-------|-------|-----|-----|------|-----|")

        for name, var in list(all_vars.items())[:15]:
            shape_str = str(var.shape)
            min_str = f"{var.min:.4g}" if var.min is not None else "-"
            max_str = f"{var.max:.4g}" if var.max is not None else "-"
            mean_str = f"{var.mean:.4g}" if var.mean is not None else "-"
            std_str = f"{var.std:.4g}" if var.std is not None else "-"
            lines.append(f"| {name} | {shape_str} | {var.dtype} | {min_str} | {max_str} | {mean_str} | {std_str} |")

        if len(all_vars) > 15:
            lines.append(f"| *... {len(all_vars) - 15} more* | | | | | | |")

    lines.append("")

    # Transformation parameters section
    if transform_report:
        transform_md = transform_report.to_markdown()
        if transform_md:
            lines.append(transform_md)
            lines.append("")

    # Quality issues
    all_issues = [issue for qr in quality_reports for issue in qr.issues]
    if all_issues:
        lines.extend([
            "## Quality Issues",
            "",
        ])

        errors = [i for i in all_issues if i.severity.value == 'error']
        warnings = [i for i in all_issues if i.severity.value == 'warning']

        if errors:
            lines.append("### Errors")
            for issue in errors[:5]:
                lines.append(f"- **{issue.check_name}** [{issue.variable}]: {issue.message}")

        if warnings:
            lines.append("")
            lines.append("### Warnings")
            for issue in warnings[:10]:
                lines.append(f"- **{issue.check_name}** [{issue.variable}]: {issue.message}")

    lines.extend([
        "",
        "---",
        "*Generated by REDI Sidecar - Observability for Data Pipelines*",
    ])

    return "\n".join(lines)


def _parse_snakefile(path: Path) -> List[dict]:
    """Parse a Snakefile to extract rule information."""
    rules = []
    content = path.read_text()

    import re
    # Simple regex to find rule definitions
    rule_pattern = re.compile(r'rule\s+(\w+):', re.MULTILINE)

    for match in rule_pattern.finditer(content):
        rule_name = match.group(1)
        # Extract rule body (simplified)
        start = match.end()
        # Find next rule or end of file
        next_match = rule_pattern.search(content, start)
        end = next_match.start() if next_match else len(content)
        rule_body = content[start:end]

        # Extract input/output patterns
        input_match = re.search(r'input:\s*\n?\s*(.+?)(?=\n\s*\w+:|$)', rule_body, re.DOTALL)
        output_match = re.search(r'output:\s*\n?\s*(.+?)(?=\n\s*\w+:|$)', rule_body, re.DOTALL)

        rules.append({
            'name': rule_name,
            'input': input_match.group(1).strip() if input_match else None,
            'output': output_match.group(1).strip() if output_match else None,
            'description': _infer_rule_description(rule_name, rule_body),
        })

    return rules


def _infer_rule_description(name: str, body: str) -> str:
    """Infer a description for a rule based on its name and content."""
    name_lower = name.lower()

    if 'download' in name_lower or 'wget' in body or 'curl' in body:
        return "Download data from remote source"
    elif 'regrid' in name_lower:
        return "Regrid data to target resolution"
    elif 'preprocess' in name_lower or 'clean' in name_lower:
        return "Preprocess and clean data"
    elif 'merge' in name_lower or 'concat' in name_lower:
        return "Merge multiple files"
    elif 'normalize' in name_lower:
        return "Normalize data values"
    elif 'split' in name_lower:
        return "Split into train/val/test"
    elif name_lower == 'all':
        return "Final target rule"
    else:
        return "Processing step"


def _map_rules_to_stages(rules: List[dict]) -> dict:
    """Map Snakemake rules to REDI pipeline stages."""
    stage_mapping = {
        'ingest': [],
        'preprocess': [],
        'transform': [],
        'structure': [],
        'output': [],
    }

    for rule in rules:
        name_lower = rule['name'].lower()
        desc_lower = rule.get('description', '').lower()

        if 'download' in name_lower or 'download' in desc_lower:
            stage_mapping['ingest'].append(rule['name'])
        elif 'regrid' in name_lower or 'clean' in name_lower or 'preprocess' in name_lower:
            stage_mapping['preprocess'].append(rule['name'])
        elif 'normalize' in name_lower or 'transform' in name_lower or 'feature' in name_lower:
            stage_mapping['transform'].append(rule['name'])
        elif 'merge' in name_lower or 'concat' in name_lower or 'structure' in name_lower:
            stage_mapping['structure'].append(rule['name'])
        elif 'split' in name_lower or 'save' in name_lower or rule['name'] == 'all':
            stage_mapping['output'].append(rule['name'])

    return stage_mapping


def _sidecar_nextflow(workflow_path: Path, output_dir: str, watch: bool, style: str):
    """Sidecar for Nextflow workflows."""
    if style == "rich":
        print(f"\n    {Colors.YELLOW}⚠ Nextflow sidecar support coming soon{Colors.END}")
    return None


# =============================================================================
# DISCOVER MODE (redi discover)
# =============================================================================

def run_discover(
    input_path: str,
    output_dir: str = "./output",
    style: str = "rich",
    interactive: bool = True,
    config: Optional[dict] = None,
):
    """Run REDI in discover mode - analyze unknown datasets.

    Args:
        input_path: Path to data file or directory
        output_dir: Output directory for reports
        style: Output style (rich, simple, json)
        interactive: If True, prompt for user confirmation before proposing pipeline
        config: Optional YAML configuration dict
    """
    config = config or {}
    path = Path(input_path)

    if not path.exists():
        if style == "rich":
            print(f"{Colors.RED}✗ Path not found: {input_path}{Colors.END}")
        else:
            print(f"Error: Path not found: {input_path}")
        return None

    if style == "rich":
        print_banner()
        print_section("Discovery Mode")
        print_metric("Path", str(path))
        print_metric("Type", "directory" if path.is_dir() else "file")

    # Phase 1: Inspect
    if style == "rich":
        print_section("Phase 1: Inspect")

    inspection = _inspect_data(path, style)

    # Phase 2: Infer
    if style == "rich":
        print_section("Phase 2: Infer")

    inference = _infer_characteristics(inspection, style)

    # Phase 3: Propose
    if style == "rich":
        print_section("Phase 3: Propose Pipeline")

    proposal = _propose_pipeline(inspection, inference, style)

    # Save discovery report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report = {
        "input_path": str(path),
        "mode": "discover",
        "inspection": inspection,
        "inference": inference,
        "proposal": proposal,
    }

    import json
    report_path = out_path / "discovery_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Generate human-readable report
    md_report = _generate_discovery_markdown(report)
    md_path = out_path / "DISCOVERY_REPORT.md"
    md_path.write_text(md_report)

    if style == "rich":
        print_section("Output")
        print(f"    📋 {Colors.BOLD}discovery_report.json{Colors.END} {Colors.DIM}(saved){Colors.END}")
        print(f"    📋 {Colors.BOLD}DISCOVERY_REPORT.md{Colors.END} {Colors.DIM}(saved){Colors.END}")
        print()

        if interactive and proposal.get('pipeline_steps'):
            print(f"{Colors.BOLD}Proposed Pipeline:{Colors.END}")
            for i, step in enumerate(proposal['pipeline_steps'], 1):
                print(f"  {i}. {step['name']}: {step['description']}")
            print()
            print(f"{Colors.DIM}To execute this pipeline:{Colors.END}")
            print(f"    redi run {input_path} -o {output_dir}")

        print()
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Discovery complete{Colors.END}")
        print()

    return report


def _inspect_data(path: Path, style: str) -> dict:
    """Inspect data files and gather metadata."""
    inspection = {
        'files': [],
        'total_size': 0,
        'formats': set(),
        'sample_data': {},
    }

    if path.is_file():
        files = [path]
    else:
        # Get all files in directory
        files = list(path.rglob('*'))
        files = [f for f in files if f.is_file()]

    for f in files[:100]:  # Limit to first 100 files
        file_info = {
            'path': str(f),
            'name': f.name,
            'suffix': f.suffix.lower(),
            'size': f.stat().st_size,
        }
        inspection['files'].append(file_info)
        inspection['total_size'] += file_info['size']
        inspection['formats'].add(f.suffix.lower())

    inspection['formats'] = list(inspection['formats'])
    inspection['file_count'] = len(files)

    # Sample first file of each format
    for fmt in inspection['formats']:
        sample_file = next((f for f in files if f.suffix.lower() == fmt), None)
        if sample_file:
            inspection['sample_data'][fmt] = _sample_file(sample_file)

    if style == "rich":
        size_str = _format_size(inspection['total_size'])
        print(f"    Files: {Colors.BOLD}{inspection['file_count']}{Colors.END}")
        print(f"    Total size: {Colors.BOLD}{size_str}{Colors.END}")
        print(f"    Formats: {Colors.BOLD}{', '.join(inspection['formats'])}{Colors.END}")

    return inspection


def _sample_file(path: Path) -> dict:
    """Sample a file to understand its structure."""
    sample = {'path': str(path), 'format': path.suffix.lower()}

    try:
        if path.suffix.lower() in ['.csv', '.tsv']:
            import pandas as pd
            df = pd.read_csv(path, nrows=5)
            sample['columns'] = list(df.columns)
            sample['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            sample['shape'] = list(df.shape)
            sample['preview'] = df.head(3).to_dict()

        elif path.suffix.lower() in ['.npy', '.npz']:
            import numpy as np
            if path.suffix.lower() == '.npy':
                arr = np.load(path, mmap_mode='r')
                sample['shape'] = list(arr.shape)
                sample['dtype'] = str(arr.dtype)
            else:
                with np.load(path) as npz:
                    sample['arrays'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                                       for k, v in npz.items()}

        elif path.suffix.lower() in ['.nc', '.nc4']:
            try:
                import xarray as xr
                ds = xr.open_dataset(path)
                sample['variables'] = list(ds.data_vars)
                sample['dims'] = dict(ds.dims)
                sample['coords'] = list(ds.coords)
                ds.close()
            except ImportError:
                sample['error'] = 'xarray not installed'

        elif path.suffix.lower() in ['.h5', '.hdf5']:
            try:
                import h5py
                with h5py.File(path, 'r') as f:
                    sample['keys'] = list(f.keys())
                    sample['structure'] = _h5_structure(f)
            except ImportError:
                sample['error'] = 'h5py not installed'

        elif path.suffix.lower() == '.json':
            import json
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                sample['type'] = 'array'
                sample['length'] = len(data)
                if data:
                    sample['first_item_keys'] = list(data[0].keys()) if isinstance(data[0], dict) else None
            elif isinstance(data, dict):
                sample['type'] = 'object'
                sample['keys'] = list(data.keys())

        elif path.suffix.lower() == '.parquet':
            import pandas as pd
            df = pd.read_parquet(path)
            sample['columns'] = list(df.columns)
            sample['shape'] = list(df.shape)
            sample['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    except Exception as e:
        sample['error'] = str(e)

    return sample


def _h5_structure(group, prefix='') -> dict:
    """Recursively get HDF5 structure."""
    structure = {}
    for key in list(group.keys())[:20]:  # Limit keys
        item = group[key]
        full_key = f"{prefix}/{key}" if prefix else key
        if hasattr(item, 'shape'):
            structure[full_key] = {'shape': list(item.shape), 'dtype': str(item.dtype)}
        elif hasattr(item, 'keys'):
            structure.update(_h5_structure(item, full_key))
    return structure


def _infer_characteristics(inspection: dict, style: str) -> dict:
    """Infer data characteristics from inspection."""
    inference = {
        'modality': 'unknown',
        'likely_domain': 'general',
        'has_labels': False,
        'has_splits': False,
        'sparsity': None,
        'suggested_format': 'npz',
    }

    formats = inspection.get('formats', [])
    samples = inspection.get('sample_data', {})

    # Infer modality
    if '.nc' in formats or '.nc4' in formats or '.grib' in formats:
        inference['modality'] = 'gridded'
        inference['likely_domain'] = 'climate'
        inference['suggested_format'] = 'zarr'
    elif '.cif' in formats or '.poscar' in formats:
        inference['modality'] = 'structured'
        inference['likely_domain'] = 'materials'
        inference['suggested_format'] = 'hdf5'
    elif '.fasta' in formats or '.pdb' in formats:
        inference['modality'] = 'sequence'
        inference['likely_domain'] = 'life_sciences'
        inference['suggested_format'] = 'hdf5'
    elif '.npy' in formats or '.npz' in formats:
        inference['modality'] = 'array'
        # Check shapes to infer type
        for fmt, sample in samples.items():
            if 'shape' in sample:
                shape = sample['shape']
                if len(shape) == 4:
                    inference['modality'] = 'image_batch'
                elif len(shape) == 3:
                    inference['modality'] = 'sequence_or_image'
                elif len(shape) == 2:
                    inference['modality'] = 'tabular_or_embedding'
    elif '.csv' in formats or '.tsv' in formats or '.parquet' in formats:
        inference['modality'] = 'tabular'
        # Check columns for domain hints
        for fmt, sample in samples.items():
            cols = sample.get('columns', [])
            cols_lower = [c.lower() for c in cols]

            if any(c in cols_lower for c in ['temperature', 'lat', 'lon', 'latitude', 'longitude']):
                inference['likely_domain'] = 'climate'
            elif any(c in cols_lower for c in ['formula', 'lattice', 'spacegroup', 'energy']):
                inference['likely_domain'] = 'materials'
            elif any(c in cols_lower for c in ['sequence', 'protein', 'gene', 'mutation']):
                inference['likely_domain'] = 'life_sciences'
            elif any(c in cols_lower for c in ['plasma', 'tokamak', 'disruption', 'ip']):
                inference['likely_domain'] = 'fusion'
            elif any(c in cols_lower for c in ['job', 'queue', 'walltime', 'nodes', 'cores']):
                inference['likely_domain'] = 'hpc'

            # Check for labels
            if any(c in cols_lower for c in ['label', 'target', 'y', 'class', 'category']):
                inference['has_labels'] = True

            # Check for splits
            if any(c in cols_lower for c in ['split', 'fold', 'train', 'test', 'val']):
                inference['has_splits'] = True

    if style == "rich":
        print(f"    Modality: {Colors.BOLD}{inference['modality']}{Colors.END}")
        print(f"    Likely domain: {Colors.BOLD}{inference['likely_domain']}{Colors.END}")
        print(f"    Has labels: {Colors.BOLD}{'yes' if inference['has_labels'] else 'no'}{Colors.END}")
        print(f"    Has splits: {Colors.BOLD}{'yes' if inference['has_splits'] else 'no'}{Colors.END}")

    return inference


def _propose_pipeline(inspection: dict, inference: dict, style: str) -> dict:
    """Propose a pipeline based on inspection and inference."""
    proposal = {
        'pipeline_steps': [],
        'output_format': inference.get('suggested_format', 'npz'),
        'confidence': 'medium',
        'notes': [],
    }

    modality = inference.get('modality', 'unknown')
    domain = inference.get('likely_domain', 'general')

    # Always start with ingest
    proposal['pipeline_steps'].append({
        'name': 'ingest',
        'description': f"Load {modality} data from {', '.join(inspection.get('formats', ['files']))}",
        'stage': 'ingest',
    })

    # Preprocessing based on modality
    if modality == 'tabular':
        proposal['pipeline_steps'].append({
            'name': 'handle_missing',
            'description': 'Handle missing values (impute or drop)',
            'stage': 'preprocess',
        })
        proposal['pipeline_steps'].append({
            'name': 'remove_duplicates',
            'description': 'Remove duplicate rows',
            'stage': 'preprocess',
        })
    elif modality == 'gridded':
        proposal['pipeline_steps'].append({
            'name': 'regrid',
            'description': 'Regrid to uniform resolution',
            'stage': 'preprocess',
        })
    elif modality in ['array', 'image_batch']:
        proposal['pipeline_steps'].append({
            'name': 'validate_shapes',
            'description': 'Validate array shapes are consistent',
            'stage': 'preprocess',
        })

    # Transform
    if modality == 'tabular':
        proposal['pipeline_steps'].append({
            'name': 'normalize',
            'description': 'Normalize numeric features',
            'stage': 'transform',
        })
        proposal['pipeline_steps'].append({
            'name': 'encode_categoricals',
            'description': 'Encode categorical variables',
            'stage': 'transform',
        })
    elif modality in ['array', 'gridded', 'image_batch']:
        proposal['pipeline_steps'].append({
            'name': 'normalize',
            'description': 'Normalize to [0,1] or standardize',
            'stage': 'transform',
        })

    # Output
    if not inference.get('has_splits'):
        proposal['pipeline_steps'].append({
            'name': 'split',
            'description': 'Create train/val/test splits',
            'stage': 'output',
        })

    proposal['pipeline_steps'].append({
        'name': 'save',
        'description': f"Save to {proposal['output_format']} format",
        'stage': 'output',
    })

    # Add notes
    if inference.get('likely_domain') == 'general':
        proposal['notes'].append("Could not confidently detect domain. Consider specifying with --domain.")
        proposal['confidence'] = 'low'

    if not inference.get('has_labels'):
        proposal['notes'].append("No label column detected. This may be unsupervised or labels need to be added.")

    if style == "rich":
        print(f"    Steps: {Colors.BOLD}{len(proposal['pipeline_steps'])}{Colors.END}")
        print(f"    Output format: {Colors.BOLD}{proposal['output_format']}{Colors.END}")
        print(f"    Confidence: {Colors.BOLD}{proposal['confidence']}{Colors.END}")
        if proposal['notes']:
            print(f"    Notes:")
            for note in proposal['notes']:
                print(f"      {Colors.YELLOW}•{Colors.END} {note}")

    return proposal


def _generate_discovery_markdown(report: dict) -> str:
    """Generate markdown report from discovery results."""
    lines = [
        "# REDI Discovery Report",
        "",
        f"**Input:** `{report['input_path']}`",
        "",
        "## Inspection",
        "",
        f"- **Files:** {report['inspection'].get('file_count', 'unknown')}",
        f"- **Total Size:** {_format_size(report['inspection'].get('total_size', 0))}",
        f"- **Formats:** {', '.join(report['inspection'].get('formats', []))}",
        "",
        "## Inference",
        "",
        f"- **Modality:** {report['inference'].get('modality', 'unknown')}",
        f"- **Likely Domain:** {report['inference'].get('likely_domain', 'general')}",
        f"- **Has Labels:** {'Yes' if report['inference'].get('has_labels') else 'No'}",
        f"- **Has Splits:** {'Yes' if report['inference'].get('has_splits') else 'No'}",
        "",
        "## Proposed Pipeline",
        "",
    ]

    for i, step in enumerate(report['proposal'].get('pipeline_steps', []), 1):
        lines.append(f"{i}. **{step['name']}** ({step['stage']}): {step['description']}")

    lines.extend([
        "",
        f"**Output Format:** {report['proposal'].get('output_format', 'npz')}",
        f"**Confidence:** {report['proposal'].get('confidence', 'medium')}",
    ])

    if report['proposal'].get('notes'):
        lines.extend(["", "### Notes", ""])
        for note in report['proposal']['notes']:
            lines.append(f"- {note}")

    lines.extend([
        "",
        "---",
        "*Generated by REDI (Readiness Engine for Data Integration)*",
    ])

    return "\n".join(lines)


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='redi',
        description="REDI - Readiness Engine for Data Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run        Run the pipeline (orchestrator mode) - default
  sidecar    Observe existing workflows
  discover   Analyze unknown datasets and propose pipelines

Examples:
  redi run data.csv                     # Run pipeline on data
  redi data.csv                         # Same as above (run is default)
  redi sidecar --workflow Snakefile     # Observe Snakemake workflow
  redi discover /path/to/data/          # Analyze unknown dataset

For command-specific help:
  redi run --help
  redi sidecar --help
  redi discover --help
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # =========================================================================
    # redi run (orchestrator)
    # =========================================================================
    run_parser = subparsers.add_parser(
        'run',
        help='Run the REDI pipeline (orchestrator mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  redi run data.csv                      # Auto-detect and run full pipeline
  redi run data.csv --domain hpc         # Specify domain
  redi run data.csv -o ./processed       # Custom output directory
  redi run data.csv --stages ingest,preprocess  # Run specific stages
  redi run "data_*.nc"                   # Process multiple files (glob)

Download examples (climate domain):
  redi run --download --source esgf --variable tas --model TaiESM1
  redi run --download --source cds --variable 2m_temperature --year 2020
        """
    )
    run_parser.add_argument("input", nargs="*", help="Input file path(s) or glob pattern")
    run_parser.add_argument("-d", "--domain",
                           choices=["climate", "materials", "life_sciences", "fusion", "hpc", "general"],
                           help="Scientific domain (auto-detected if not specified)")
    run_parser.add_argument("-o", "--output", default="./output", help="Output directory")
    run_parser.add_argument("-s", "--stages",
                           help="Comma-separated list of stages to run. "
                                "Available: ingest, preprocess, transform, structure, output. "
                                "Prerequisites are auto-included (e.g., -s preprocess runs ingest,preprocess)")
    run_parser.add_argument("-f", "--format", choices=["npz", "parquet", "csv", "hdf5", "zarr"],
                           default="npz", help="Output format")
    run_parser.add_argument("--array", action="store_true",
                           help="Optimize pipeline for array data (skip DataFrame-specific steps)")
    run_parser.add_argument("--normalize", choices=["minmax", "zscore"],
                           help="Normalization method for array data")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    run_parser.add_argument("--verify", action="store_true",
                           help="Run verification checks after each pipeline step")
    run_parser.add_argument("--style", choices=["rich", "simple", "json"], default="rich",
                           help="Output style")
    run_parser.add_argument("-c", "--config", help="YAML configuration file path")

    # Download options (for climate domain)
    download_group = run_parser.add_argument_group("download options",
        "Download data from remote sources (ESGF for CMIP6, CDS for ERA5)")
    download_group.add_argument("--download", action="store_true",
                               help="Download data before processing")
    download_group.add_argument("--source", choices=["esgf", "cds"], default="esgf",
                               help="Data source: esgf (CMIP6) or cds (ERA5)")
    download_group.add_argument("--variable",
                               help="Variable to download (e.g., tas, pr, t2m)")
    download_group.add_argument("--model",
                               help="Climate model/source_id (e.g., TaiESM1, MPI-ESM1-2-HR)")
    download_group.add_argument("--experiment", default="historical",
                               help="Experiment ID (e.g., historical, ssp585)")
    download_group.add_argument("--frequency", default="6hr",
                               help="Time frequency (e.g., 6hr, day, mon)")
    download_group.add_argument("--year", type=int,
                               help="Year to download (for CDS/ERA5)")

    # =========================================================================
    # redi sidecar
    # =========================================================================
    sidecar_parser = subparsers.add_parser(
        'sidecar',
        help='Observe existing workflows (sidecar mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  redi sidecar --workflow Snakefile
  redi sidecar --workflow pipeline.nf --type nextflow
  redi sidecar --workflow Snakefile --watch
  redi sidecar -c sidecar_config.yaml
        """
    )
    sidecar_parser.add_argument("-w", "--workflow",
                                help="Path to workflow file (Snakefile, *.nf, etc.)")
    sidecar_parser.add_argument("-t", "--type", dest="workflow_type", default="auto",
                                choices=["auto", "snakemake", "nextflow", "bash"],
                                help="Workflow type (auto-detected if not specified)")
    sidecar_parser.add_argument("-o", "--output", default="./output", help="Output directory")
    sidecar_parser.add_argument("--data-dir", dest="data_dir",
                                help="Directory containing data files to scan for statistics")
    sidecar_parser.add_argument("--watch", action="store_true",
                                help="Watch for changes and re-run")
    sidecar_parser.add_argument("--style", choices=["rich", "simple", "json"], default="rich",
                                help="Output style")
    sidecar_parser.add_argument("-c", "--config", help="YAML configuration file")

    # =========================================================================
    # redi discover
    # =========================================================================
    discover_parser = subparsers.add_parser(
        'discover',
        help='Analyze unknown datasets and propose pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  redi discover /path/to/data/
  redi discover dataset.csv -o ./analysis
  redi discover /data --no-interactive
  redi discover -c discover_config.yaml
        """
    )
    discover_parser.add_argument("input", nargs="?", help="Path to data file or directory")
    discover_parser.add_argument("-o", "--output", default="./output", help="Output directory")
    discover_parser.add_argument("--style", choices=["rich", "simple", "json"], default="rich",
                                help="Output style")
    discover_parser.add_argument("--no-interactive", action="store_true",
                                help="Don't prompt for confirmation")
    discover_parser.add_argument("-c", "--config", help="YAML configuration file")

    # Parse arguments
    args = parser.parse_args()

    # Handle backward compatibility: if no command but has positional args, assume 'run'
    if args.command is None:
        # Check if there are any remaining args that look like input files
        # Re-parse with 'run' as default
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            # First arg is likely a file, treat as 'redi run'
            sys.argv.insert(1, 'run')
            args = parser.parse_args()
        elif len(sys.argv) > 1:
            # Has flags but no command - might be old-style usage
            # Try to handle common flags
            parser.print_help()
            return
        else:
            parser.print_help()
            return

    # Dispatch to appropriate handler
    if args.command == 'run':
        _handle_run_command(args)
    elif args.command == 'sidecar':
        _handle_sidecar_command(args)
    elif args.command == 'discover':
        _handle_discover_command(args)
    else:
        parser.print_help()


def _handle_run_command(args):
    """Handle the 'run' subcommand."""
    config_path = args.config
    input_args = args.input

    # Check if first input is a YAML config file
    if not config_path and input_args and len(input_args) == 1:
        if input_args[0].endswith((".yml", ".yaml")):
            config_path = input_args[0]
            input_args = []

    # Expand glob patterns in input arguments
    input_paths = []
    if input_args:
        input_paths = expand_input_paths(input_args)

    # Load YAML config if provided
    config = {}
    if config_path:
        from .config import load_yaml_config, merge_with_defaults, validate_config
        try:
            yaml_config = load_yaml_config(config_path)
            config = merge_with_defaults(yaml_config)

            issues = validate_config(config)
            if issues:
                for issue in issues:
                    print(f"{Colors.YELLOW}Warning: {issue}{Colors.END}")
        except FileNotFoundError as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
            sys.exit(1)

    # CLI arguments override YAML config
    if args.domain:
        config['domain'] = args.domain
    if args.output != './output':
        config['output_dir'] = args.output
    if args.format != 'npz':
        config['output_format'] = args.format
    if args.verbose:
        config['verbose'] = True
    if args.stages:
        config['stages'] = args.stages.split(',')

    if 'output_format' not in config:
        config['output_format'] = args.format

    # Build download config from CLI args
    if hasattr(args, 'download') and args.download:
        config['download'] = {
            'enabled': True,
            'source': args.source,
            'variable': args.variable,
            'model': args.model,
            'experiment': args.experiment,
            'frequency': args.frequency,
            'year': args.year,
        }
        # Remove None values
        config['download'] = {k: v for k, v in config['download'].items() if v is not None}

    verbose = args.verbose or config.get('verbose', False)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle input from config if not provided on CLI
    if not input_paths:
        config_input = config.get('input_path') or config.get('input')
        config_input_paths = config.get('input_paths')

        if config_input_paths:
            input_paths = expand_input_paths(config_input_paths)
        elif config_input:
            input_paths = expand_input_paths([config_input])

    # Check for input paths (not required if download mode is enabled)
    download_enabled = config.get('download', {}).get('enabled', False)
    if not input_paths and not download_enabled:
        print(f"{Colors.RED}Error: input file path(s) required. Provide on CLI or set "
              f"'input_path' or 'input_paths' in the YAML config.{Colors.END}")
        print(f"{Colors.DIM}Tip: Use --download to fetch data from ESGF or CDS{Colors.END}")
        sys.exit(1)

    # Run pipeline
    stages = config.get('stages')
    if args.stages:
        stages = args.stages.split(",")

    # Array mode settings
    array_mode = args.array if hasattr(args, 'array') else config.get('array_mode', False)
    normalize_method = args.normalize if hasattr(args, 'normalize') else config.get('normalize_method')

    verify = args.verify if hasattr(args, 'verify') else config.get('verify', False)

    ctx = run_pipeline(
        output_dir=config.get('output_dir', args.output),
        domain=config.get('domain', args.domain),
        stages=stages,
        config=config,
        verbose=verbose,
        style=args.style,
        input_paths=[str(p) for p in input_paths],
        array_mode=array_mode,
        normalize_method=normalize_method,
        verify=verify
    )



def _handle_sidecar_command(args):
    """Handle the 'sidecar' subcommand."""
    import yaml

    config = {}

    # Load YAML config if provided
    if args.config:
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"{Colors.RED}Error: Config file not found: {args.config}{Colors.END}")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
            sys.exit(1)

    # CLI args override config
    workflow_file = args.workflow or config.get('workflow')
    if not workflow_file:
        print(f"{Colors.RED}Error: workflow file required. Use -w/--workflow or set in config.{Colors.END}")
        sys.exit(1)

    workflow_type = args.workflow_type if args.workflow_type != 'auto' else config.get('workflow_type', 'auto')
    output_dir = args.output if args.output != './output' else config.get('output_dir', './output')
    watch = args.watch or config.get('watch', False)
    style = args.style if args.style != 'rich' else config.get('style', 'rich')
    data_dir = args.data_dir or config.get('data_dir')

    run_sidecar(
        workflow_file=workflow_file,
        workflow_type=workflow_type,
        output_dir=output_dir,
        watch=watch,
        style=style,
        config=config,
        data_dir=data_dir,
    )


def _handle_discover_command(args):
    """Handle the 'discover' subcommand."""
    import yaml

    config = {}

    # Load YAML config if provided
    if args.config:
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"{Colors.RED}Error: Config file not found: {args.config}{Colors.END}")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
            sys.exit(1)

    # CLI args override config
    input_path = args.input or config.get('input')
    if not input_path:
        print(f"{Colors.RED}Error: input path required. Provide on CLI or set in config.{Colors.END}")
        sys.exit(1)

    output_dir = args.output if args.output != './output' else config.get('output_dir', './output')
    style = args.style if args.style != 'rich' else config.get('style', 'rich')
    interactive = not args.no_interactive if args.no_interactive else config.get('interactive', True)

    run_discover(
        input_path=input_path,
        output_dir=output_dir,
        style=style,
        interactive=interactive,
        config=config,
    )


if __name__ == "__main__":
    main()
