"""
redi/discover — analyze unknown datasets and propose pipelines.
"""

import json
from pathlib import Path
from typing import Optional

from redi.discover.processors import registry, FileDiscovery, ProcessorError
from redi.discover.report import render_processing_plan


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


def _print_banner():
    print(f"""{Colors.BOLD}{Colors.CYAN}
 ██████╗ ███████╗██████╗ ██╗
 ██╔══██╗██╔════╝██╔══██╗██║
 ██████╔╝█████╗  ██║  ██║██║
 ██╔══██╗██╔══╝  ██║  ██║██║
 ██║  ██║███████╗██████╔╝██║
 ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝
{Colors.END}{Colors.DIM} Readiness Engine for Data Integration{Colors.END}
""")


def _print_section(title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}▸ {title}{Colors.END}")


def _print_metric(label: str, value: str, indent: int = 4):
    print(f"{' ' * indent}{Colors.DIM}{label}:{Colors.END} {Colors.BOLD}{value}{Colors.END}")


# =============================================================================
# Public entry point
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
        _print_banner()
        _print_section("Discovery Mode")
        _print_metric("Path", str(path))
        _print_metric("Type", "directory" if path.is_dir() else "file")

    # Phase 1: Inspect
    if style == "rich":
        _print_section("Phase 1: Inspect")

    inspection = _inspect_data(path, style)

    # Phase 2: Infer
    if style == "rich":
        _print_section("Phase 2: Infer")

    inference = _infer_characteristics(inspection, style)

    # Phase 3: Propose
    if style == "rich":
        _print_section("Phase 3: Propose Pipeline")

    proposal = _propose_pipeline(inspection, inference, style)

    # Save discovery report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Pull FileDiscovery objects out before JSON serialisation (they are not
    # plain dicts and are handled separately by render_processing_plan).
    discoveries: list[FileDiscovery] = inspection.pop("discoveries", [])

    from redi.repro import record_run
    provenance = record_run(
        str(out_path), mode="discover", input_paths=[str(path)],
        metadata_filename="discovery_report.json",
    )

    report = {
        "input_path": str(path),
        "mode": "discover",
        "inspection": inspection,
        "inference": inference,
        "proposal": proposal,
        "provenance": provenance,
    }

    report_path = out_path / "discovery_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    md_report = _generate_discovery_markdown(report)
    md_path = out_path / "DISCOVERY_REPORT.md"
    md_path.write_text(md_report)

    # Write per-file processing plan
    report["discoveries"] = discoveries  # restore for return value
    if discoveries:
        plan_path = out_path / "processing_plan.md"
        render_processing_plan(discoveries, plan_path)

    if style == "rich":
        _print_section("Output")
        print(f"    📋 {Colors.BOLD}discovery_report.json{Colors.END} {Colors.DIM}(saved){Colors.END}")
        print(f"    📋 {Colors.BOLD}DISCOVERY_REPORT.md{Colors.END} {Colors.DIM}(saved){Colors.END}")
        if discoveries:
            print(f"    📋 {Colors.BOLD}processing_plan.md{Colors.END} {Colors.DIM}(saved){Colors.END}")
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


# =============================================================================
# Internal helpers
# =============================================================================

def _inspect_data(path: Path, style: str) -> dict:
    """Inspect data files and gather metadata."""
    inspection = {
        'files': [],
        'total_size': 0,
        'formats': set(),
        'sample_data': {},
        'discoveries': [],
    }

    # Extensions that are directories treated as single logical files
    _DIR_EXTENSIONS = {'.bp'}

    if path.is_file():
        files = [path]
    elif path.is_dir() and path.suffix.lower() in _DIR_EXTENSIONS:
        files = [path]
    else:
        files = []
        for item in path.rglob('*'):
            if item.is_dir() and item.suffix.lower() in _DIR_EXTENSIONS:
                files.append(item)
            elif item.is_file():
                # Skip files that live inside a .bp directory
                if not any(p.suffix.lower() in _DIR_EXTENSIONS for p in item.parents):
                    files.append(item)

    # Full inventory: size + formats over all files
    _ANALYSIS_CAP_PER_FORMAT = 10
    for f in files:
        if f.is_dir():
            size = sum(c.stat().st_size for c in f.rglob('*') if c.is_file())
        else:
            size = f.stat().st_size
        inspection['total_size'] += size
        inspection['formats'].add(f.suffix.lower())

    # Sample up to N files per format for analysis
    from collections import defaultdict
    _per_format: dict = defaultdict(list)
    for f in files:
        _per_format[f.suffix.lower()].append(f)
    files_to_analyze = []
    for fmt_files in _per_format.values():
        files_to_analyze.extend(fmt_files[:_ANALYSIS_CAP_PER_FORMAT])

    for f in files_to_analyze:
        file_info = {
            'path': str(f),
            'name': f.name,
            'suffix': f.suffix.lower(),
            'size': (sum(c.stat().st_size for c in f.rglob('*') if c.is_file())
                     if f.is_dir() else f.stat().st_size),
        }
        inspection['files'].append(file_info)

    inspection['formats'] = list(inspection['formats'])
    inspection['file_count'] = len(files)

    # Run per-file processor on sampled files
    processor_errors: list[ProcessorError] = []
    for f in files_to_analyze:
        try:
            discovery = registry.analyze_file(f)
            inspection['discoveries'].append(discovery)
        except ProcessorError as err:
            processor_errors.append(err)

    inspection['processor_errors'] = [
        {"file": str(e.path), "file_type": e.file_type, "reason": e.reason, "hint": e.hint}
        for e in processor_errors
    ]

    if processor_errors and style == "rich":
        print(f"\n    {Colors.YELLOW}⚠  {len(processor_errors)} file(s) could not be analysed:{Colors.END}")
        for err in processor_errors:
            print(f"      {Colors.DIM}•{Colors.END} {err.user_message}")
    elif processor_errors and style == "simple":
        for err in processor_errors:
            print(f"Warning: {err.user_message}")

    # Also sample first file of each format for the existing inference logic
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

    # Infer modality — also check discoveries for ADIOS2 BP fusion data
    if '.lmdb' in formats or '.extxyz' in formats or '.xz' in formats:
        inference['modality'] = 'graph'
        inference['likely_domain'] = 'materials'
        inference['suggested_format'] = 'hdf5'
        # Check discoveries for energy/forces labels
        for d in inspection.get('discoveries', []):
            meta = d.metadata if hasattr(d, 'metadata') else {}
            # extXYZ discoveries expose has_energy/has_forces directly
            if meta.get('has_energy') or meta.get('has_forces'):
                inference['has_labels'] = True
                break
            # LMDB discoveries expose schema.fields
            schema = meta.get('schema', {})
            fields = schema.get('fields', {})
            if 'energy' in fields or 'forces' in fields or 'y' in fields:
                inference['has_labels'] = True
                break
    elif '.bp' in formats:
        inference['modality'] = 'simulation'
        inference['suggested_format'] = 'hdf5'
        # Check variable names from discoveries to confirm fusion domain
        for d in inspection.get('discoveries', []):
            meta = d.metadata if hasattr(d, 'metadata') else {}
            if meta.get('has_electron_fields') or meta.get('has_ion_fields'):
                inference['likely_domain'] = 'fusion'
                break
        else:
            inference['likely_domain'] = 'fusion'  # .bp is most commonly fusion/HPC sim
    elif '.nc' in formats or '.nc4' in formats or '.grib' in formats:
        inference['modality'] = 'gridded'
        inference['likely_domain'] = 'climate'
        inference['suggested_format'] = 'zarr'
    elif '.fasta' in formats or '.pdb' in formats or '.a3m' in formats:
        inference['modality'] = 'sequence'
        inference['likely_domain'] = 'life_sciences'
        inference['suggested_format'] = 'hdf5'
        # .cif + .fasta + .a3m together is unambiguously protein structure
        if '.cif' in formats and ('.fasta' in formats or '.a3m' in formats):
            inference['confidence_override'] = 'high'
    elif '.poscar' in formats:
        inference['modality'] = 'structured'
        inference['likely_domain'] = 'materials'
        inference['suggested_format'] = 'hdf5'
    elif '.cif' in formats:
        # .cif alone is ambiguous: mmCIF (life sciences) vs CIF (materials)
        # Default to life sciences; user can override with --domain
        inference['modality'] = 'structured'
        inference['likely_domain'] = 'life_sciences'
        inference['suggested_format'] = 'hdf5'
    elif '.npy' in formats or '.npz' in formats:
        inference['modality'] = 'array'
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
        for fmt, sample in samples.items():
            cols = sample.get('columns', [])
            cols_lower = [c.lower() for c in cols]

            _hpc_keywords = ['job', 'queue', 'walltime', 'node', 'core', 'cpu', 'gpu',
                              'wait', 'submit', 'partition', 'scheduler', 'slurm', 'pbs']

            if any(c in cols_lower for c in ['temperature', 'lat', 'lon', 'latitude', 'longitude']):
                inference['likely_domain'] = 'climate'
            elif any(c in cols_lower for c in ['formula', 'lattice', 'spacegroup', 'energy']):
                inference['likely_domain'] = 'materials'
            elif any(c in cols_lower for c in ['sequence', 'protein', 'gene', 'mutation']):
                inference['likely_domain'] = 'life_sciences'
            elif any(c in cols_lower for c in ['plasma', 'tokamak', 'disruption', 'ip']):
                inference['likely_domain'] = 'fusion'
            elif any(kw in col for col in cols_lower for kw in _hpc_keywords):
                inference['likely_domain'] = 'hpc'

            _label_names = ['label', 'target', 'class', 'category', 'completed', 'success',
                             'failed', 'outcome', 'result']
            if any(any(ln in c for ln in _label_names) for c in cols_lower):
                inference['has_labels'] = True

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
        'confidence': inference.get('confidence_override', 'medium'),
        'notes': [],
    }

    modality = inference.get('modality', 'unknown')
    domain = inference.get('likely_domain', 'general')

    proposal['pipeline_steps'].append({
        'name': 'ingest',
        'description': f"Load {modality} data from {', '.join(inspection.get('formats', ['files']))}",
        'stage': 'ingest',
    })

    if modality == 'graph':
        proposal['pipeline_steps'].append({
            'name': 'build_graphs',
            'description': 'Convert atomic structures to radius graphs with node/edge features',
            'stage': 'preprocess',
        })
        proposal['pipeline_steps'].append({
            'name': 'normalize_targets',
            'description': 'Subtract reference energies, Z-score normalize energy and forces',
            'stage': 'transform',
        })
        proposal['output_format'] = 'hdf5'
        proposal['confidence'] = 'high'
    elif modality == 'simulation':
        proposal['pipeline_steps'].append({
            'name': 'extract_fields',
            'description': 'Extract and assemble physics field arrays from simulation output',
            'stage': 'preprocess',
        })
        proposal['pipeline_steps'].append({
            'name': 'normalize',
            'description': 'Z-score normalize each field independently',
            'stage': 'transform',
        })
        proposal['confidence'] = 'high'
    elif modality == 'tabular':
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

    if inference.get('likely_domain') == 'general':
        proposal['notes'].append("Could not confidently detect domain. Consider specifying with --domain.")
        proposal['confidence'] = 'low'

    if not inference.get('has_labels') and modality not in ('simulation', 'graph', 'sequence', 'gridded'):
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
