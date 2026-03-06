"""
Nextflow workflow parser for REDI inspect.

Parses Nextflow DSL2 workflows to extract:
- Process definitions (inputs, outputs, scripts)
- Workflow structure
- Include statements (module imports)
- Channel definitions
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class NextflowProcess:
    """A Nextflow process definition."""
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    script: Optional[str] = None
    container: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "script": self.script[:200] if self.script else None,
            "container": self.container,
            "description": self.description,
        }


@dataclass
class NextflowWorkflow:
    """A Nextflow workflow definition."""
    name: str
    processes: List[str] = field(default_factory=list)  # Process names called
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "processes": self.processes,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


@dataclass
class NextflowInclude:
    """An include statement."""
    module_name: str
    source_path: str
    alias: Optional[str] = None


@dataclass
class NextflowParseResult:
    """Result of parsing a Nextflow file."""
    file_path: str
    dsl_version: str = "2"
    processes: List[NextflowProcess] = field(default_factory=list)
    workflows: List[NextflowWorkflow] = field(default_factory=list)
    includes: List[NextflowInclude] = field(default_factory=list)
    params: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "dsl_version": self.dsl_version,
            "processes": [p.to_dict() for p in self.processes],
            "workflows": [w.to_dict() for w in self.workflows],
            "includes": [{"module": i.module_name, "from": i.source_path, "alias": i.alias} for i in self.includes],
            "params": self.params,
            "errors": self.errors,
        }


def parse_nextflow_file(file_path: Path) -> NextflowParseResult:
    """
    Parse a Nextflow workflow file (.nf).

    Args:
        file_path: Path to the .nf file

    Returns:
        NextflowParseResult with extracted information
    """
    result = NextflowParseResult(file_path=str(file_path))

    if not file_path.exists():
        result.errors.append(f"File not found: {file_path}")
        return result

    try:
        content = file_path.read_text()
    except Exception as e:
        result.errors.append(f"Could not read file: {e}")
        return result

    # Check DSL version
    dsl_match = re.search(r'nextflow\.enable\.dsl\s*=\s*(\d+)', content)
    if dsl_match:
        result.dsl_version = dsl_match.group(1)

    # Parse include statements
    result.includes = _parse_includes(content)

    # Parse process definitions
    result.processes = _parse_processes(content)

    # Parse workflow definitions
    result.workflows = _parse_workflows(content)

    # Parse params
    result.params = _parse_params(content)

    return result


def _parse_includes(content: str) -> List[NextflowInclude]:
    """Parse include statements from Nextflow content."""
    includes = []

    # Pattern: include { NAME } from './path'
    # Also handles: include { NAME as ALIAS } from './path'
    pattern = re.compile(
        r"include\s*\{\s*(\w+)(?:\s+as\s+(\w+))?\s*\}\s*from\s*['\"]([^'\"]+)['\"]",
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        module_name = match.group(1)
        alias = match.group(2)
        source_path = match.group(3)
        includes.append(NextflowInclude(
            module_name=module_name,
            source_path=source_path,
            alias=alias
        ))

    return includes


def _parse_processes(content: str) -> List[NextflowProcess]:
    """Parse process definitions from Nextflow content."""
    processes = []

    # Find process blocks using brace counting for proper nesting
    process_blocks = _extract_process_blocks(content)

    for name, body in process_blocks:
        process = NextflowProcess(name=name)

        # Parse input section
        input_match = re.search(r'input:\s*(.*?)(?=output:|script:|shell:|exec:|stub:|$)', body, re.DOTALL)
        if input_match:
            process.inputs = _parse_io_declarations(input_match.group(1))

        # Parse output section - stop at when:, script:, shell:, exec:, or stub:
        output_match = re.search(r'output:\s*(.*?)(?=when:|script:|shell:|exec:|stub:|$)', body, re.DOTALL)
        if output_match:
            process.outputs = _parse_io_declarations(output_match.group(1))

        # Parse script section - handle Groovy code before triple-quoted script
        # nf-core modules often have: script:\n def var = ...\n """actual script"""
        script_section = re.search(r'(?:script|shell):\s*(.*?)(?=\n\s*stub:|$)', body, re.DOTALL)
        if script_section:
            script_content = script_section.group(1)
            # Find the triple-quoted block within the script section
            script_match = re.search(r'"""(.*?)"""', script_content, re.DOTALL)
            if not script_match:
                script_match = re.search(r"'''(.*?)'''", script_content, re.DOTALL)
            if script_match:
                process.script = script_match.group(1).strip()

        # Parse container - handle multi-line ternary expressions
        # Pattern 1: Simple string: container 'image:tag'
        container_match = re.search(r"container\s*'([^']+)'", body)
        if not container_match:
            container_match = re.search(r'container\s*"([^$"][^"]*)"', body)
        if container_match:
            process.container = container_match.group(1)
        else:
            # Pattern 2: nf-core style ternary: container "${ ... ? 'url' : 'image' }"
            # Extract the second option (Docker image) which is typically cleaner
            container_match = re.search(
                r"container\s*\"\$\{[^}]+:\s*'([^']+)'",
                body, re.DOTALL
            )
            if container_match:
                process.container = container_match.group(1)
            else:
                # Try to get the Singularity URL (first option)
                container_match = re.search(
                    r"container\s*\"\$\{[^}]+\?\s*'([^']+)'",
                    body, re.DOTALL
                )
                if container_match:
                    process.container = container_match.group(1)

        # Infer description from name
        process.description = _infer_process_description(name, process.script)

        processes.append(process)

    return processes


def _extract_process_blocks(content: str) -> List[Tuple[str, str]]:
    """Extract process blocks using brace counting for proper nesting."""
    blocks = []

    # Find all process declarations
    process_starts = list(re.finditer(r'process\s+(\w+)\s*\{', content))

    for match in process_starts:
        name = match.group(1)
        start_pos = match.end() - 1  # Position of opening brace

        # Count braces to find matching close
        brace_count = 0
        end_pos = start_pos

        for i, char in enumerate(content[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i
                    break

        if end_pos > start_pos:
            body = content[start_pos + 1:end_pos]
            blocks.append((name, body))

    return blocks


def _parse_io_declarations(io_block: str) -> List[str]:
    """Parse input/output declarations."""
    declarations = []

    # Common patterns:
    # path "filename"
    # tuple val(x), path(y)
    # val variable
    # file pattern
    # tuple val(meta), path("*.html"), emit: html

    lines = io_block.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('//'):
            continue

        # Remove emit: and topic: directives for cleaner parsing
        line_clean = re.sub(r',\s*emit:\s*\w+', '', line)
        line_clean = re.sub(r',\s*topic:\s*\w+', '', line_clean)

        # Handle tuple declarations: tuple val(meta), path(reads)
        if line_clean.startswith('tuple'):
            # Extract all path() contents from tuple
            path_matches = re.findall(r'path\s*\(([^)]+)\)', line_clean)
            for pm in path_matches:
                # Clean up the path - remove quotes and stageAs directives
                clean_path = re.sub(r'["\']', '', pm)
                clean_path = re.sub(r',?\s*stageAs:.*', '', clean_path).strip()
                if clean_path:
                    declarations.append(clean_path)
            continue

        # Extract path declarations (standalone)
        path_match = re.search(r'path\s*\(?\s*["\']([^"\']+)["\']', line_clean)
        if path_match:
            declarations.append(path_match.group(1))
            continue

        # Also handle path(variable) without quotes
        path_var_match = re.search(r'path\s*\(\s*(\w+)', line_clean)
        if path_var_match:
            declarations.append(path_var_match.group(1))
            continue

        # Extract file declarations (older syntax)
        file_match = re.search(r'file\s*[("\']([^"\'()]+)["\')]', line_clean)
        if file_match:
            declarations.append(file_match.group(1))
            continue

        # Extract val declarations
        val_match = re.search(r'val\s*[(\s](\w+)', line_clean)
        if val_match:
            declarations.append(f"val:{val_match.group(1)}")
            continue

        # If line has content but didn't match, add as-is
        if line and not line.startswith('//'):
            # Clean up the line
            clean = re.sub(r'^\s*\w+\s*', '', line).strip()
            if clean:
                declarations.append(clean)

    return declarations


def _parse_workflows(content: str) -> List[NextflowWorkflow]:
    """Parse workflow definitions from Nextflow content."""
    workflows = []

    # Named workflow pattern: workflow NAME { ... }
    # Also matches anonymous workflow: workflow { ... }
    workflow_pattern = re.compile(
        r'workflow\s+(\w+)?\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}',
        re.MULTILINE | re.DOTALL
    )

    for match in workflow_pattern.finditer(content):
        name = match.group(1) or "main"
        body = match.group(2)

        workflow = NextflowWorkflow(name=name)

        # Find process calls (PROCESS_NAME(...) or PROCESS_NAME)
        process_calls = re.findall(r'\b([A-Z][A-Z0-9_]+)\s*(?:\(|\.out)', body)
        workflow.processes = list(set(process_calls))

        # Find take: section for inputs
        take_match = re.search(r'take:\s*(.*?)(?=main:|emit:|$)', body, re.DOTALL)
        if take_match:
            workflow.inputs = [line.strip() for line in take_match.group(1).strip().split('\n') if line.strip()]

        # Find emit: section for outputs
        emit_match = re.search(r'emit:\s*(.*?)(?=$)', body, re.DOTALL)
        if emit_match:
            workflow.outputs = [line.strip() for line in emit_match.group(1).strip().split('\n') if line.strip()]

        workflows.append(workflow)

    return workflows


def _parse_params(content: str) -> Dict[str, str]:
    """Parse params definitions."""
    params = {}

    # Pattern: params.name = value
    param_pattern = re.compile(r'params\.(\w+)\s*=\s*([^\n]+)')

    for match in param_pattern.finditer(content):
        name = match.group(1)
        value = match.group(2).strip().rstrip(';')
        params[name] = value

    return params


def _infer_process_description(name: str, script: Optional[str]) -> str:
    """Infer a description for a process based on its name and script."""
    name_lower = name.lower()

    # Common process name patterns - organized by category
    # Data acquisition
    if 'download' in name_lower or 'fetch' in name_lower:
        return "Download data from remote source"
    # Quality control
    elif 'fastqc' in name_lower:
        return "Sequence quality control"
    elif 'multiqc' in name_lower:
        return "Aggregate QC reports"
    elif 'qualimap' in name_lower:
        return "Alignment quality control"
    elif 'qc' in name_lower:
        return "Quality control"
    # Trimming/filtering
    elif 'fastp' in name_lower or 'trim' in name_lower or 'cutadapt' in name_lower:
        return "Quality trimming"
    # Alignment
    elif 'bwa' in name_lower:
        return "BWA sequence alignment"
    elif 'bowtie' in name_lower:
        return "Bowtie sequence alignment"
    elif 'star' in name_lower:
        return "STAR RNA-seq alignment"
    elif 'hisat' in name_lower:
        return "HISAT2 alignment"
    elif 'align' in name_lower or 'msa' in name_lower:
        return "Sequence alignment"
    # Samtools/BAM processing
    elif 'samtools' in name_lower:
        if 'sort' in name_lower:
            return "Sort BAM file"
        elif 'index' in name_lower:
            return "Index BAM file"
        elif 'merge' in name_lower:
            return "Merge BAM files"
        elif 'view' in name_lower:
            return "Convert/filter BAM"
        else:
            return "SAMtools processing"
    # Variant calling
    elif 'gatk' in name_lower or 'haplotype' in name_lower:
        return "GATK variant calling"
    elif 'bcftools' in name_lower:
        return "BCFtools variant processing"
    elif 'variant' in name_lower or 'snp' in name_lower:
        return "Variant calling"
    # Structure prediction
    elif 'alphafold' in name_lower:
        return "AlphaFold structure prediction"
    elif 'colabfold' in name_lower:
        return "ColabFold structure prediction"
    elif 'esmfold' in name_lower:
        return "ESMFold structure prediction"
    elif 'predict' in name_lower or 'inference' in name_lower:
        return "Run prediction"
    elif 'relax' in name_lower or 'amber' in name_lower:
        return "Molecular dynamics relaxation"
    # Indexing
    elif 'index' in name_lower:
        return "Build search index"
    elif 'search' in name_lower:
        return "Search sequence database"
    elif 'template' in name_lower:
        return "Template search"
    # Feature/quantification
    elif 'feature' in name_lower or 'count' in name_lower:
        return "Feature counting"
    elif 'quant' in name_lower or 'salmon' in name_lower or 'kallisto' in name_lower:
        return "Transcript quantification"
    # Annotation
    elif 'annot' in name_lower or 'vep' in name_lower or 'snpeff' in name_lower:
        return "Variant annotation"
    # Assembly
    elif 'assembl' in name_lower or 'spades' in name_lower:
        return "Sequence assembly"
    # Other
    elif 'combine' in name_lower or 'merge' in name_lower:
        return "Combine results"
    elif 'convert' in name_lower:
        return "Format conversion"
    elif 'sort' in name_lower:
        return "Sort data"
    elif 'filter' in name_lower:
        return "Filter data"
    else:
        return "Processing step"


def map_processes_to_stages(processes: List[NextflowProcess]) -> Dict[str, List[str]]:
    """Map Nextflow processes to REDI pipeline stages."""
    stage_mapping = {
        'ingest': [],
        'preprocess': [],
        'transform': [],
        'structure': [],
        'output': [],
    }

    for process in processes:
        name_lower = process.name.lower()
        desc_lower = (process.description or '').lower()

        # Output stage first (check specific tools before generic patterns)
        if any(kw in name_lower for kw in [
            'multiqc', 'report', 'output', 'combine', 'aggregate', 'publish'
        ]):
            stage_mapping['output'].append(process.name)
        # Ingest stage: data acquisition
        elif any(kw in name_lower for kw in ['download', 'fetch', 'input', 'samplesheet', 'sra']):
            stage_mapping['ingest'].append(process.name)
        # Transform stage: quantification, variant calling, prediction
        elif any(kw in name_lower for kw in [
            'predict', 'inference', 'fold', 'alphafold', 'esmfold', 'colabfold',
            'count', 'quant', 'salmon', 'kallisto', 'featurecount',
            'gatk', 'haplotype', 'variant', 'bcftools', 'snp',
            'assembl', 'spades'
        ]):
            stage_mapping['transform'].append(process.name)
        # Structure stage: refinement, relaxation, annotation
        elif any(kw in name_lower for kw in [
            'relax', 'amber', 'refine', 'annot', 'vep', 'snpeff'
        ]):
            stage_mapping['structure'].append(process.name)
        # Preprocess stage: QC, trimming, alignment, indexing
        elif any(kw in name_lower for kw in [
            'fastqc', 'qc', 'fastp', 'trim', 'cutadapt', 'filter',
            'bwa', 'bowtie', 'star', 'hisat', 'align', 'msa',
            'samtools', 'sort', 'index', 'merge', 'markdup',
            'search', 'template'
        ]):
            stage_mapping['preprocess'].append(process.name)
        else:
            # Default to transform for unknown processes
            stage_mapping['transform'].append(process.name)

    return stage_mapping


def find_nextflow_logs(workflow_dir: Path) -> List[Path]:
    """Find Nextflow execution logs and traces."""
    logs = []

    # Check for .nextflow.log
    nextflow_log = workflow_dir / ".nextflow.log"
    if nextflow_log.exists():
        logs.append(nextflow_log)

    # Check for trace files
    trace_files = list(workflow_dir.glob("trace*.txt"))
    logs.extend(sorted(trace_files, key=lambda p: p.stat().st_mtime, reverse=True))

    # Check in work directory
    work_dir = workflow_dir / "work"
    if work_dir.exists():
        # Nextflow stores logs in work/.command.log files
        command_logs = list(work_dir.rglob(".command.log"))
        logs.extend(sorted(command_logs, key=lambda p: p.stat().st_mtime, reverse=True)[:10])

    return logs
