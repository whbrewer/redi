"""Tests for Nextflow workflow parser."""
import pytest
from pathlib import Path
import tempfile

from redi.sidecar.nextflow_parser import (
    parse_nextflow_file,
    map_processes_to_stages,
    NextflowProcess,
    NextflowParseResult,
)


SAMPLE_WORKFLOW = '''
#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input = "data/*.fastq"
params.outdir = "results"

process DOWNLOAD {
    container 'ubuntu:latest'

    input:
    val(id)

    output:
    path("*.txt")

    script:
    """
    wget http://example.com/${id}.txt
    """
}

process ALIGN {
    container 'biocontainers/bwa:latest'

    input:
    path(reads)

    output:
    path("aligned.bam")

    script:
    """
    bwa mem ref.fa ${reads} > aligned.bam
    """
}

process PREDICT_STRUCTURE {
    container 'alphafold:latest'

    input:
    path(sequence)

    output:
    path("model.pdb")

    script:
    """
    python predict.py ${sequence}
    """
}

workflow {
    ids = Channel.of('sample1', 'sample2')
    DOWNLOAD(ids)
    ALIGN(DOWNLOAD.out)
    PREDICT_STRUCTURE(ALIGN.out)
}

workflow SUBWORKFLOW {
    take:
    input_ch

    main:
    ALIGN(input_ch)

    emit:
    aligned = ALIGN.out
}
'''


@pytest.fixture
def sample_nf_file(tmp_path):
    """Create a sample Nextflow file for testing."""
    nf_path = tmp_path / "main.nf"
    nf_path.write_text(SAMPLE_WORKFLOW)
    return nf_path


def test_parse_nextflow_file(sample_nf_file):
    """Test parsing a Nextflow file."""
    result = parse_nextflow_file(sample_nf_file)

    assert isinstance(result, NextflowParseResult)
    assert result.dsl_version == "2"
    assert len(result.processes) == 3
    assert len(result.workflows) == 2
    assert len(result.errors) == 0


def test_parse_processes(sample_nf_file):
    """Test that processes are parsed correctly."""
    result = parse_nextflow_file(sample_nf_file)

    proc_names = [p.name for p in result.processes]
    assert "DOWNLOAD" in proc_names
    assert "ALIGN" in proc_names
    assert "PREDICT_STRUCTURE" in proc_names

    # Check DOWNLOAD process details
    download = next(p for p in result.processes if p.name == "DOWNLOAD")
    assert download.container == "ubuntu:latest"
    assert any("val:id" in inp for inp in download.inputs)
    assert any("*.txt" in out for out in download.outputs)
    assert download.script is not None
    assert "wget" in download.script


def test_parse_scripts(sample_nf_file):
    """Test that scripts are extracted from triple-quoted strings."""
    result = parse_nextflow_file(sample_nf_file)

    align = next(p for p in result.processes if p.name == "ALIGN")
    assert align.script is not None
    assert "bwa mem" in align.script


def test_parse_workflows(sample_nf_file):
    """Test that workflows are parsed correctly."""
    result = parse_nextflow_file(sample_nf_file)

    wf_names = [w.name for w in result.workflows]
    assert "workflow" in wf_names or "main" in wf_names
    assert "SUBWORKFLOW" in wf_names

    # Check SUBWORKFLOW
    subwf = next(w for w in result.workflows if w.name == "SUBWORKFLOW")
    assert "input_ch" in subwf.inputs
    assert any("aligned" in out for out in subwf.outputs)


def test_parse_params(sample_nf_file):
    """Test that params are parsed correctly."""
    result = parse_nextflow_file(sample_nf_file)

    assert "input" in result.params
    assert "outdir" in result.params
    assert result.params["outdir"] == '"results"'


def test_map_processes_to_stages(sample_nf_file):
    """Test mapping processes to REDI stages."""
    result = parse_nextflow_file(sample_nf_file)
    stage_mapping = map_processes_to_stages(result.processes)

    assert "ingest" in stage_mapping
    assert "preprocess" in stage_mapping
    assert "transform" in stage_mapping

    # DOWNLOAD should map to ingest
    assert "DOWNLOAD" in stage_mapping["ingest"]

    # ALIGN should map to preprocess
    assert "ALIGN" in stage_mapping["preprocess"]

    # PREDICT_STRUCTURE should map to transform (due to "predict" keyword)
    assert "PREDICT_STRUCTURE" in stage_mapping["transform"]


def test_infer_process_description(sample_nf_file):
    """Test that descriptions are inferred from process names."""
    result = parse_nextflow_file(sample_nf_file)

    download = next(p for p in result.processes if p.name == "DOWNLOAD")
    assert "download" in download.description.lower()

    align = next(p for p in result.processes if p.name == "ALIGN")
    assert "align" in align.description.lower()

    predict = next(p for p in result.processes if p.name == "PREDICT_STRUCTURE")
    assert "predict" in predict.description.lower()


def test_parse_missing_file():
    """Test handling of missing file."""
    result = parse_nextflow_file(Path("/nonexistent/file.nf"))

    assert len(result.errors) > 0
    assert "not found" in result.errors[0].lower()


def test_empty_workflow():
    """Test parsing an empty/minimal workflow."""
    with tempfile.NamedTemporaryFile(suffix=".nf", mode="w", delete=False) as f:
        f.write("// Empty workflow\n")
        f.flush()

        result = parse_nextflow_file(Path(f.name))

        assert len(result.processes) == 0
        assert len(result.workflows) == 0
        assert len(result.errors) == 0


# nf-core style module test
NFCORE_MODULE = '''
process FASTQC {
    tag "${meta.id}"
    label 'process_low'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/fastqc:0.12.1--hdfd78af_0' :
        'biocontainers/fastqc:0.12.1--hdfd78af_0' }"

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.zip"), emit: zip

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    fastqc --threads ${task.cpus} ${args} ${reads}
    """

    stub:
    """
    touch ${prefix}.html
    touch ${prefix}.zip
    """
}
'''


@pytest.fixture
def nfcore_module_file(tmp_path):
    """Create an nf-core style module for testing."""
    nf_path = tmp_path / "fastqc.nf"
    nf_path.write_text(NFCORE_MODULE)
    return nf_path


def test_nfcore_module_parsing(nfcore_module_file):
    """Test parsing an nf-core style module."""
    result = parse_nextflow_file(nfcore_module_file)

    assert len(result.processes) == 1
    proc = result.processes[0]

    assert proc.name == "FASTQC"
    assert "quality control" in proc.description.lower()


def test_nfcore_container_extraction(nfcore_module_file):
    """Test that container is extracted from nf-core ternary expression."""
    result = parse_nextflow_file(nfcore_module_file)
    proc = result.processes[0]

    # Should extract either the Docker or Singularity container
    assert proc.container is not None
    assert "fastqc" in proc.container.lower()


def test_nfcore_tuple_io_parsing(nfcore_module_file):
    """Test parsing tuple input/output declarations."""
    result = parse_nextflow_file(nfcore_module_file)
    proc = result.processes[0]

    # Input should extract 'reads' from tuple val(meta), path(reads)
    assert "reads" in proc.inputs

    # Output should extract *.html and *.zip patterns
    assert any("*.html" in out for out in proc.outputs)
    assert any("*.zip" in out for out in proc.outputs)


def test_nfcore_script_with_groovy(nfcore_module_file):
    """Test that script is extracted when Groovy code precedes it."""
    result = parse_nextflow_file(nfcore_module_file)
    proc = result.processes[0]

    assert proc.script is not None
    assert "fastqc" in proc.script
    assert "--threads" in proc.script


def test_nfcore_stage_mapping(nfcore_module_file):
    """Test that FASTQC maps to preprocess stage."""
    result = parse_nextflow_file(nfcore_module_file)
    stages = map_processes_to_stages(result.processes)

    assert "FASTQC" in stages["preprocess"]


def test_multiqc_maps_to_output():
    """Test that MULTIQC maps to output stage, not preprocess."""
    from redi.sidecar.nextflow_parser import NextflowProcess

    processes = [
        NextflowProcess(name="MULTIQC", description="Aggregate QC reports"),
        NextflowProcess(name="FASTQC", description="Sequence QC"),
    ]
    stages = map_processes_to_stages(processes)

    assert "MULTIQC" in stages["output"]
    assert "FASTQC" in stages["preprocess"]
