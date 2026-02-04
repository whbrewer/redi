#!/usr/bin/env nextflow
/*
 * Bioinformatics Pipeline Example
 *
 * A realistic bioinformatics workflow demonstrating nf-core style patterns.
 * Processes protein sequences through QC, MSA, and structure prediction.
 */

nextflow.enable.dsl=2

// Parameters
params.input = "data/sequences.fasta"
params.output_dir = "output"
params.database = "uniref90"
params.skip_msa = false
params.max_cpus = 4

// Process: Validate input sequences
process VALIDATE_SEQUENCES {
    tag "${meta.id}"
    label 'process_low'

    container 'biocontainers/seqkit:2.3.1--h9ee0642_0'

    input:
    tuple val(meta), path(fasta)

    output:
    tuple val(meta), path("validated_*.fasta"), emit: sequences
    tuple val(meta), path("seq_stats.json"), emit: stats

    script:
    def prefix = meta.id ?: "sample"
    """
    # Validate FASTA format
    seqkit stats ${fasta} -T > seq_stats.tsv

    # Convert to JSON
    echo '{"num_seqs": '"\$(tail -1 seq_stats.tsv | cut -f4)"', "total_len": '"\$(tail -1 seq_stats.tsv | cut -f5)"'}' > seq_stats.json

    # Copy validated sequences
    cp ${fasta} validated_${prefix}.fasta
    """
}

// Process: Run sequence QC
process FASTQC {
    tag "${meta.id}"
    label 'process_low'

    container "${ workflow.containerEngine == 'singularity' ?
        'https://depot.galaxyproject.org/singularity/fastqc:0.12.1--hdfd78af_0' :
        'biocontainers/fastqc:0.12.1--hdfd78af_0' }"

    input:
    tuple val(meta), path(sequences)

    output:
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.zip"), emit: zip

    when:
    !params.skip_msa

    script:
    def args = task.ext.args ?: ''
    def prefix = meta.id ?: "sample"
    """
    # Note: FastQC is typically for FASTQ, this is illustrative
    echo "QC report for ${prefix}" > ${prefix}_qc_report.html
    zip ${prefix}_qc.zip ${prefix}_qc_report.html
    """
}

// Process: Generate MSA
process RUN_MSA {
    tag "${meta.id}"
    label 'process_medium'

    container 'biocontainers/clustalo:1.2.4--h87f3376_5'

    input:
    tuple val(meta), path(sequences)

    output:
    tuple val(meta), path("*.a3m"), emit: msa
    tuple val(meta), path("msa_stats.json"), emit: stats

    script:
    def prefix = meta.id ?: "sample"
    def threads = task.cpus ?: 1
    """
    # Run Clustal Omega for MSA
    clustalo -i ${sequences} -o ${prefix}.a3m --threads=${threads} --force

    # Generate stats
    num_seqs=\$(grep -c "^>" ${sequences})
    echo '{"num_sequences": '\$num_seqs', "alignment_format": "a3m"}' > msa_stats.json
    """
}

// Process: Search templates
process SEARCH_TEMPLATES {
    tag "${meta.id}"
    label 'process_high'

    container 'biocontainers/hhsuite:3.3.0--py39pl5321h67e14b5_5'

    input:
    tuple val(meta), path(msa)

    output:
    tuple val(meta), path("*.hhr"), emit: templates
    tuple val(meta), path("template_hits.json"), emit: hits

    script:
    def prefix = meta.id ?: "sample"
    def database = params.database
    """
    # Search for templates (simplified - normally uses hhblits)
    echo "Template search results for ${prefix}" > ${prefix}.hhr
    echo '{"num_hits": 10, "database": "${database}"}' > template_hits.json
    """
}

// Process: Predict structure
process PREDICT_STRUCTURE {
    tag "${meta.id}"
    label 'process_high'

    container 'nvcr.io/nvidia/clara/alphafold:2.3.1'

    input:
    tuple val(meta), path(msa), path(templates)

    output:
    tuple val(meta), path("ranked_*.pdb"), emit: structures
    tuple val(meta), path("ranking_debug.json"), emit: ranking
    tuple val(meta), path("timings.json"), emit: timings

    script:
    def prefix = meta.id ?: "sample"
    """
    # Simulate structure prediction
    echo "HEADER PREDICTED STRUCTURE" > ranked_0_${prefix}.pdb
    echo "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C" >> ranked_0_${prefix}.pdb
    echo "END" >> ranked_0_${prefix}.pdb

    echo '{"order": [0], "plddts": {"ranked_0": 85.5}}' > ranking_debug.json
    echo '{"prediction": 120.5, "relaxation": 30.2}' > timings.json
    """
}

// Process: Relax structure
process AMBER_RELAX {
    tag "${meta.id}"
    label 'process_medium'

    container 'nvcr.io/nvidia/clara/alphafold:2.3.1'

    input:
    tuple val(meta), path(structure)

    output:
    tuple val(meta), path("relaxed_*.pdb"), emit: relaxed

    script:
    def prefix = meta.id ?: "sample"
    """
    # Simulate AMBER relaxation
    cp ${structure} relaxed_${prefix}.pdb
    """
}

// Process: Aggregate reports
process MULTIQC {
    tag "multiqc"
    label 'process_low'
    publishDir params.output_dir, mode: 'copy'

    container "${ workflow.containerEngine == 'singularity' ?
        'https://depot.galaxyproject.org/singularity/multiqc:1.14--pyhdfd78af_0' :
        'biocontainers/multiqc:1.14--pyhdfd78af_0' }"

    input:
    path(qc_files)
    path(stats_files)

    output:
    path("multiqc_report.html"), emit: report
    path("multiqc_data"), emit: data

    script:
    """
    mkdir -p multiqc_data

    # Create summary report
    echo "<html><body><h1>MultiQC Report</h1>" > multiqc_report.html
    echo "<h2>Processed Files</h2><ul>" >> multiqc_report.html
    for f in ${qc_files}; do
        echo "<li>\$f</li>" >> multiqc_report.html
    done
    echo "</ul></body></html>" >> multiqc_report.html

    # Create data summary
    echo '{"report_type": "multiqc", "files_processed": '"\$(echo ${qc_files} | wc -w)"'}' > multiqc_data/summary.json
    """
}

// Subworkflow: Structure prediction pipeline
workflow STRUCTURE_PREDICTION {
    take:
    sequences

    main:
    RUN_MSA(sequences)
    SEARCH_TEMPLATES(RUN_MSA.out.msa)
    PREDICT_STRUCTURE(
        RUN_MSA.out.msa.join(SEARCH_TEMPLATES.out.templates)
    )
    AMBER_RELAX(PREDICT_STRUCTURE.out.structures)

    emit:
    structures = AMBER_RELAX.out.relaxed
    msa = RUN_MSA.out.msa
    rankings = PREDICT_STRUCTURE.out.ranking
}

// Main workflow
workflow {
    // Create input channel with metadata
    Channel
        .fromPath(params.input)
        .map { file ->
            def meta = [id: file.baseName]
            [meta, file]
        }
        .set { input_ch }

    // Validate sequences
    VALIDATE_SEQUENCES(input_ch)

    // Run QC
    FASTQC(VALIDATE_SEQUENCES.out.sequences)

    // Run structure prediction pipeline
    STRUCTURE_PREDICTION(VALIDATE_SEQUENCES.out.sequences)

    // Collect QC files for MultiQC
    qc_files = FASTQC.out.zip.map { meta, file -> file }.collect()
    stats_files = VALIDATE_SEQUENCES.out.stats.map { meta, file -> file }.collect()

    MULTIQC(qc_files, stats_files)
}
