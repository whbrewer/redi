#!/usr/bin/env nextflow
/*
 * Simple Nextflow Workflow Example
 *
 * A minimal DSL2 workflow demonstrating REDI sidecar functionality.
 * This workflow reads a CSV file, processes it, and outputs results.
 */

nextflow.enable.dsl=2

// Parameters
params.input = "data/sample_data.csv"
params.output_dir = "output"
params.normalize = true

// Process: Load and validate input data
process LOAD_DATA {
    tag "load"

    input:
    path(csv_file)

    output:
    path("validated_*.csv")

    script:
    """
    # Validate CSV has required columns
    head -1 ${csv_file} > header.txt

    # Copy with validation marker
    cp ${csv_file} validated_data.csv

    echo "Loaded \$(wc -l < ${csv_file}) rows"
    """
}

// Process: Compute statistics and normalize
process NORMALIZE_DATA {
    tag "normalize"

    input:
    path(data)

    output:
    path("normalized_*.csv")
    path("stats.json")

    script:
    """
    # Compute mean and std (simplified)
    echo '{"mean": 0.5, "std": 0.2}' > stats.json

    # Output normalized data
    cp ${data} normalized_data.csv

    echo "Normalized data saved"
    """
}

// Process: Split into train/val/test
process SPLIT_DATA {
    tag "split"

    input:
    path(data)

    output:
    path("train.csv")
    path("val.csv")
    path("test.csv")

    script:
    """
    # Simple split (in practice, use proper splitting)
    total=\$(wc -l < ${data})
    train_end=\$((total * 70 / 100))
    val_end=\$((total * 85 / 100))

    head -1 ${data} > train.csv
    head -1 ${data} > val.csv
    head -1 ${data} > test.csv

    tail -n +2 ${data} | head -n \$train_end >> train.csv
    tail -n +2 ${data} | tail -n +\$((train_end + 1)) | head -n \$((val_end - train_end)) >> val.csv
    tail -n +2 ${data} | tail -n +\$((val_end + 1)) >> test.csv

    echo "Split complete: train=\$train_end, val=\$((val_end - train_end)), test=\$((total - val_end))"
    """
}

// Process: Generate summary report
process GENERATE_REPORT {
    tag "report"
    publishDir params.output_dir, mode: 'copy'

    input:
    path(train)
    path(val)
    path(test)
    path(stats)

    output:
    path("summary_report.txt")

    script:
    """
    echo "Data Processing Summary" > summary_report.txt
    echo "=======================" >> summary_report.txt
    echo "" >> summary_report.txt
    echo "Train samples: \$(wc -l < ${train})" >> summary_report.txt
    echo "Val samples: \$(wc -l < ${val})" >> summary_report.txt
    echo "Test samples: \$(wc -l < ${test})" >> summary_report.txt
    echo "" >> summary_report.txt
    echo "Statistics:" >> summary_report.txt
    cat ${stats} >> summary_report.txt
    """
}

// Main workflow
workflow {
    // Input channel
    input_ch = Channel.fromPath(params.input)

    // Pipeline
    LOAD_DATA(input_ch)
    NORMALIZE_DATA(LOAD_DATA.out)
    SPLIT_DATA(NORMALIZE_DATA.out[0])
    GENERATE_REPORT(
        SPLIT_DATA.out[0],
        SPLIT_DATA.out[1],
        SPLIT_DATA.out[2],
        NORMALIZE_DATA.out[1]
    )
}
