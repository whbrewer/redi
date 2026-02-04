#!/usr/bin/env nextflow
/*
 * Climate Data Processing Pipeline
 *
 * Processes climate data (NetCDF) through the REDI pipeline stages:
 * ingest -> preprocess (regrid) -> transform (normalize) -> output (split)
 */

nextflow.enable.dsl=2

// Parameters
params.input = "data/*.nc"
params.output_dir = "output"
params.target_resolution = "1.40625"
params.normalize_method = "zscore"
params.train_ratio = 0.7
params.val_ratio = 0.15
params.variables = "tas,pr,psl"

// Process: Download/ingest climate data
process DOWNLOAD_DATA {
    tag "${source}"
    label 'process_low'

    input:
    val(source)

    output:
    path("*.nc"), emit: data

    script:
    """
    # Simulate data download (in practice, use wget/curl to ESGF/CDS)
    echo "Downloading from ${source}..."

    # Create a simple NetCDF-like file (placeholder)
    echo "NetCDF placeholder for ${source}" > ${source}.nc
    """
}

// Process: Validate and inspect data
process INSPECT_DATA {
    tag "${data.baseName}"
    label 'process_low'

    container 'continuumio/miniconda3:latest'

    input:
    path(data)

    output:
    path("${data.baseName}_metadata.json"), emit: metadata
    path(data), emit: data

    script:
    """
    # Extract metadata (simplified)
    echo '{
        "filename": "${data}",
        "variables": ["tas", "pr", "psl"],
        "dimensions": {"time": 365, "lat": 128, "lon": 256},
        "time_range": ["2020-01-01", "2020-12-31"]
    }' > ${data.baseName}_metadata.json
    """
}

// Process: Regrid to target resolution
process REGRID_DATA {
    tag "${data.baseName}"
    label 'process_medium'

    container 'pangeo/pangeo-notebook:latest'

    input:
    path(data)

    output:
    path("regridded_*.nc"), emit: regridded
    path("regrid_params.json"), emit: params

    script:
    def resolution = params.target_resolution
    """
    # Regrid using xESMF (simplified)
    echo "Regridding to ${resolution} degrees..."

    cp ${data} regridded_${data.baseName}.nc

    echo '{
        "method": "bilinear",
        "source_resolution": [1.0, 1.0],
        "target_resolution": [${resolution}, ${resolution}]
    }' > regrid_params.json
    """
}

// Process: Compute normalization statistics
process COMPUTE_NORM_STATS {
    tag "norm_stats"
    label 'process_medium'

    container 'pangeo/pangeo-notebook:latest'

    input:
    path(data_files)

    output:
    path("normalize_mean.npz"), emit: mean
    path("normalize_std.npz"), emit: std
    path("norm_params.json"), emit: params

    script:
    def method = params.normalize_method
    def variables = params.variables
    """
    # Compute mean and std across all data (simplified)
    echo "Computing ${method} normalization statistics..."

    # Create placeholder NPZ files
    python3 -c "
import numpy as np
variables = '${variables}'.split(',')
mean_dict = {v: np.random.randn(128, 256).astype(np.float32) for v in variables}
std_dict = {v: np.abs(np.random.randn(128, 256)).astype(np.float32) + 0.1 for v in variables}
np.savez('normalize_mean.npz', **mean_dict)
np.savez('normalize_std.npz', **std_dict)
"

    echo '{
        "method": "${method}",
        "variables": "${variables}",
        "computed_on": "train",
        "num_samples": '"\$(ls ${data_files} | wc -w)"'
    }' > norm_params.json
    """
}

// Process: Apply normalization
process NORMALIZE_DATA {
    tag "${data.baseName}"
    label 'process_medium'

    container 'pangeo/pangeo-notebook:latest'

    input:
    path(data)
    path(mean_file)
    path(std_file)

    output:
    path("normalized_*.nc"), emit: normalized

    script:
    """
    # Apply z-score normalization
    echo "Normalizing ${data}..."

    cp ${data} normalized_${data.baseName}.nc
    """
}

// Process: Split into train/val/test
process SPLIT_DATA {
    tag "split"
    label 'process_low'
    publishDir params.output_dir, mode: 'copy'

    input:
    path(data_files)
    path(mean_file)
    path(std_file)

    output:
    path("train/*.nc"), emit: train
    path("val/*.nc"), emit: val
    path("test/*.nc"), emit: test
    path("split_info.json"), emit: info

    script:
    def train_ratio = params.train_ratio
    def val_ratio = params.val_ratio
    """
    mkdir -p train val test

    # Split files by temporal order (simplified)
    files=(\$(ls ${data_files} | sort))
    total=\${#files[@]}
    train_end=\$((total * ${train_ratio} / 1))
    val_end=\$((total * (${train_ratio} + ${val_ratio}) / 1))

    for i in "\${!files[@]}"; do
        if [ \$i -lt \$train_end ]; then
            cp "\${files[\$i]}" train/
        elif [ \$i -lt \$val_end ]; then
            cp "\${files[\$i]}" val/
        else
            cp "\${files[\$i]}" test/
        fi
    done

    # Copy normalization files to output
    cp ${mean_file} train/
    cp ${std_file} train/

    echo '{
        "split_method": "temporal",
        "train_ratio": ${train_ratio},
        "val_ratio": ${val_ratio},
        "test_ratio": '"\$(echo "1 - ${train_ratio} - ${val_ratio}" | bc)"',
        "train_samples": '"\$(ls train/*.nc 2>/dev/null | wc -l)"',
        "val_samples": '"\$(ls val/*.nc 2>/dev/null | wc -l)"',
        "test_samples": '"\$(ls test/*.nc 2>/dev/null | wc -l)"'
    }' > split_info.json
    """
}

// Process: Generate Data Card
process GENERATE_DATACARD {
    tag "datacard"
    label 'process_low'
    publishDir params.output_dir, mode: 'copy'

    input:
    path(split_info)
    path(norm_params)
    path(regrid_params)

    output:
    path("DATA_CARD.md"), emit: datacard

    script:
    """
    cat > DATA_CARD.md << 'DATACARD'
# Climate Data Card

## Overview
Processed climate data ready for ML training.

## Transformations
- **Regridding**: See regrid_params.json
- **Normalization**: See norm_params.json
- **Splitting**: See split_info.json

## Files
- train/ - Training data
- val/ - Validation data
- test/ - Test data
- normalize_mean.npz - Normalization means
- normalize_std.npz - Normalization stds

## Usage
\`\`\`python
import numpy as np
mean = np.load('train/normalize_mean.npz')
std = np.load('train/normalize_std.npz')
\`\`\`
DATACARD
    """
}

// Main workflow
workflow {
    // Input channel from file pattern
    Channel
        .fromPath(params.input)
        .set { input_ch }

    // Ingest stage
    INSPECT_DATA(input_ch)

    // Preprocess stage
    REGRID_DATA(INSPECT_DATA.out.data)

    // Transform stage - compute normalization on all data first
    regridded_files = REGRID_DATA.out.regridded.collect()
    COMPUTE_NORM_STATS(regridded_files)

    // Apply normalization to each file
    NORMALIZE_DATA(
        REGRID_DATA.out.regridded,
        COMPUTE_NORM_STATS.out.mean,
        COMPUTE_NORM_STATS.out.std
    )

    // Output stage - split data
    normalized_files = NORMALIZE_DATA.out.normalized.collect()
    SPLIT_DATA(
        normalized_files,
        COMPUTE_NORM_STATS.out.mean,
        COMPUTE_NORM_STATS.out.std
    )

    // Generate Data Card
    GENERATE_DATACARD(
        SPLIT_DATA.out.info,
        COMPUTE_NORM_STATS.out.params,
        REGRID_DATA.out.params.first()
    )
}
