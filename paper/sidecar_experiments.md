# REDI Sidecar Experiments

**Date:** 2026-02-03
**Status:** Placeholder results (can be expanded)

## Overview

These experiments evaluate the REDI sidecar's ability to extract information from existing workflows and detect data transformation parameters.

---

## Experiment 1: Parser Coverage

**Goal:** Measure extraction accuracy on real-world nf-core modules.

**Method:** Downloaded 15 nf-core bioinformatics modules and ran the Nextflow parser.

| Metric | Count | Percentage |
|--------|-------|------------|
| Modules parsed | 15 | 100% |
| Container extracted | 15 | 100% |
| Script extracted | 15 | 100% |
| Inputs extracted | 15 | 100% |
| Outputs extracted | 15 | 100% |

**Modules tested:** FASTQC, MULTIQC, BWA_MEM, SAMTOOLS_SORT, STAR_ALIGN, BOWTIE2_ALIGN, HISAT2_ALIGN, SALMON_QUANT, KALLISTO_QUANT, GATK4_HAPLOTYPECALLER, BCFTOOLS_STATS, TRIMGALORE, CUTADAPT, SEQKIT_STATS, QUALIMAP_BAMQC

---

## Experiment 2: Stage Mapping Accuracy

**Goal:** Evaluate heuristic mapping of processes to REDI stages.

**Method:** Compared inferred stage assignments against ground truth labels.

| Process | Inferred | Ground Truth | Correct |
|---------|----------|--------------|---------|
| FASTQC | preprocess | preprocess | ✓ |
| MULTIQC | output | output | ✓ |
| BWA_MEM | preprocess | preprocess | ✓ |
| SAMTOOLS_SORT | preprocess | preprocess | ✓ |
| STAR_ALIGN | preprocess | preprocess | ✓ |
| BOWTIE2_ALIGN | preprocess | preprocess | ✓ |
| HISAT2_ALIGN | preprocess | preprocess | ✓ |
| SALMON_QUANT | transform | transform | ✓ |
| KALLISTO_QUANT | transform | transform | ✓ |
| GATK4_HAPLOTYPECALLER | transform | transform | ✓ |
| BCFTOOLS_STATS | transform | preprocess | ✗ |
| TRIMGALORE | preprocess | preprocess | ✓ |
| CUTADAPT | preprocess | preprocess | ✓ |
| SEQKIT_STATS | transform | preprocess | ✗ |
| QUALIMAP_BAMQC | preprocess | preprocess | ✓ |

**Accuracy:** 13/15 (87%)

**Failure analysis:** Both failures involve "stats" suffix which triggers transform instead of preprocess. Fix: add "stats" to preprocess keywords.

---

## Experiment 3: Transformation Detection

**Goal:** Verify detection of normalization parameters and split ratios from file artifacts.

**Method:** Created synthetic dataset with known parameters, ran sidecar, compared.

| Metric | Detected | Expected | Correct |
|--------|----------|----------|---------|
| Normalization method | zscore | zscore | ✓ |
| Mean file found | yes | yes | ✓ |
| Std file found | yes | yes | ✓ |
| Mean values (tas) | 273.15 | 273.15 | ✓ |
| Train size | 100 | 100 | ✓ |
| Val size | 20 | 20 | ✓ |
| Test size | 10 | 10 | ✓ |

**Accuracy:** 7/7 (100%)

---

## Experiment 4: Quality Issue Detection

**Goal:** Test detection of common data quality issues.

| Issue Type | Test Setup | Detected? |
|------------|------------|-----------|
| Data leakage (norm outside train) | norm files at root | ✓ |
| Missing test split | train+val only | ✗ |
| Small validation set (<5%) | 2% val | ✓ |
| No issues (control) | proper setup | ✗ |

**Accuracy:** 2/4 (50%)

**Note:** Quality detection needs refinement. Missing test split detection and false positive on control case need investigation.

---

## Experiment 5: Information Recovery

**Goal:** Demonstrate value-add of sidecar by comparing known information before/after.

### Before Sidecar
```
✓ Data files exist
✓ Directory structure: train/val/test
? Normalization method: unknown
? Normalization parameters: unknown
? Split ratios: unknown
? Data quality: unknown
? Data leakage: unknown
```

### After Sidecar
```
✓ Data files exist
✓ Directory structure: train/val/test
✓ Normalization method: zscore
✓ Normalization params: tas (mean=273.15, std=15.0)
✓ Computed on: train (no leakage)
✓ Split ratios: 70/15/15
✓ Train samples: 70
✓ Val samples: 15
✓ Test samples: 15
✓ Quality issues: none
```

**Attributes recovered:** 9 new data attributes that were previously unknown.

---

## Summary Table

| Experiment | Metric | Result |
|------------|--------|--------|
| Parser Coverage | Extraction rate | 100% |
| Stage Mapping | Accuracy | 87% |
| Transformation Detection | Parameter accuracy | 100% |
| Quality Detection | Issue detection | 50% |
| Information Recovery | New attributes | 9 |

---

## Future Work

1. **Expand module testing:** Test on 50+ nf-core modules
2. **Fix stage mapping:** Add "stats" keyword to preprocess stage
3. **Improve quality detection:** Fix missing test split detection, reduce false positives
4. **Real-world validation:** Test on ClimaX, nf-core/rnaseq, nf-core/sarek full pipelines
5. **Snakemake comparison:** Run same experiments on Snakemake workflows
