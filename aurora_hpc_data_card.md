
# Data Card: ANL-ALCF Aurora HPC Job Accounting Dataset

## 1. Dataset Overview
- **Name:** ANL-ALCF-DJC-AURORA Job Accounting Data
- **Version:** 20250127_20251031
- **Size:** 469.13 MB
- **Records:** 663,506 jobs
- **Features:** 64 columns
- **Format:** CSV (Comma-Separated Values)
- **Machine:** Aurora supercomputer at Argonne Leadership Computing Facility (ALCF)

## 2. Provenance
- **Organization:** Argonne National Laboratory - Argonne Leadership Computing Facility (ANL-ALCF)
- **System:** Aurora (next-generation exascale supercomputer)
- **Collection Method:** Cobalt job scheduler accounting logs
- **Collection Period:** 2024-12-31 to 2025-10-31
- **Data Sources:** Job scheduler logs, resource manager
- **License:** Not specified (check with ALCF for usage rights)

## 3. Dataset Characteristics
- **Domain:** HPC / High-Performance Computing
- **Data Types:** 
  - Numerical: Resource usage (nodes, cores, GPUs), timing (queue, runtime, walltime)
  - Categorical: Queue names, allocation categories, exit statuses, machine modes
  - Temporal: Timestamps for queued, start, and end times
  - Boolean: Job type flags (subblock, consecutive, multilocation, etc.)
  - Identifiers: Job IDs, anonymized usernames, project names

- **Temporal Coverage:** 304 days
  - Start: 2024-12-31 02:22:48
  - End: 2025-10-31 23:47:57

- **Resource Coverage:**
  - Nodes: 0 to 10,400 (mean: 67.3)
  - Cores: 0 to variable (mean used: 67.3)
  - GPUs: Requested (GPUS_REQUESTED column present)
  - Walltime: 0 to 864,000 seconds (240 hours max)

- **User Base:**
  - Unique users: 826 (anonymized as USERNAME_GENID)
  - Unique projects: 222 (anonymized as PROJECT_NAME_GENID)
  - Allocation types: INCITE (131,591), Discretionary (502,380), ALCC (29,508)

## 4. Sensitive Attributes
- **Contains PII:** Partially
  - Original usernames and project names have been anonymized (USERNAME_GENID, PROJECT_NAME_GENID)
  - Job names and queue names may contain identifying information
  - No direct personal identifiers (email, phone, etc.)
  
- **Anonymization Applied:** Yes
  - Usernames converted to generic IDs
  - Project names converted to generic IDs
  - Recommended: Review JOB_NAME and QUEUE_NAME for any remaining PII

## 5. Transformations Applied
- Usernames and project names anonymized
- Derived features computed:
  - Core-hour calculations (REQUESTED_CORE_HOURS, USED_CORE_HOURS)
  - Capability usage breakdown
  - Job type classifications (IS_SINGLE, IS_CONSECUTIVE, etc.)
  - Size bucket categorizations (SIZE_BUCKETS3)
  - Python job detection (IS_PYTHON)

## 6. Annotations
- **Has Labels:** Yes (for supervised learning)
- **Label Types:** 
  - **Primary Target:** QUEUED_WAIT_SECONDS (continuous, regression)
  - **Alternative Targets:** 
    - RUNTIME_SECONDS (job duration prediction)
    - EXIT_STATUS (job success/failure classification)
    - IS_TASK_EXIT_CODES_NON_ZERO (binary failure classification)

- **Label Quality:**
  - QUEUED_WAIT_SECONDS: Complete (0% missing), range 0 to 6.9M seconds
  - Mean: 2,230 seconds (37 minutes)
  - Median: 111 seconds (2 minutes)
  - 13.89% outliers beyond Q3 + 1.5*IQR

## 7. Limitations
- **Data Quality Issues:**
  - 5.86% of cells contain missing values
  - Columns with 100% missing: MACHINE_PARTITION, PYTHON_EXECUTABLE_PATH, PYTHON_EXECUTABLE_VERSION
  - MODE column 48.39% missing
  - TASK_EXIT_CODES 16.36% missing
  - Some columns have mixed data types requiring cleaning

- **Skewed Distribution:**
  - Queue times are heavily right-skewed (mean >> median)
  - 13.89% outliers with very long queue times
  - May require log transformation or robust scaling

- **Class Imbalance:**
  - SIZE_BUCKETS3: 95.8% in smallest bucket (0-20%)
  - ALLOCATION_AWARD_CATEGORY: 75.7% Discretionary allocations

- **Temporal Bias:**
  - Data represents specific time period on Aurora system
  - Scheduler policies and workload patterns may change over time
  - May not generalize to other HPC systems or time periods

- **Use Restrictions:**
  - Research and analysis purposes only
  - Cannot de-anonymize users or projects
  - Check ALCF data use policies before publication

## 8. AI Readiness Assessment

### Current Readiness Level: 2-3 (CLEANED with TARGET DEFINED)

**Level 2 (CLEANED):**
- ✓ Data exists in structured format (CSV)
- ✓ Columns have defined data types
- ✗ 5.86% missing values need handling
- ✓ No duplicate records

**Level 3 (LABELED):**
- ✓ Target variable defined (QUEUED_WAIT_SECONDS)
- ✓ Target is complete with no missing values
- ✓ Clear prediction task defined

**Level 4 (FEATURE ENGINEERED):**
- ✗ Domain-specific features not yet extracted
- ✗ Features not normalized/standardized
- ✗ Categorical variables not encoded
- ✗ Temporal features not extracted from timestamps

**Level 5 (FULLY AI-READY):**
- ✗ Not sharded for distributed training
- ✗ Not in optimized format (currently CSV)
- ✗ Train/val/test splits not created

### Key Features for Queue Time Prediction

**Highly Correlated with Target (|r| > 0.1):**
1. ELIGIBLE_WAIT_SECONDS (r=0.747) - strong predictor
2. NODES_REQUESTED (r=0.210)
3. CORES_REQUESTED (r=0.210)
4. REQUESTED_CORE_HOURS (r=0.171)
5. RUNTIME_SECONDS (r=0.153)
6. WALLTIME_SECONDS (r=0.136)

**Categorical Predictors:**
- QUEUE_NAME (153 unique values)
- ALLOCATION_AWARD_CATEGORY (4 categories)
- SIZE_BUCKETS3 (job size categories)

**Temporal Predictors:**
- QUEUED_TIMESTAMP (time of day, day of week effects)
- Date-based features (holidays, weekends)

### Detected Domain: HPC (High-Performance Computing)

**Confidence:** High
- Job accounting columns (COBALT_JOBID, QUEUE_NAME)
- Resource metrics (NODES_USED, CORES_USED, WALLTIME_SECONDS)
- HPC-specific features (capability usage, allocation categories)
- Scheduler wait times (QUEUED_WAIT_SECONDS, ELIGIBLE_WAIT_SECONDS)

---

**Generated:** 2026-01-08 09:34:19
**Tool:** DRAI (Data Readiness for AI)
