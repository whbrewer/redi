 Plan: Auto-extract RAPS Job Objects from HPC Telemetry Data

 Context

 RAPS (in ExaDigiT) has ~10 hand-coded dataloaders that each manually map system-specific HPC telemetry columns to a standard
 Job object. Every HPC system uses different column names, time formats, trace data formats, and node representations.
 Currently, adding a new system requires writing a new dataloader from scratch.

 REDI's HPC domain handler (redi/domains/hpc.py) currently treats HPC data as generic tabular data for ML preparation. It
 knows nothing about the RAPS Job model.

 Goal: Add a system to REDI's HPC domain that can take any HPC telemetry DataFrame and automatically map its columns to RAPS
 Job fields, derive missing fields, normalize traces, and output a list of RAPS-compatible Job dictionaries — without needing
 a per-system custom dataloader.

 Architecture

 The new functionality lives in a single new module redi/domains/hpc_job_mapper.py and integrates into the existing HPC
 pipeline as an alternative output path activated by a --raps-jobs CLI flag.

 Pipeline flow when --raps-jobs is active:
 INGEST:      DetectFormat → LoadHPCData (existing)
                 ↓
              AutoMapColumns (NEW - maps arbitrary columns → canonical Job fields)
                 ↓
 PREPROCESS:  CleanHPCData → ParseTimestamps (existing)
                 ↓
 TRANSFORM:   DeriveJobFields (NEW - compute missing fields from available data)
                 ↓
 STRUCTURE:   BuildJobDicts (NEW - assemble RAPS job_dict per row, normalize traces)
                 ↓
 OUTPUT:      SaveJobDicts (NEW - serialize as JSON + optional pickle)

 When --raps-jobs is not active, the existing ML-prep pipeline is unchanged.

 New File: redi/domains/hpc_job_mapper.py

 1. Field Registry

 A list of FieldSpec dataclasses, one per RAPS Job field, each containing:
 - canonical_name — the job_dict key (e.g., "nodes_required")
 - aliases — every known column name across the 10 RAPS dataloaders (e.g., ["node_count", "num_nodes", "num_nodes_alloc",
 "nodes_alloc", "nnumr"])
 - field_type — "int", "str", "timestamp", "trace", "state", "list"
 - required — whether job_dict() requires it
 - default — fallback value if missing and not required

 Full alias inventory compiled from all RAPS dataloaders:
 Canonical Field: id
 Aliases (from 10 dataloaders): job_id, id_job, jobid, jid, primary_job_id, job_ID, allocation_id
 ────────────────────────────────────────
 Canonical Field: name
 Aliases (from 10 dataloaders): name, job_name, jobname, jnam, name_job
 ────────────────────────────────────────
 Canonical Field: account
 Aliases (from 10 dataloaders): account, user, user_id, id_user, usr, hashed_user_id, user_name
 ────────────────────────────────────────
 Canonical Field: nodes_required
 Aliases (from 10 dataloaders): nodes_required, node_count, num_nodes, num_nodes_alloc, nodes_alloc, nnumr, nodect
 ────────────────────────────────────────
 Canonical Field: submit_time
 Aliases (from 10 dataloaders): submit_time, time_submit, time_submission, adt, submitted_time, qtime, job_submit_time,
   queued_timestamp
 ────────────────────────────────────────
 Canonical Field: start_time
 Aliases (from 10 dataloaders): start_time, time_start, sdt, begin_time, start_timestamp, start
 ────────────────────────────────────────
 Canonical Field: end_time
 Aliases (from 10 dataloaders): end_time, time_end, edt, end_timestamp, end
 ────────────────────────────────────────
 Canonical Field: time_limit
 Aliases (from 10 dataloaders): time_limit, timelimit, walltime, elpl
 ────────────────────────────────────────
 Canonical Field: expected_run_time
 Aliases (from 10 dataloaders): run_time, runtime, runtime_seconds, wall_time, duration, elapsed
 ────────────────────────────────────────
 Canonical Field: cpu_trace
 Aliases (from 10 dataloaders): cpu_trace, cpu_utilisation, cpu_util, cpu_usage, CPU_usage_rate, cpu_power_consumption,
   sum_cpu0_power, perf1, avgpcon
 ────────────────────────────────────────
 Canonical Field: gpu_trace
 Aliases (from 10 dataloaders): gpu_trace, gpu_util, gpu_usage, gpu_power, sum_gpu_power, gpu_power_consumption, gpu_energy
 ────────────────────────────────────────
 Canonical Field: ntx_trace
 Aliases (from 10 dataloaders): ntx_trace, ib_tx, net_tx, ipogif0_tx_bytes
 ────────────────────────────────────────
 Canonical Field: nrx_trace
 Aliases (from 10 dataloaders): nrx_trace, ib_rx, net_rx, ipogif0_rx_bytes
 ────────────────────────────────────────
 Canonical Field: end_state
 Aliases (from 10 dataloaders): end_state, job_state, state_current, state_end, exit_status, status, exit_code
 ────────────────────────────────────────
 Canonical Field: priority
 Aliases (from 10 dataloaders): priority, pri, scheduling_class
 ────────────────────────────────────────
 Canonical Field: partition
 Aliases (from 10 dataloaders): partition, queue, qos
 ────────────────────────────────────────
 Canonical Field: scheduled_nodes
 Aliases (from 10 dataloaders): scheduled_nodes, nodes, xnames, exec_host, nodelist, machineId
 ────────────────────────────────────────
 Canonical Field: cpu_cores_required
 Aliases (from 10 dataloaders): cpu_cores_required, cpus, num_cpus, cores, ppn
 ────────────────────────────────────────
 Canonical Field: gpu_units_required
 Aliases (from 10 dataloaders): gpu_units_required, gpus, num_gpus, gres_gpu
 2. ColumnMapper

 Multi-phase column matching algorithm:

 1. Manual overrides (from config) — confidence 1.0
 2. Exact match (case-insensitive against alias list) — confidence 0.95
 3. Token match (split on _-. , compute token overlap with aliases) — confidence 0.5-0.8
 4. Fuzzy match (difflib.SequenceMatcher ratio against aliases) — confidence 0.6+

 Returns a ColumnMapping object with mapped: Dict[canonical, column], confidence scores, and lists of unmapped
 columns/required fields.

 3. FieldDeriver

 Computes missing canonical fields from available ones using derivation rules:
 - expected_run_time ← end_time - start_time
 - name ← f"job_{id}" (if only ID is available)
 - account ← "unknown" (placeholder)
 - nodes_required ← len(scheduled_nodes) (if node list exists but count doesn't)
 - submit_time ← start_time (fallback if no submit time)

 4. TraceNormalizer

 Detects trace data format and converts to RAPS-compatible utilization (0-1 scale):
 ┌─────────────────┬────────────────────────────────────────────────────────┬────────────────────────────────────────────────┐
 │     Format      │                       Detection                        │                   Conversion                   │
 ├─────────────────┼────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ Power (watts)   │ Column name contains "power"/"watt"/"energy"/"pcon",   │ (val - idle) / (max - idle), needs             │
 │                 │ OR values > 1000                                       │ power_config or heuristic min/max              │
 ├─────────────────┼────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ Percentage      │ Max value > 10 but ≤ 100                               │ val / 100                                      │
 │ (0-100)         │                                                        │                                                │
 ├─────────────────┼────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ Utilization     │ Max value ≤ 1.0                                        │ Pass through                                   │
 │ (0-1)           │                                                        │                                                │
 ├─────────────────┼────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ Scalar          │ Single number, not a list                              │ Keep as scalar (RAPS supports this)            │
 ├─────────────────┼────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ Missing         │ None or empty                                          │ Default to 0                                   │
 └─────────────────┴────────────────────────────────────────────────────────┴────────────────────────────────────────────────┘
 5. JobDictBuilder

 Iterates DataFrame rows and constructs job_dict-compatible dictionaries:
 - Normalizes all times relative to a reference time (min submit_time)
 - Handles timestamp types (pandas datetime → epoch → relative seconds)
 - Handles duration strings (HH:MM:SS, D-HH:MM:SS → seconds)
 - Parses node list strings (e.g., "node001,node002", "26742/0-31+26743/0-31")
 - Applies trace normalization per row
 - Skips invalid rows (e.g., nodes_required=0, start > end) with warnings

 6. RAPS Compatibility

 def get_job_class():
     try:
         from raps.job import Job, job_dict
         return Job, job_dict
     except ImportError:
         return None, _local_job_dict  # local fallback with same dict structure

 Modified Files

 redi/domains/hpc.py

 Add a get_raps_pipeline() function that returns the RAPS-job extraction steps, and modify get_full_pipeline() to accept a
 raps_jobs flag:

 def get_raps_pipeline(column_overrides=None, power_config=None):
     """Get the RAPS Job extraction pipeline."""
     from .hpc_job_mapper import AutoMapColumns, DeriveJobFields, BuildJobDicts, SaveJobDicts
     from ..pipeline.ingest import DetectFormat
     return [
         DetectFormat(), LoadHPCData(),           # existing ingest
         AutoMapColumns(overrides=column_overrides),  # NEW
         CleanHPCData(), ParseTimestamps(),       # existing preprocess
         DeriveJobFields(),                       # NEW
         BuildJobDicts(power_config=power_config),# NEW
         SaveJobDicts(),                          # NEW
     ]

 def get_full_pipeline(raps_jobs=False, **kwargs):
     if raps_jobs:
         return get_raps_pipeline(**kwargs)
     return (get_ingest_steps() + get_preprocess_steps() +
             get_transform_steps() + get_structure_steps() + get_output_steps())

 redi/cli.py

 - Add --raps-jobs flag to the run subparser
 - In the HPC domain branch of run_pipeline(), pass raps_jobs=True to get_full_pipeline() when the flag is set
 - Display mapping summary in the rich output

 Configuration

 Configurable via CLI flag or YAML config:

 raps_jobs:
   enabled: true
   column_overrides:       # manual: column_name → canonical_field
     my_weird_col: id
   power_config:           # for power→utilization conversion
     cpu: { idle: 100, max: 500 }
     gpu: { idle: 50, max: 400 }
   trace_quanta: 60        # seconds per trace sample

 Testing

 New: tests/test_hpc_job_mapper.py

 - TestColumnMapper: Parametrized tests with column schemas from all 10 systems (Frontier, Fugaku, MIT Supercloud, Marconi100,
  Lassen, Kestrel, Blue Waters, Philly, Google Cloud v2, Adastra MI250)
 - TestFieldDeriver: Derive run_time from start/end, name from id, etc.
 - TestTraceNormalizer: Detect power vs percentage vs utilization; normalize each
 - TestJobDictBuilder: Build jobs from sample DataFrames, verify all fields present

 Modified: tests/test_smoke_hpc.py

 - Add smoke test running the full RAPS-job pipeline end-to-end on sample HPC CSV data
 - Verify output JSON contains valid job dicts with required fields

 Verification

 source /opt/venvs/drai/bin/activate
 pytest tests/test_hpc_job_mapper.py -v
 pytest tests/test_smoke_hpc.py -v

 Files Summary
 File: redi/domains/hpc_job_mapper.py
 Action: CREATE
 Purpose: FieldSpec, ColumnMapper, FieldDeriver, TraceNormalizer, JobDictBuilder, pipeline steps
 ────────────────────────────────────────
 File: redi/domains/hpc.py
 Action: MODIFY
 Purpose: Add get_raps_pipeline(), update get_full_pipeline() signature
 ────────────────────────────────────────
 File: redi/cli.py
 Action: MODIFY
 Purpose: Add --raps-jobs flag, route to RAPS pipeline for HPC domain
 ────────────────────────────────────────
 File: tests/test_hpc_job_mapper.py
 Action: CREATE
 Purpose: Unit tests for mapper components
 ────────────────────────────────────────
 File: tests/test_smoke_hpc.py
 Action: MODIFY
 Purpose: Add RAPS job extraction smoke test



-------------------------



  Files Created

  redi/domains/hpc_job_mapper.py — New module with all RAPS Job extraction logic:
  - FieldSpec registry — 19 canonical RAPS Job fields with aliases from all 10 RAPS dataloaders
  - ColumnMapper — 4-phase column matching (manual overrides → exact → token → fuzzy)
  - FieldDeriver — Computes missing fields (run_time from timestamps, name from id, nodes from node list, submit_time fallback)
  - TraceNormalizer — Detects power/percentage/utilization format and normalizes to 0-1 scale
  - JobDictBuilder — Builds RAPS-compatible job dicts per row with timestamp normalization, duration parsing, state mapping,
  node list parsing
  - Pipeline steps — AutoMapColumns, DeriveJobFields, BuildJobDicts, SaveJobDicts

  tests/test_hpc_job_mapper.py — 60 unit tests covering:
  - Parametrized ColumnMapper tests against all 10 HPC system schemas (Frontier, Fugaku, MIT Supercloud, Marconi100, Lassen,
  Kestrel, Blue Waters, Philly, Google Cloud v2, Adastra MI250)
  - FieldDeriver, TraceNormalizer, JobDictBuilder tests
  - Helper function tests (_parse_duration, _normalize_state, _parse_node_list, etc.)

  Files Modified

  redi/domains/hpc.py — Added get_raps_pipeline() and updated get_full_pipeline() to accept raps_jobs=True with kwargs
  passthrough

  redi/cli.py — Added --raps-jobs CLI flag, raps_jobs parameter to run_pipeline(), HPC domain routing to RAPS pipeline, YAML
  config support for raps_jobs section

  tests/test_smoke_hpc.py — Added TestHPCRapsJobs class with 2 end-to-end smoke tests

  Usage

  # CLI
  redi run hpc_data.csv --raps-jobs -o ./output

  # YAML config
  raps_jobs:
    enabled: true
    column_overrides:
      my_weird_col: id
    power_config:
      cpu: { idle: 100, max: 500 }

  # Python API
  ctx = run_pipeline(input_path="data.csv", domain="hpc", raps_jobs=True)


---------------------------------


❯ can you create yaml files in examples/hpc for each of the systems listed in /opt/data


● All 69 tests pass. Here's what was created:

  examples/hpc/ — 11 YAML configs
  ┌─────────────────────┬───────────────────────┬────────────────────────┬──────────────────┬──────────────┐
  │        File         │        System         │      Input Format      │ Column Overrides │ Power Config │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ adastra_mi250.yaml  │ Adastra MI250 (CINES) │ Parquet                │ 6                │ CPU + GPU    │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ aurora.yaml         │ Aurora (ANL-ALCF)     │ CSV                    │ 14               │ -            │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ bluewaters.yaml     │ Blue Waters (NCSA)    │ Pre-processed CSV*     │ 10               │ -            │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ frontier.yaml       │ Frontier (ORNL)       │ Parquet (partitioned)  │ 13               │ -            │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ fugaku.yaml         │ Fugaku (RIKEN)        │ Parquet                │ 14               │ CPU          │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ gcloud_v2.yaml      │ Google Cluster v2     │ Pre-processed CSV*     │ 5                │ -            │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ gcloud_v3.yaml      │ Google Cluster v3     │ Pre-processed Parquet* │ 6                │ -            │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ lassen.yaml         │ Lassen (LLNL)         │ CSV                    │ 12               │ -            │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ marconi100.yaml     │ Marconi100 (CINECA)   │ Parquet                │ 15               │ CPU + GPU    │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ mit_supercloud.yaml │ MIT Supercloud        │ CSV                    │ 13               │ -            │
  ├─────────────────────┼───────────────────────┼────────────────────────┼──────────────────┼──────────────┤
  │ philly.yaml         │ Philly (Microsoft)    │ JSON                   │ 5                │ -            │
  └─────────────────────┴───────────────────────┴────────────────────────┴──────────────────┴──────────────┘
  *Systems with non-standard formats (PBS logs, headerless CSVs, nested JSON) include notes on the pre-processing needed before
   running.

  Each config documents the data source, record counts, column details, and includes column_overrides mapping the system's
  actual column names to RAPS canonical fields. Systems with power telemetry (Adastra, Fugaku, Marconi100) also include
  power_config for watts-to-utilization conversion.

  I also fixed a bug where raps_jobs.enabled: true in the YAML config wasn't being respected when the --raps-jobs CLI flag
  wasn't passed.



