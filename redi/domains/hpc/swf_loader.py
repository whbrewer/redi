"""
REDI SWF Ingester - Standard Workload Format loader for HPC scheduler traces.

SWF (Standard Workload Format, Parallel Workloads Archive) is a whitespace-
delimited job-trace format with a `;`-commented header block. This loader
targets the public Frontier traces (Maheshwari, Klusacek, Suter, Brewer,
JSSPP 2026) but follows the documented SWF field layout closely enough to
generalize to other PWA-style traces that use the same 0-20 field set plus
a trailing site-specific descriptor field.

Header block (as documented in frontier-2024.swf / frontier-2025.swf):
  0: job_id
  1: submit_time          (seconds relative to UnixStartTime)
  2: wait_time            (seconds)
  3: runtime              (seconds)
  4: allocated_nodes      (nodes OR cores -- ambiguous, see D7 below)
  5: CPU_time_used        (documented "not used, same as runtime" -- observed
                           to NOT equal runtime in practice; treat with caution)
  6: RAM (used, kB)
  7: required_nodes       (documented "same as allocated nodes/cores")
  8: walltime_limit       (seconds)
  9: RAM (requested, kB)
 10: job_status           (documented "always 1, not used" -- observed to
                           vary in {-2, -1, 0, 1} in practice; semantics
                           uncertain, left as raw values, NOT sentinel-cleaned)
 11: user_id
 12: group_id
 13: executable_number    (-1, not used)
 14: queue_id             (-1, not used)
 15: partition            (-1, not used)
 16: preceding_job        (-1, not used)
 17: think_time           (-1, not used)
 18: used_GPUs            (sum over all used nodes; must be divisible by the
                           number of used nodes -- this invariant is the key
                           evidence used to resolve D7, see below)
 19: soft_walltime        (-1, not used in observed data)
 20: required_properties  site-specific descriptor, e.g. "frontier:excl:2500x112"

D7 -- allocated/required node-count defect
-------------------------------------------
Fields 4 and 7 (`nodes_numeric_raw` / `required_nodes_numeric_raw`) are
documented as node counts but are frequently inconsistent with a ~9,408-node
machine (observed max: 589,824 in the 2024 file). The trailing descriptor
field `frontier:excl:<N>x112` is present on 100% of rows in both files with
a *constant* "x112" suffix.

Resolution (empirically verified against both real files, see
`if __name__ == "__main__"` report below):
  - `used_GPUs` (field 18) is documented to be "divisible by the number of
    used nodes" whenever it is populated (not -1). Checking this invariant
    against the two candidate node counts is a clean, independent test:
      * divisible by descriptor-derived N:            100.0% of populated rows
        (both files, ~80k / ~51k rows respectively)
      * divisible by the raw numeric field (4/7):       16-43% of populated rows
    This is decisive: the descriptor field's leading integer N is the
    reliable/canonical node count; the numeric column is NOT.
  - The raw numeric column agrees with N exactly on ~53-54% of rows (ratio
    numeric/N == 1). On the remaining rows, numeric/N clusters at rational
    values (56, 8, 64, 2, 7, 32, 48, ... -- i.e. values related to the "112"
    cores/node figure), consistent with the numeric column sometimes
    recording *cores* (with a variable, job-specific cores-per-node
    assumption) rather than nodes, exactly as the header's own hedge
    ("number of all allocated nodes/or number of all allocated CPU cores")
    admits. Multiplying N by the descriptor's "x112" suffix does NOT
    reconcile with the numeric column either (agrees on <0.2% of rows) --
    so "x112" is not a multiplier to apply to N, it's descriptive metadata
    (cores-per-node used by the (synthetic, per the file header) trace
    generator), constant across the whole file.
  - Conclusion: `nodes_required_canonical` = N parsed directly from the
    descriptor (NOT N * 112, NOT the raw numeric column). `nodes_mismatch`
    flags rows where the raw numeric column disagrees with N.

All -1 sentinel values (SWF convention for "not available") are converted to
NaN for fields where -1 is not itself a valid domain value, EXCEPT
`job_status`, whose real-world values (-2, -1, 0, 1) don't cleanly separate
into "valid code" vs. "missing" -- documented above, left raw, flagged for a
human data-quality decision (see report at bottom of this file).
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...pipeline import PipelineStep, PipelineStage, PipelineContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

# Canonical column names for SWF fields 0-20, in file order. Frontier's SWF
# files always carry all 21 fields (verified: 100% of 1,217,397 + 977,884
# data lines have exactly 21 whitespace-delimited tokens). If a future trace
# only has the "documented minimum" 18 fields (0-17, no GPU/soft-walltime/
# descriptor extension), parse_swf() falls back gracefully (see below).
SWF_COLUMNS_FULL = [
    "job_id",                       # 0
    "submit_time_rel",              # 1
    "wait_time",                    # 2
    "runtime",                      # 3
    "nodes_numeric_raw",            # 4
    "cpu_time_used",                # 5
    "ram_used_kb",                  # 6
    "required_nodes_numeric_raw",   # 7
    "walltime_limit",               # 8
    "ram_requested_kb",             # 9
    "job_status",                   # 10
    "user_id",                      # 11
    "group_id",                     # 12
    "executable_number",            # 13
    "queue_id",                     # 14
    "partition_id",                 # 15
    "preceding_job",                # 16
    "think_time",                   # 17
    "used_gpus",                    # 18
    "soft_walltime",                # 19
    "descriptor_raw",               # 20
]

SWF_COLUMNS_MINIMAL = SWF_COLUMNS_FULL[:18]

# Fields where -1 unambiguously means "not available" and should become NaN.
# job_id/submit_time_rel are excluded (0 is a legitimate value, not missing).
# job_status is excluded -- see module docstring; its -1/-2 values appear to
# be real (if undocumented) status codes, not a missing-data sentinel.
_SENTINEL_MINUS1_COLS = [
    "wait_time", "runtime", "nodes_numeric_raw", "cpu_time_used",
    "ram_used_kb", "required_nodes_numeric_raw", "walltime_limit",
    "ram_requested_kb", "user_id", "group_id", "executable_number",
    "queue_id", "partition_id", "preceding_job", "think_time",
    "used_gpus", "soft_walltime",
]

_DESCRIPTOR_RE = re.compile(r"^[^:]+:[^:]+:(\d+)x(\d+)$")


def parse_swf_header(path: Path) -> Dict[str, Any]:
    """Parse the `;`-commented header block of an SWF file.

    Returns a dict of whatever `; Key: Value` pairs are present, plus
    `unix_start_time` (int, required for absolute-timestamp conversion) and
    `declared_num_jobs` (int, from the "; number of jobs: N" line) when
    found. Raises ValueError if UnixStartTime is missing -- absolute
    timestamps can't be computed without it.
    """
    meta: Dict[str, Any] = {}
    kv_re = re.compile(r"^;\s*([A-Za-z][A-Za-z0-9 _]*?)\s*:\s*(.*?)\s*$")
    num_jobs_re = re.compile(r"^;\s*number of jobs:\s*(\d+)\s*$", re.IGNORECASE)

    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.startswith(";"):
                break
            m_jobs = num_jobs_re.match(line)
            if m_jobs:
                meta["declared_num_jobs"] = int(m_jobs.group(1))
                continue
            m = kv_re.match(line)
            if m:
                key, val = m.group(1).strip(), m.group(2).strip()
                meta[key] = val

    if "UnixStartTime" not in meta:
        raise ValueError(f"SWF header in {path} has no 'UnixStartTime' field; "
                          f"cannot compute absolute timestamps")
    meta["unix_start_time"] = int(meta["UnixStartTime"])
    return meta


def _peek_field_count(path: Path) -> int:
    """Return the whitespace-token count of the first non-comment, non-blank line."""
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            return len(line.split())
    raise ValueError(f"No data lines found in {path}")


def parse_swf(path: Path) -> pd.DataFrame:
    """Parse an SWF job trace file into a DataFrame.

    Pure function, independent of the pipeline machinery -- takes a path,
    returns a DataFrame. Vectorized throughout (single `pandas.read_csv`
    over the whitespace-delimited body plus vectorized string ops for the
    descriptor field; no per-row Python loop over ~1.2M lines).

    Columns produced:
      - All documented SWF fields (see SWF_COLUMNS_FULL), -1 sentinels
        converted to NaN per `_SENTINEL_MINUS1_COLS` (job_status is the one
        deliberate exception -- see module docstring).
      - `submit_time`, `start_time`, `end_time`: absolute Unix epoch seconds,
        derived from the file's own `UnixStartTime` header field.
      - `descriptor_raw`: the raw trailing site-specific string, kept as-is.
      - `cores_per_node_descriptor`: the "112" in "frontier:excl:2500x112",
        parsed out (nullable Int64; NaN if the descriptor doesn't match the
        expected `site:flags:NxM` shape).
      - `nodes_required_canonical`: the D7-resolved node count (see module
        docstring) -- descriptor-derived N when parseable, else falls back
        to `nodes_numeric_raw` with `descriptor_parseable=False`.
      - `nodes_mismatch`: bool, True where `nodes_numeric_raw` disagrees with
        `nodes_required_canonical` (NaN where the raw numeric field itself
        is missing).
      - `descriptor_parseable`: bool, whether the descriptor matched the
        expected `site:flags:NxM` pattern (should be True for every Frontier
        row; guards against silently mis-handling a differently-shaped trace).

    `df.attrs["header"]` carries the full parsed header dict (including
    `unix_start_time` and, if present, `declared_num_jobs`).
    """
    path = Path(path)
    header = parse_swf_header(path)
    unix_start_time = header["unix_start_time"]

    n_fields = _peek_field_count(path)
    if n_fields >= len(SWF_COLUMNS_FULL):
        names = SWF_COLUMNS_FULL
        has_descriptor = True
    else:
        # Older/minimal SWF trace without the GPU/soft-walltime/descriptor
        # extension fields. Fall back to the documented 0-17 minimum.
        names = SWF_COLUMNS_MINIMAL[:n_fields] if n_fields < len(SWF_COLUMNS_MINIMAL) else SWF_COLUMNS_MINIMAL
        has_descriptor = False
        logger.info(
            f"     {path.name}: only {n_fields} whitespace fields detected "
            f"(expected {len(SWF_COLUMNS_FULL)}); no site-specific descriptor "
            f"field -- D7 node-count cross-check will be skipped for this file"
        )

    try:
        df = pd.read_csv(
            path,
            comment=";",
            sep=r"\s+",
            header=None,
            names=names,
            engine="c",
        )
    except Exception:
        # Fall back to a tolerant line-by-line parse if the fast path chokes
        # on malformed/ragged lines, so we can report exactly what's wrong
        # instead of failing the whole file.
        logger.warning(f"     Fast CSV parse failed for {path}; falling back to per-line parsing")
        rows = []
        malformed = 0
        with open(path, "r", errors="replace") as f:
            for line in f:
                if line.startswith(";") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) != len(names):
                    malformed += 1
                    continue
                rows.append(parts)
        df = pd.DataFrame(rows, columns=names)
        for c in names:
            if c != "descriptor_raw":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if malformed:
            logger.warning(f"     {path.name}: {malformed:,} malformed data lines skipped")

    # -1 sentinel -> NaN, for the fields where -1 is unambiguously "missing"
    for col in _SENTINEL_MINUS1_COLS:
        if col in df.columns:
            df[col] = df[col].mask(df[col] == -1)

    # Absolute timestamps, derived from this file's own UnixStartTime
    df["submit_time"] = unix_start_time + df["submit_time_rel"]
    df["start_time"] = df["submit_time"] + df["wait_time"]
    df["end_time"] = df["start_time"] + df["runtime"]

    # D7: resolve canonical node count from the descriptor field
    if has_descriptor and "descriptor_raw" in df.columns:
        extracted = df["descriptor_raw"].astype(str).str.extract(_DESCRIPTOR_RE)
        descr_n = pd.to_numeric(extracted[0], errors="coerce")
        descr_cpn = pd.to_numeric(extracted[1], errors="coerce")

        df["descriptor_parseable"] = descr_n.notna()
        df["cores_per_node_descriptor"] = descr_cpn.astype("Int64")

        canonical = descr_n.where(descr_n.notna(), df["nodes_numeric_raw"])
        df["nodes_required_canonical"] = canonical.astype("Int64")

        mismatch = (df["nodes_numeric_raw"] != descr_n)
        # Only meaningful where both sides are known; NaN elsewhere
        mismatch = mismatch.where(df["nodes_numeric_raw"].notna() & descr_n.notna())
        df["nodes_mismatch"] = mismatch.astype("boolean")

        n_unparseable = int((~df["descriptor_parseable"]).sum())
        if n_unparseable:
            logger.warning(
                f"     {path.name}: {n_unparseable:,} rows had a descriptor field "
                f"that didn't match the expected 'site:flags:NxM' pattern; "
                f"nodes_required_canonical fell back to the raw numeric field "
                f"for those rows"
            )
    else:
        df["descriptor_parseable"] = False
        df["cores_per_node_descriptor"] = pd.array([pd.NA] * len(df), dtype="Int64")
        df["nodes_required_canonical"] = df["nodes_numeric_raw"].astype("Int64")
        df["nodes_mismatch"] = pd.array([pd.NA] * len(df), dtype="boolean")

    df.attrs["header"] = header
    return df


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

class LoadSWFData(PipelineStep):
    """Load Standard Workload Format (SWF) HPC scheduler traces.

    Detects applicability via `ctx.metadata["file_format"] == "swf"` (set by
    format detection elsewhere -- not this file's responsibility). Supports
    single- or multi-file ingestion (e.g. frontier-2024.swf + frontier-2025.swf
    loaded together); a `source_file` column tags provenance, and per-file
    UnixStartTime values are recorded in `ctx.metadata["swf_headers"]`.
    """

    name = "load_swf"
    stage = PipelineStage.INGEST
    description = "Load Standard Workload Format (SWF) HPC scheduler traces"

    def validate(self, ctx: PipelineContext) -> bool:
        return ctx.metadata.get("file_format") == "swf"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            paths = ctx.input_paths if ctx.input_paths else [ctx.input_path]

            dfs = []
            headers = {}
            for p in paths:
                p = Path(p)
                logger.debug(f"     Parsing SWF file: {p}")
                df = parse_swf(p)
                df["source_file"] = p.name
                headers[p.name] = df.attrs.get("header", {})
                dfs.append(df)

            if not dfs:
                ctx.add_error("No SWF input files provided")
                return ctx

            df = pd.concat(dfs, axis=0, ignore_index=True) if len(dfs) > 1 else dfs[0]

            if len(dfs) > 1:
                # job_id is only unique WITHIN a single SWF file -- each file's
                # IDs are independently sequential starting from 1 (verified:
                # frontier-2024.swf is 1..1,217,397, frontier-2025.swf is
                # 1..977,884, a full 977,884-row overlap). Left alone, the
                # generic RAPS-oriented CleanHPCForRAPS step downstream (which
                # dedups by job_id, keep="last") would silently splice
                # unrelated 2024/2025 jobs together under reused IDs rather
                # than just dropping rows -- a correctness bug, not merely an
                # undercount. Namespace job_id by source file so it's globally
                # unique before that step runs; the original per-file ID is
                # preserved in job_id_in_file.
                df["job_id_in_file"] = df["job_id"]
                df["job_id"] = df["source_file"].astype(str) + ":" + df["job_id"].astype(str)
                logger.info(
                    f"     {len(dfs)} SWF files loaded together; job_id "
                    f"namespaced by source file to avoid cross-file ID collisions"
                )

            ctx.data = df

            ctx.metadata["num_rows"] = len(df)
            ctx.metadata["num_columns"] = len(df.columns)
            ctx.metadata["columns"] = list(df.columns)
            ctx.metadata["swf_headers"] = headers

            unix_start_times = {name: h.get("unix_start_time") for name, h in headers.items()}
            if len(set(unix_start_times.values())) > 1:
                logger.info(
                    f"     Multiple UnixStartTime values across input files: "
                    f"{unix_start_times} -- absolute timestamps were computed "
                    f"per-file before concatenation, so this is expected, not an error"
                )

            if "nodes_mismatch" in df.columns:
                known = df["nodes_mismatch"].notna()
                if known.any():
                    mismatch_rate = df.loc[known, "nodes_mismatch"].mean()
                    ctx.metadata["d7_node_mismatch_rate"] = float(mismatch_rate)
                    logger.info(
                        f"     D7 check: {mismatch_rate:.1%} of rows have a raw numeric "
                        f"node/core field that disagrees with the descriptor-derived "
                        f"canonical node count (descriptor is treated as authoritative)"
                    )

            logger.info(
                f"Loaded SWF trace(s): {len(df):,} jobs from {len(paths)} file(s)"
            )

        except Exception as e:
            ctx.add_error(f"Error loading SWF data: {str(e)}")

        return ctx


# ---------------------------------------------------------------------------
# Standalone verification report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    import traceback

    FILES = {
        "2024": ("/opt/data/hpc/frontier/frontier-2024.swf", 1_217_397),
        "2025": ("/opt/data/hpc/frontier/frontier-2025.swf", 977_884),
    }

    SAMPLE_ROWS_2024 = {
        1: dict(submit_time_rel=0, wait_time=24698, runtime=43219, nodes_numeric_raw=1024,
                required_nodes_numeric_raw=1024, walltime_limit=43200, descriptor_raw="frontier:excl:1024x112"),
        2: dict(submit_time_rel=0, wait_time=267809, runtime=6207, nodes_numeric_raw=20000,
                required_nodes_numeric_raw=20000, walltime_limit=6300, descriptor_raw="frontier:excl:2500x112"),
        3: dict(submit_time_rel=0, wait_time=486768, runtime=43211, nodes_numeric_raw=2048,
                required_nodes_numeric_raw=2048, walltime_limit=43200, descriptor_raw="frontier:excl:2048x112"),
        4: dict(submit_time_rel=0, wait_time=486887, runtime=41818, nodes_numeric_raw=2050,
                required_nodes_numeric_raw=2050, walltime_limit=43200, descriptor_raw="frontier:excl:2050x112"),
        5: dict(submit_time_rel=0, wait_time=200168, runtime=483, nodes_numeric_raw=32768,
                required_nodes_numeric_raw=32768, walltime_limit=21600, descriptor_raw="frontier:excl:1024x112"),
    }

    print("=" * 78)
    print("REDI SWF Ingester -- Verification Report")
    print("=" * 78)

    parsed = {}
    for label, (path_str, expected_rows) in FILES.items():
        path = Path(path_str)
        print(f"\n--- {label}: {path} ---")
        try:
            df = parse_swf(path)
            parsed[label] = df
            header = df.attrs["header"]
            n = len(df)
            ok = "OK" if n == expected_rows else "MISMATCH"
            print(f"  Rows parsed: {n:,}  (expected {expected_rows:,})  [{ok}]")
            print(f"  UnixStartTime: {header['unix_start_time']} "
                  f"({datetime.datetime.fromtimestamp(header['unix_start_time'], datetime.timezone.utc)})")
            if "declared_num_jobs" in header:
                decl = header["declared_num_jobs"]
                ok2 = "OK" if decl == n else "MISMATCH"
                print(f"  Header-declared 'number of jobs': {decl:,}  [{ok2}]")
            other_header_keys = {k: v for k, v in header.items()
                                  if k not in ("unix_start_time", "UnixStartTime", "declared_num_jobs")}
            if other_header_keys:
                print(f"  Other header fields: {other_header_keys}")
        except Exception:
            print(f"  UNHANDLED EXCEPTION while parsing {path}:")
            print(traceback.format_exc())

    # --- Spot-check sample rows 1-5 against frontier-2024.swf ---
    print("\n--- Spot-check: rows 1-5 of frontier-2024.swf against parsed output ---")
    if "2024" in parsed:
        df24 = parsed["2024"]
        by_id = df24.set_index("job_id")
        all_ok = True
        for jid, expected in SAMPLE_ROWS_2024.items():
            if jid not in by_id.index:
                print(f"  job_id={jid}: NOT FOUND in parsed output")
                all_ok = False
                continue
            row = by_id.loc[jid]
            mismatches = []
            for k, v in expected.items():
                actual = row[k]
                if k == "descriptor_raw":
                    match = (str(actual) == v)
                else:
                    match = (not pd.isna(actual)) and (int(actual) == v)
                if not match:
                    mismatches.append((k, v, actual))
            status = "OK" if not mismatches else f"MISMATCH: {mismatches}"
            if mismatches:
                all_ok = False
            print(f"  job_id={jid}: {status}")
        print(f"  Overall spot-check: {'PASSED' if all_ok else 'FAILED'}")
    else:
        print("  Skipped -- 2024 file failed to parse")

    # --- D7 mismatch rate (2024 file) ---
    print("\n--- D7: node-count mismatch rate (frontier-2024.swf) ---")
    if "2024" in parsed:
        df24 = parsed["2024"]
        known = df24["nodes_mismatch"].notna()
        n_known = int(known.sum())
        n_mismatch = int(df24.loc[known, "nodes_mismatch"].sum())
        rate = n_mismatch / n_known if n_known else float("nan")
        print(f"  Rows with a determinable mismatch flag: {n_known:,} / {len(df24):,}")
        print(f"  Mismatching rows (raw numeric field != descriptor-derived N): "
              f"{n_mismatch:,} ({rate:.1%})")
        print(f"  Raw numeric field, max observed value: {df24['nodes_numeric_raw'].max():,.0f} "
              f"(machine has ~9,408 nodes -- confirms field 4/7 is not reliably a node count)")
        print(f"  Descriptor-derived N, max observed value: "
              f"{df24['nodes_required_canonical'].max():,} (plausible for a 9,408-node machine, "
              f"with a small number of jobs exceeding it -- see anomaly note below)")

        # used_GPUs divisibility check -- the evidence used to pick the descriptor as canonical
        gpu_known = df24["used_gpus"].notna()
        if gpu_known.any():
            div_by_descr = (df24.loc[gpu_known, "used_gpus"] % df24.loc[gpu_known, "nodes_required_canonical"] == 0)
            div_by_numeric = (df24.loc[gpu_known, "used_gpus"] % df24.loc[gpu_known, "nodes_numeric_raw"] == 0)
            print(f"  Cross-check ('used_GPUs must be divisible by node count', n={int(gpu_known.sum()):,}):")
            print(f"    divisible by descriptor-derived N:  {div_by_descr.mean():.1%}")
            print(f"    divisible by raw numeric field:     {div_by_numeric.mean():.1%}")

        print("  Example mismatches (numeric field vs. descriptor):")
        examples = df24.loc[df24["nodes_mismatch"] == True,
                             ["job_id", "nodes_numeric_raw", "descriptor_raw", "nodes_required_canonical"]].head(6)
        for _, r in examples.iterrows():
            print(f"    job_id={r['job_id']}: numeric={r['nodes_numeric_raw']:.0f}  "
                  f"descriptor='{r['descriptor_raw']}'  canonical={r['nodes_required_canonical']}")
    else:
        print("  Skipped -- 2024 file failed to parse")

    # --- Private-day soft sanity check (2024-01-18 UTC) ---
    print("\n--- Soft sanity check: 2024-01-18 UTC window (private Frontier day has 995 jobs) ---")
    if "2024" in parsed:
        df24 = parsed["2024"]
        unix_start = df24.attrs["header"]["unix_start_time"]
        win_start = unix_start + 17 * 86400
        win_end = unix_start + 18 * 86400
        print(f"  Window: [{datetime.datetime.fromtimestamp(win_start, datetime.timezone.utc)}, "
              f"{datetime.datetime.fromtimestamp(win_end, datetime.timezone.utc)})")

        submit_in_window = df24[(df24["submit_time"] >= win_start) & (df24["submit_time"] < win_end)]
        print(f"  Jobs SUBMITTED in window:                {len(submit_in_window):,}")

        overlap = df24[(df24["start_time"] < win_end) & (df24["end_time"] > win_start)]
        print(f"  Jobs whose [start,end) OVERLAPS window:  {len(overlap):,}")
        print(f"  Private (unshareable) Slurm ground truth: 995 jobs")
        ratio_submit = len(submit_in_window) / 995
        ratio_overlap = len(overlap) / 995
        print(f"  Ratio to private count: submit={ratio_submit:.2f}x, overlap={ratio_overlap:.2f}x")
        if 0.5 <= ratio_submit <= 2.0 and 0.5 <= ratio_overlap <= 2.0:
            print("  -> Plausible ballpark (same order of magnitude); exact match not expected "
                  "since this public SWF trace may filter/aggregate differently than raw Slurm.")
        else:
            print("  -> WILDLY OFF from the private count -- would itself be a useful/concerning finding.")
    else:
        print("  Skipped -- 2024 file failed to parse")

    # --- Other parsing ambiguities / surprises ---
    print("\n--- Other parsing ambiguities / surprises ---")
    for label, df in parsed.items():
        print(f"  [{label}]")
        print(f"    job_status observed values: {sorted(df['job_status'].dropna().unique().tolist())} "
              f"(header claims 'always 1, not used' -- this is FALSE in the data; left un-sentineled, "
              f"semantics uncertain, flagged for human review)")
        cpu_eq_runtime = (df["cpu_time_used"] == df["runtime"]).mean()
        print(f"    cpu_time_used == runtime fraction: {cpu_eq_runtime:.1%} "
              f"(header claims 'not used, same as runtime' -- also appears FALSE)")
        cpn_vals = df["cores_per_node_descriptor"].dropna().unique().tolist()
        print(f"    descriptor cores-per-node ('x112' suffix) distinct values: {cpn_vals}")
        big = df[df["nodes_required_canonical"] > 9408]
        print(f"    Rows with descriptor-derived node count > 9,408 (machine size): {len(big):,}")
        n_unparseable = int((~df["descriptor_parseable"]).sum()) if "descriptor_parseable" in df else None
        print(f"    Rows with an unparseable descriptor field: {n_unparseable:,}")
        print(f"    submit_time_rel monotonic non-decreasing: {df['submit_time_rel'].is_monotonic_increasing}")
        print(f"    Trace span: {df['submit_time_rel'].max() / 86400:.1f} days")

    print("\n" + "=" * 78)
    print("End of report")
    print("=" * 78)
