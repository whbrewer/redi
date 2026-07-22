"""
REDI power-data-quality detectors for HPC job accounting (QC1 / QC2 / QC3).

This module implements W4 from notes/uc-demonstration-plan2.md §3 ("Automated
power-data QC: detect + repair"): the *detection* half, restricted to the
checks that can be built before the pm_counters ingester (W3) lands.

  - QC1 — dimensionless envelope check (`qc1_envelope_check`): Gerard Gil's
    (CINES) rule formalized via the plan's Phase-0 non-dimensionalization.
    Per job, the per-node average power implied by Slurm's ConsumedEnergyRaw
    must lie within the partition's physical [idle, max] node-power envelope:

        ū = (E / (Δt · N) − P_idle) / (P_max − P_idle)   must lie in
        [0 − ε, 1 + ε]

    Pure function, vectorized pandas, runs on the full 249,545-job Adastra
    sacct dump in well under a second.

  - QC2 — cross-source energy consistency (`qc2_cross_source`): Slurm
    ConsumedEnergyRaw vs. energy integrated from an independent per-node
    power/energy trace over the job's nodes × [start, end) window. The trace
    side is abstracted behind the `NodePowerSource` protocol because the real
    pm_counters ingester (W3) is blocked on CINES schema confirmation (D8);
    a `SyntheticPowerSource` makes QC2 fully testable today.

  - QC3 — trace-level stream checks (`qc3_check_stream`): pure function over
    generic per-device streams (t, optional cumulative-energy counter,
    optional instantaneous power) — timestamp monotonicity, sampling gaps,
    counter resets/rollbacks (with a taxonomy of negative-delta classes),
    energy/power cross-consistency, per-sample power envelope, stuck values.
    Deliberately contains NO pm_counters-specific parsing: the raw-log column
    schema is UNCONFIRMED (D8, awaiting CINES); the eventual W3 ingester maps
    columns onto (t, energy_cum, power_w) and this function only measures
    stream behavior.

Entry points:
  - `qc1_envelope_check(df, envelopes, epsilon)` — pure function.
  - `qc2_cross_source(df, source, rel_tol)` — pure function.
  - `qc3_check_stream(t, energy_cum, power_w, device_p_max, ...)` — pure
    function returning a `Qc3StreamReport`.
  - `PowerEnvelopeQC` — a PREPROCESS-stage `PipelineStep` wrapping QC1.
    NOT registered in any get_*_steps() list — wiring is a separate change,
    same policy as `LoadAdastraSacctDump` in sacct_loader.py.
  - `python -m redi.domains.hpc.power_qc` — synthetic known-answer tests
    (exit nonzero on any failure) followed by a QC1 verification report on
    the real Adastra sacct dump.

Input contract: the DataFrame produced by
`redi.domains.hpc.sacct_loader.parse_sacct_dump` (columns: partition,
energy_joules [Float64, may exceed 2^63 on u64 counter wraparound],
elapsed_seconds [Int64], nodes_required [Int64], node_list [list of node-name
strings], start_time / end_time [epoch seconds, Float64], ...).
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Unsigned-64-bit energy counters that wrapped negative are exported by Slurm
# as values just below 2^64 (e.g. 18446744061398995278). Anything above 2^63
# (~9.2e18 J — physically absurd: > 1 GW sustained for 292 years) is treated
# as a wraparound sentinel, matching the loader's Float64 rationale.
U64_WRAP_THRESHOLD = float(2**63)

# Ordered verdict taxonomy for QC1 (precedence order; first match wins).
QC1_VERDICTS = [
    "wraparound",         # energy_joules > 2^63 (u64 counter wrapped negative)
    "missing_energy",     # energy_joules is null (blank ConsumedEnergyRaw)
    "zero_energy",        # energy_joules is exactly 0 (e.g. NODE_FAIL jobs)
    "zero_duration",      # elapsed_seconds null or <= 0
    "missing_envelope",   # partition has no configured envelope (e.g. genoa
                          # pending CINES confirmation)
    "fail_high",          # ū > 1 + ε: impossible on any partition
    "fail_low",           # ū < 0 − ε on an *exclusive* partition: below idle
    "below_idle_shared",  # ū < 0 − ε on a *shared* partition: informational,
                          # NOT a failure (job owns only a slice of the node)
    "not_computable",     # defensive catch-all: ū is NaN after all the above
                          # (e.g. nodes_required null/<=0). Not in the plan's
                          # taxonomy; expected count 0 on the real dump.
    "pass",
]

QC2_VERDICTS = [
    "missing_trace",        # trace source has no/insufficient data for >=1 node
    "missing_energy",       # sacct-side energy_joules is null
    "agree",                # relative error <= rel_tol
    "disagree_sacct_low",   # sacct energy < trace energy beyond tolerance
    "disagree_sacct_high",  # sacct energy > trace energy beyond tolerance
]


# ---------------------------------------------------------------------------
# 1. Envelope config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PartitionEnvelope:
    """Per-NODE physical power envelope for one Slurm partition.

    Attributes:
        idle_w: idle power of one full node, watts.
        max_w: maximum power of one full node, watts.
        shared: True for shared-node partitions (`*-shared` on Adastra),
            where a job owns only a slice of each node, so a per-node-share
            average power *below* idle is legitimate. Above-max remains
            impossible regardless.
        source: free-text provenance of the numbers (who/what/when), so QC
            output is auditable.
    """

    idle_w: float
    max_w: float
    shared: bool
    source: str


# ---------------------------------------------------------------------------
# PROVISIONAL Adastra MI250 envelopes, derived from RAPS per-device constants
# in /home/w1b/exadigit/raps/config/adastraMI250.yaml:
#
#   system:  gpus_per_node: 8   (RAPS counts MI250X GCDs: 4 cards x 2 GCDs —
#                                see defect D3 in the plan)
#            cpus_per_node: 1
#            nics_per_node: 4
#   power:   power_gpu_idle: 44    power_gpu_max: 238     (per GCD, W)
#            power_cpu_idle: 90    power_cpu_max: 280     (per CPU, W)
#            power_mem: 37.13      (per node, constant — RAPS has no
#                                   idle/max split for memory)
#            power_nic: 20         (per NIC, constant)
#            power_nvme: 0         (per node, constant)
#
#   idle = 8*44 + 1*90 + 37.13 + 4*20 + 0
#        = 352  + 90   + 37.13 + 80   + 0  =  559.13 W/node
#   max  = 8*238 + 1*280 + 37.13 + 4*20 + 0
#        = 1904  + 280   + 37.13 + 80   + 0  = 2301.13 W/node
#
# Deliberately EXCLUDED from the node envelope: power_switch (Slingshot
# switches), power_cdu (cooling), and sivoc/rectifier conversion losses —
# those are facility/blade-infrastructure terms in RAPS, not part of what a
# node-level Slurm energy counter measures.
#
# genoa / genoa-shared are deliberately ABSENT: RAPS has no Adastra-Genoa
# config locally, and per the plan the genoa idle/max envelopes have been
# requested from CINES (D8 request pending). Jobs on those partitions get
# verdict "missing_envelope" rather than a guessed number (v1 lesson: unit
# assumptions fail silently).
# ---------------------------------------------------------------------------
_PROVISIONAL_SOURCE = (
    "PROVISIONAL: derived from RAPS adastraMI250.yaml per-device constants; "
    "replace with CINES-confirmed envelopes (D8 request pending)"
)

_MI250_IDLE_W = 8 * 44 + 1 * 90 + 37.13 + 4 * 20 + 0    # = 559.13
_MI250_MAX_W = 8 * 238 + 1 * 280 + 37.13 + 4 * 20 + 0   # = 2301.13

PROVISIONAL_ADASTRA_ENVELOPES: Dict[str, PartitionEnvelope] = {
    "mi250": PartitionEnvelope(
        idle_w=_MI250_IDLE_W, max_w=_MI250_MAX_W, shared=False,
        source=_PROVISIONAL_SOURCE,
    ),
    "mi250-shared": PartitionEnvelope(
        idle_w=_MI250_IDLE_W, max_w=_MI250_MAX_W, shared=True,
        source=_PROVISIONAL_SOURCE,
    ),
}


# ---------------------------------------------------------------------------
# 2. QC1 — dimensionless envelope check
# ---------------------------------------------------------------------------
def qc1_envelope_check(
    df: pd.DataFrame,
    envelopes: Dict[str, PartitionEnvelope],
    epsilon: float = 0.05,
) -> pd.DataFrame:
    """QC1: dimensionless per-node power-envelope check (Gerard's rule).

    Formalizes the CINES plausibility rule — "energy is incorrect when the
    average power derived from it is lower or higher than the idle or maximum
    power of the nodes used by the job" — as the plan's (§3 QC1) Phase-0
    non-dimensionalization applied as a validator:

        avg_power = energy_joules / (elapsed_seconds * nodes_required)
        ū         = (avg_power − P_idle) / (P_max − P_idle)
        valid    ⇔  ū ∈ [0 − ε, 1 + ε]

    Basis: *per-node share* — the job's total energy is divided evenly across
    its nodes (sacct reports one job-level energy; we have no per-node split
    at this stage). A multi-node job with one hot node and idle others can
    therefore still pass; QC2/QC3 (trace-level) catch that class.

    Shared-partition asymmetry: on `*-shared` partitions a job owns only a
    slice of each node, so its energy share can legitimately imply per-node
    power below idle — ū < 0 − ε there yields the *informational* verdict
    `below_idle_shared`, not a failure. ū > 1 + ε (`fail_high`) is impossible
    on any partition: a node slice cannot draw more than the whole node.

    Args:
        df: sacct-loader DataFrame; requires columns `partition`,
            `energy_joules`, `elapsed_seconds`, `nodes_required`.
        envelopes: partition name → PartitionEnvelope. Partitions absent
            from this dict get verdict `missing_envelope`.
        epsilon: dimensionless tolerance ε on both envelope edges.

    Returns:
        A copy of `df` with added columns:
          - `qc1_avg_power_w` (float64): implied average power per node,
            NaN where not computable.
          - `qc1_u` (float64): dimensionless utilization ū, NaN where
            avg-power is not computable or the partition has no envelope.
          - `qc1_verdict` (Categorical over QC1_VERDICTS): first-match-wins
            precedence — wraparound → missing_energy → zero_energy →
            zero_duration → missing_envelope → fail_high → fail_low →
            below_idle_shared → (not_computable) → pass.
    """
    out = df.copy()

    # Nullable extension dtypes -> plain float64 numpy with NaN for missing,
    # so comparisons return bool arrays rather than pd.NA.
    energy = out["energy_joules"].astype("Float64").to_numpy(
        dtype="float64", na_value=np.nan)
    elapsed = out["elapsed_seconds"].astype("Float64").to_numpy(
        dtype="float64", na_value=np.nan)
    nodes = out["nodes_required"].astype("Float64").to_numpy(
        dtype="float64", na_value=np.nan)

    # --- avg power (NaN where not computable) -----------------------------
    denom = elapsed * nodes
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_power = np.where(denom > 0, energy / denom, np.nan)
    out["qc1_avg_power_w"] = avg_power

    # --- per-partition envelope lookup (vectorized via Series.map) --------
    part = out["partition"].astype(str)
    idle = part.map({p: e.idle_w for p, e in envelopes.items()}).to_numpy(
        dtype="float64", na_value=np.nan)
    pmax = part.map({p: e.max_w for p, e in envelopes.items()}).to_numpy(
        dtype="float64", na_value=np.nan)
    shared = part.map(
        {p: e.shared for p, e in envelopes.items()}
    ).astype("boolean").to_numpy(dtype="bool", na_value=False)
    has_envelope = part.isin(envelopes.keys()).to_numpy(dtype="bool")

    with np.errstate(divide="ignore", invalid="ignore"):
        u = (avg_power - idle) / (pmax - idle)
    out["qc1_u"] = u

    # --- verdict, first-match-wins precedence ------------------------------
    # NaN-aware: comparisons against NaN are False, so each row falls through
    # to the first condition it genuinely meets.
    conditions = [
        energy > U64_WRAP_THRESHOLD,                    # wraparound
        np.isnan(energy),                               # missing_energy
        energy == 0,                                    # zero_energy
        np.isnan(elapsed) | (elapsed <= 0),             # zero_duration
        ~has_envelope,                                  # missing_envelope
        u > 1 + epsilon,                                # fail_high
        (u < 0 - epsilon) & ~shared,                    # fail_low
        (u < 0 - epsilon) & shared,                     # below_idle_shared
        np.isnan(u),                                    # not_computable
    ]
    choices = QC1_VERDICTS[:-1]  # everything but "pass"
    verdict = np.select(conditions, choices, default="pass")
    out["qc1_verdict"] = pd.Categorical(verdict, categories=QC1_VERDICTS)

    return out


# ---------------------------------------------------------------------------
# 3. QC2 — cross-source energy consistency (interface now, pm_counters later)
# ---------------------------------------------------------------------------
@runtime_checkable
class NodePowerSource(Protocol):
    """Pluggable per-node energy source for QC2 (plan §3 QC2).

    The eventual W3 implementation will wrap the pm_counters ingest
    (blocked on CINES schema confirmation, defect D8). When it lands, it
    SHOULD compute job energy by *cumulative-energy-counter differencing* —
    counter(t_end) − counter(t_start) — rather than integrating instantaneous
    power samples: per Bertrand, the raw pm_counters logs carry both a
    cumulative energy counter (J) and instantaneous power (W) per domain, and
    the cumulative counter is an exact integral (no quadrature error from the
    ~60 s sampling cadence). Counter resets/rollbacks within the window are a
    QC3 concern; on detecting one this method should return None rather than
    a corrupted difference.
    """

    def job_energy_joules(
        self, node: str, t_start: float, t_end: float
    ) -> Optional[float]:
        """Energy consumed by ONE node over [t_start, t_end) epoch seconds.

        Returns None when the source has no (or insufficient) trace data to
        cover the window for that node.
        """
        ...


class SyntheticPowerSource:
    """NodePowerSource backed by constant per-node power values — for tests.

    `powers_w` maps node name → constant power draw in watts; energy over a
    window is simply P · (t_end − t_start). Unknown nodes return None,
    emulating trace gaps.
    """

    def __init__(self, powers_w: Dict[str, float]):
        self.powers_w = dict(powers_w)

    def job_energy_joules(
        self, node: str, t_start: float, t_end: float
    ) -> Optional[float]:
        power = self.powers_w.get(node)
        if power is None:
            return None
        if not (math.isfinite(t_start) and math.isfinite(t_end)) or t_end <= t_start:
            return None
        return power * (t_end - t_start)


def qc2_cross_source(
    df: pd.DataFrame,
    source: NodePowerSource,
    rel_tol: float = 0.10,
) -> pd.DataFrame:
    """QC2: Slurm ConsumedEnergyRaw vs. independent per-node trace energy.

    For each job, sums `source.job_energy_joules(node, start_time, end_time)`
    over the job's `node_list` and compares against the sacct-side
    `energy_joules`. Disagreement beyond `rel_tol` localizes which source is
    wrong when combined with QC1 on each side (plan §3 QC2).

    Args:
        df: sacct-loader DataFrame; requires columns `energy_joules`,
            `node_list`, `start_time`, `end_time`.
        source: a NodePowerSource (pm_counters-backed once W3 lands;
            SyntheticPowerSource for tests).
        rel_tol: relative-error agreement threshold.

    Returns:
        A copy of `df` with added columns:
          - `qc2_trace_energy_joules` (float64): sum of per-node trace
            energies over [start_time, end_time); NaN if the job has an
            empty node_list, a missing/invalid time window, or ANY node
            returns None (partial sums would silently understate energy).
          - `qc2_rel_error` (float64): |sacct − trace| / max(trace, 1);
            the 1 J floor guards the trace≈0 case.
          - `qc2_verdict` (Categorical over QC2_VERDICTS), precedence:
            missing_trace → missing_energy → agree →
            disagree_sacct_low / disagree_sacct_high.
    """
    out = df.copy()

    # Row-wise loop, deliberately: node_list is a ragged list-valued column
    # and each (node, window) lookup is a call into an arbitrary trace
    # backend — not expressible as a vectorized pandas op. Acceptable here
    # because QC2 runs are bounded by trace availability (only jobs whose
    # nodes/window the pm_counters corpus covers), unlike QC1 which must
    # sweep all 250k sacct rows.
    starts = out["start_time"].astype("Float64").to_numpy(
        dtype="float64", na_value=np.nan)
    ends = out["end_time"].astype("Float64").to_numpy(
        dtype="float64", na_value=np.nan)

    trace_energy = np.full(len(out), np.nan)
    for i, (nodes, t0, t1) in enumerate(zip(out["node_list"], starts, ends)):
        if not isinstance(nodes, (list, tuple)) or len(nodes) == 0:
            continue
        if not (math.isfinite(t0) and math.isfinite(t1)):
            continue
        total = 0.0
        complete = True
        for node in nodes:
            e = source.job_energy_joules(node, t0, t1)
            if e is None:
                complete = False
                break
            total += e
        if complete:
            trace_energy[i] = total
    out["qc2_trace_energy_joules"] = trace_energy

    sacct_energy = out["energy_joules"].astype("Float64").to_numpy(
        dtype="float64", na_value=np.nan)
    with np.errstate(invalid="ignore"):
        rel_error = np.abs(sacct_energy - trace_energy) / np.maximum(trace_energy, 1.0)
    out["qc2_rel_error"] = rel_error

    conditions = [
        np.isnan(trace_energy),                                   # missing_trace
        np.isnan(sacct_energy),                                   # missing_energy
        rel_error <= rel_tol,                                     # agree
        (rel_error > rel_tol) & (sacct_energy < trace_energy),    # disagree_sacct_low
        (rel_error > rel_tol) & (sacct_energy >= trace_energy),   # disagree_sacct_high
    ]
    verdict = np.select(conditions, QC2_VERDICTS, default="missing_trace")
    out["qc2_verdict"] = pd.Categorical(verdict, categories=QC2_VERDICTS)

    return out


# ---------------------------------------------------------------------------
# 4. PipelineStep wrapper for QC1 (QC2 gets no step until W3 lands)
# ---------------------------------------------------------------------------
def _parse_envelope_config(raw: Dict) -> Dict[str, PartitionEnvelope]:
    """Parse a ctx.config envelope mapping into PartitionEnvelope objects.

    Expected shape: {partition: {"idle_w": float, "max_w": float,
    "shared": bool, "source": str (optional)}}. Values that are already
    PartitionEnvelope instances pass through unchanged.
    """
    parsed = {}
    for partition, spec in raw.items():
        if isinstance(spec, PartitionEnvelope):
            parsed[partition] = spec
        else:
            parsed[partition] = PartitionEnvelope(
                idle_w=float(spec["idle_w"]),
                max_w=float(spec["max_w"]),
                shared=bool(spec.get("shared", False)),
                source=str(spec.get("source", "ctx.config['power_qc']['envelopes']")),
            )
    return parsed


def _build_power_envelope_qc_step():
    """Build the PowerEnvelopeQC PipelineStep class (lazy import of
    ..pipeline to match the pattern used elsewhere in this package)."""
    from ...pipeline import PipelineStep, PipelineStage, PipelineContext

    class PowerEnvelopeQC(PipelineStep):
        """QC1 dimensionless power-envelope check as a PREPROCESS step.

        Envelopes come from ctx.config["power_qc"]["envelopes"] when present
        (partition → {idle_w, max_w, shared[, source]}); otherwise falls back
        to PROVISIONAL_ADASTRA_ENVELOPES with a logged warning (mi250 only —
        genoa partitions then get verdict `missing_envelope` pending CINES).

        NOT registered in get_preprocess_steps() — wiring is a separate
        change, same policy as LoadAdastraSacctDump.
        """

        name = "power_envelope_qc"
        stage = PipelineStage.PREPROCESS
        description = "QC1: per-job dimensionless power-envelope check"

        _REQUIRED_COLUMNS = (
            "partition", "energy_joules", "elapsed_seconds", "nodes_required",
        )

        def validate(self, ctx: PipelineContext) -> bool:
            if not isinstance(ctx.data, pd.DataFrame):
                return False
            return all(c in ctx.data.columns for c in self._REQUIRED_COLUMNS)

        def execute(self, ctx: PipelineContext) -> PipelineContext:
            try:
                raw_envelopes = ctx.config.get("power_qc", {}).get("envelopes")
                if raw_envelopes:
                    envelopes = _parse_envelope_config(raw_envelopes)
                    logger.info(
                        f"power_envelope_qc: using {len(envelopes)} envelope(s) "
                        f"from ctx.config['power_qc']['envelopes']"
                    )
                else:
                    envelopes = PROVISIONAL_ADASTRA_ENVELOPES
                    logger.warning(
                        "power_envelope_qc: no envelopes in "
                        "ctx.config['power_qc']['envelopes']; falling back to "
                        "PROVISIONAL_ADASTRA_ENVELOPES (mi250/mi250-shared "
                        "only, derived from RAPS adastraMI250.yaml — replace "
                        "with CINES-confirmed values; genoa partitions will "
                        "get verdict 'missing_envelope')"
                    )

                epsilon = ctx.config.get("power_qc", {}).get("epsilon", 0.05)
                df = qc1_envelope_check(ctx.data, envelopes, epsilon=epsilon)
                ctx.data = df

                counts = df["qc1_verdict"].value_counts()
                ctx.metadata["qc1_verdicts"] = {
                    k: int(v) for k, v in counts.items() if v > 0
                }
                ctx.metadata["qc1_epsilon"] = epsilon
                ctx.metadata["qc1_envelope_sources"] = {
                    p: e.source for p, e in envelopes.items()
                }

                logger.info(
                    f"power_envelope_qc: {len(df):,} jobs -> "
                    f"{ctx.metadata['qc1_verdicts']}"
                )

            except Exception as e:
                ctx.add_error(f"Error in power envelope QC: {str(e)}")

            return ctx

    return PowerEnvelopeQC


PowerEnvelopeQC = _build_power_envelope_qc_step()


# ---------------------------------------------------------------------------
# 5. QC3 — trace-level per-device stream checks (plan §3 QC3)
# ---------------------------------------------------------------------------

# Negative-delta taxonomy for the cumulative-energy counter, in PRECEDENCE
# order (first match wins). These labels are HYPOTHESES about the mechanism
# behind a measured negative delta, not confirmed semantics — the pm_counters
# column schema itself is unconfirmed (D8, awaiting CINES).
QC3_RESET_CLASSES = ("u64_wrap", "jitter", "reset_to_zero", "rollback_other")

# ΔE more negative than −2^62 J is physically absurd (cf. U64_WRAP_THRESHOLD
# above) and is taken as an unsigned-64-bit counter wrapping / being
# reinterpreted. Checked FIRST: a u64 wrap also lands "near zero relative to
# the old value", so it would otherwise be misclassified as reset_to_zero.
QC3_U64_WRAP_DELTA_J = -float(2**62)

# Fallback |ΔE| bound for the "jitter" class when no device_p_max is given.
QC3_JITTER_FALLBACK_J = 1.0e4

# "reset_to_zero": new counter value at or below this fraction of the old one.
QC3_RESET_ZERO_FRACTION = 0.01


@dataclass
class Qc3StreamReport:
    """Result of `qc3_check_stream` on ONE per-device stream.

    Index convention: interval-level defects (timestamp decrease/duplicate,
    sampling gap, counter reset, energy/power inconsistency) between samples
    i and i+1 are reported at index i+1 — the first sample AFTER the
    defective interval. Sample-level defects (power envelope, stuck runs) use
    the sample's own index; stuck runs are (start_index, run_length, value).

    `checks_run` lists which checks had sufficient inputs; a check absent
    from it reports zero counts vacuously, not as evidence of health.
    """

    n_samples: int
    checks_run: tuple
    # timestamp_nonmonotonic
    ts_decreasing_count: int
    ts_decreasing_idx: np.ndarray
    ts_duplicate_count: int
    ts_duplicate_idx: np.ndarray
    # sampling_gap
    median_dt_s: float
    gap_count: int
    gap_idx: np.ndarray
    gap_max_s: float
    gap_total_s: float
    # counter_reset
    reset_count: int
    reset_idx: np.ndarray
    reset_class_idx: Dict[str, np.ndarray]
    # energy_power_consistency
    epc_high_count: int
    epc_high_idx: np.ndarray
    epc_low_count: int
    epc_low_idx: np.ndarray
    # power_out_of_envelope
    pow_high_count: int
    pow_high_idx: np.ndarray
    pow_low_count: int
    pow_low_idx: np.ndarray
    # stuck_value (informational — see qc3_check_stream docstring)
    stuck_run_count: int
    stuck_runs: tuple
    stuck_sample_count: int

    @property
    def has_defects(self) -> bool:
        """True if any non-informational check flagged anything.

        Includes duplicate timestamps (a trace anomaly even though not
        strictly decreasing). EXCLUDES stuck_value, which is informational.
        """
        return (
            self.ts_decreasing_count + self.ts_duplicate_count
            + self.gap_count + self.reset_count
            + self.epc_high_count + self.epc_low_count
            + self.pow_high_count + self.pow_low_count
        ) > 0

    def summary(self) -> Dict:
        """Flat dict of per-check counts for logging/aggregation."""
        return {
            "n_samples": self.n_samples,
            "checks_run": list(self.checks_run),
            "ts_decreasing": self.ts_decreasing_count,
            "ts_duplicate": self.ts_duplicate_count,
            "median_dt_s": self.median_dt_s,
            "gap_count": self.gap_count,
            "gap_max_s": self.gap_max_s,
            "gap_total_s": self.gap_total_s,
            "reset_count": self.reset_count,
            "reset_by_class": {
                c: int(v.size) for c, v in self.reset_class_idx.items()
            },
            "epc_high": self.epc_high_count,
            "epc_low": self.epc_low_count,
            "pow_high": self.pow_high_count,
            "pow_low": self.pow_low_count,
            "stuck_runs": self.stuck_run_count,
            "stuck_samples": self.stuck_sample_count,
            "has_defects": self.has_defects,
        }


def qc3_check_stream(
    t: np.ndarray,
    energy_cum: Optional[np.ndarray] = None,
    power_w: Optional[np.ndarray] = None,
    device_p_max: Optional[float] = None,
    gap_factor: float = 5.0,
    stuck_min_run: int = 30,
    epsilon: float = 0.05,
) -> Qc3StreamReport:
    """QC3: trace-level checks on ONE per-device power/energy stream.

    Operates on generic numpy streams — `t` (epoch seconds), optional
    `energy_cum` (cumulative Joules), optional `power_w` (instantaneous
    Watts) — with NO pm_counters-specific parsing: the raw-log column schema
    is unconfirmed (D8, awaiting CINES), so the caller is responsible for
    mapping whatever columns it trusts onto these three arrays. Each check
    is skipped (absent from `checks_run`) when its inputs are missing;
    NaN samples never trigger a defect (comparisons with NaN are False),
    so gappy inputs degrade to fewer checked intervals, not false alarms.

    Checks:
      - timestamp_nonmonotonic (t): strictly decreasing consecutive pairs;
        exact duplicate timestamps counted separately.
      - sampling_gap (t): dt > gap_factor × median(positive dt). Reports
        count, max gap, and total gapped seconds — gaps matter downstream
        because QC2's counter-differencing integrates across them blindly.
      - counter_reset (energy_cum): negative counter deltas, classified in
        precedence order (QC3_RESET_CLASSES): `u64_wrap` (ΔE < −2^62 J —
        checked first because a wrap also looks like a reset-to-zero),
        `jitter` (|ΔE| ≤ device_p_max·dt when device_p_max is given, else
        ≤ QC3_JITTER_FALLBACK_J — plausibly out-of-order/near-simultaneous
        counter reads), `reset_to_zero` (old > 0 and new ≤ 1% of old),
        `rollback_other` (everything else). The class names are HYPOTHESES
        about mechanism, to be revisited against the CINES schema answer.
      - energy_power_consistency (energy_cum + device_p_max): implied power
        ΔE/Δt over each interval with Δt > 0 must lie in
        [−ε·p_max, p_max·(1+ε)]. This is the strong check: it catches
        trace-level corruption (e.g. a counter mid-stream scaled or spliced)
        that per-sample checks miss. Reset intervals necessarily also fail
        low here; consumers should de-duplicate against `reset_idx`.
      - power_out_of_envelope (power_w + device_p_max): per-sample power
        outside [−ε·p_max, p_max·(1+ε)].
      - stuck_value (power_w): runs of ≥ stuck_min_run consecutive IDENTICAL
        power values. INFORMATIONAL, not a failure: constant idle power
        reported in integer watts can be legitimate for a parked node, so a
        stuck run is a review flag, not a verdict. Default 30 samples ≈ 30
        minutes at the ~60 s pm_counters cadence. NaN breaks a run.

    Deliberately NO slew-rate check: at ~60 s cadence a node can
    legitimately swing idle→max (or back) within a single sampling
    interval, so any per-interval dP/dt bound would be an arbitrary knob
    that flags real workload transitions. Slew checking only becomes
    meaningful on the ~9 Hz underlying counters, which we do not have.

    Args:
        t: sample timestamps, epoch seconds (float array, length n).
        energy_cum: cumulative energy counter, Joules (length n) or None.
        power_w: instantaneous power, Watts (length n) or None.
        device_p_max: physical max power of the device this stream
            measures, Watts; enables the envelope/consistency checks and
            scales the reset `jitter` threshold. None disables those.
        gap_factor: dt threshold multiplier on the median positive dt.
        stuck_min_run: minimum identical-run length to report.
        epsilon: dimensionless envelope tolerance (same role as QC1's ε).

    Returns:
        Qc3StreamReport (see its docstring for the index convention).
    """
    t = np.asarray(t, dtype="float64")
    n = int(t.size)
    if energy_cum is not None:
        energy_cum = np.asarray(energy_cum, dtype="float64")
        if energy_cum.size != n:
            raise ValueError(
                f"energy_cum length {energy_cum.size} != t length {n}")
    if power_w is not None:
        power_w = np.asarray(power_w, dtype="float64")
        if power_w.size != n:
            raise ValueError(
                f"power_w length {power_w.size} != t length {n}")

    checks_run = ["timestamp_nonmonotonic", "sampling_gap"]
    empty_idx = np.empty(0, dtype=np.int64)

    # --- timestamp_nonmonotonic -------------------------------------------
    dt = np.diff(t) if n >= 2 else np.empty(0, dtype="float64")
    with np.errstate(invalid="ignore"):
        ts_decreasing_idx = np.flatnonzero(dt < 0) + 1
        ts_duplicate_idx = np.flatnonzero(dt == 0) + 1

    # --- sampling_gap -------------------------------------------------------
    pos_dt = dt[dt > 0]
    median_dt = float(np.median(pos_dt)) if pos_dt.size else float("nan")
    if math.isfinite(median_dt) and median_dt > 0:
        gap_mask = dt > gap_factor * median_dt
    else:
        gap_mask = np.zeros(dt.shape, dtype=bool)
    gap_idx = np.flatnonzero(gap_mask) + 1
    gap_max_s = float(dt[gap_mask].max()) if gap_idx.size else 0.0
    gap_total_s = float(dt[gap_mask].sum()) if gap_idx.size else 0.0

    # --- counter_reset ------------------------------------------------------
    reset_idx = empty_idx
    reset_class_idx = {c: empty_idx for c in QC3_RESET_CLASSES}
    dE = None
    if energy_cum is not None and n >= 2:
        checks_run.append("counter_reset")
        dE = np.diff(energy_cum)  # NaN where either endpoint NaN -> excluded
        old = energy_cum[:-1]
        new = energy_cum[1:]
        if device_p_max is not None:
            jitter_thresh = device_p_max * np.maximum(dt, 0.0)
        else:
            jitter_thresh = np.full(dt.shape, QC3_JITTER_FALLBACK_J)
        with np.errstate(invalid="ignore"):
            neg = dE < 0
            is_wrap = neg & (dE < QC3_U64_WRAP_DELTA_J)
            is_jitter = neg & ~is_wrap & (np.abs(dE) <= jitter_thresh)
            is_zero = (neg & ~is_wrap & ~is_jitter & (old > 0)
                       & (new <= QC3_RESET_ZERO_FRACTION * old))
            is_other = neg & ~is_wrap & ~is_jitter & ~is_zero
        reset_idx = np.flatnonzero(neg) + 1
        reset_class_idx = {
            "u64_wrap": np.flatnonzero(is_wrap) + 1,
            "jitter": np.flatnonzero(is_jitter) + 1,
            "reset_to_zero": np.flatnonzero(is_zero) + 1,
            "rollback_other": np.flatnonzero(is_other) + 1,
        }

    # --- energy_power_consistency -------------------------------------------
    epc_high_idx = empty_idx
    epc_low_idx = empty_idx
    if dE is not None and device_p_max is not None:
        checks_run.append("energy_power_consistency")
        with np.errstate(divide="ignore", invalid="ignore"):
            implied_w = np.where(dt > 0, dE / dt, np.nan)
            epc_high_idx = np.flatnonzero(
                implied_w > device_p_max * (1.0 + epsilon)) + 1
            epc_low_idx = np.flatnonzero(
                implied_w < -epsilon * device_p_max) + 1

    # --- power_out_of_envelope -----------------------------------------------
    pow_high_idx = empty_idx
    pow_low_idx = empty_idx
    if power_w is not None and device_p_max is not None:
        checks_run.append("power_out_of_envelope")
        with np.errstate(invalid="ignore"):
            pow_high_idx = np.flatnonzero(
                power_w > device_p_max * (1.0 + epsilon))
            pow_low_idx = np.flatnonzero(power_w < -epsilon * device_p_max)

    # --- stuck_value (informational) ----------------------------------------
    stuck_runs = []
    if power_w is not None and n >= 2 and stuck_min_run >= 2:
        checks_run.append("stuck_value")
        with np.errstate(invalid="ignore"):
            same = power_w[1:] == power_w[:-1]  # False across NaN
        i = 0
        m = same.size
        while i < m:
            if same[i]:
                j = i
                while j < m and same[j]:
                    j += 1
                run_len = j - i + 1  # samples i .. j inclusive
                if run_len >= stuck_min_run:
                    stuck_runs.append((int(i), int(run_len),
                                       float(power_w[i])))
                i = j
            else:
                i += 1

    return Qc3StreamReport(
        n_samples=n,
        checks_run=tuple(checks_run),
        ts_decreasing_count=int(ts_decreasing_idx.size),
        ts_decreasing_idx=ts_decreasing_idx,
        ts_duplicate_count=int(ts_duplicate_idx.size),
        ts_duplicate_idx=ts_duplicate_idx,
        median_dt_s=median_dt,
        gap_count=int(gap_idx.size),
        gap_idx=gap_idx,
        gap_max_s=gap_max_s,
        gap_total_s=gap_total_s,
        reset_count=int(reset_idx.size),
        reset_idx=reset_idx,
        reset_class_idx=reset_class_idx,
        epc_high_count=int(epc_high_idx.size),
        epc_high_idx=epc_high_idx,
        epc_low_count=int(epc_low_idx.size),
        epc_low_idx=epc_low_idx,
        pow_high_count=int(pow_high_idx.size),
        pow_high_idx=pow_high_idx,
        pow_low_count=int(pow_low_idx.size),
        pow_low_idx=pow_low_idx,
        stuck_run_count=len(stuck_runs),
        stuck_runs=tuple(stuck_runs),
        stuck_sample_count=sum(r[1] for r in stuck_runs),
    )


# ---------------------------------------------------------------------------
# 6. Self-tests (synthetic known-answer) + real-data verification report
# ---------------------------------------------------------------------------
def _run_synthetic_qc1_tests() -> int:
    """Known-answer tests exercising every QC1 verdict. Returns #failures.

    Test envelope: idle=400 W, max=2400 W (span 2000 W), ε=0.05.
    Hand computations are in each case's comment.
    """
    envelopes = {
        "excl": PartitionEnvelope(400.0, 2400.0, shared=False, source="test"),
        "shr": PartitionEnvelope(400.0, 2400.0, shared=True, source="test"),
    }
    WRAP = float(2**64 - 100)  # u64 counter that wrapped to -100

    # (case name, partition, energy J, elapsed s, nodes,
    #  expected verdict, expected u or None)
    cases = [
        # 240 kJ / (100 s * 1 node) = 2400 W -> u = (2400-400)/2000 = 1.00
        ("pass at exactly max", "excl", 240_000.0, 100, 1, "pass", 1.00),
        # 250 kJ -> 2500 W -> u = 1.05: NOT > 1+eps -> boundary pass
        ("pass at u=1+eps boundary", "excl", 250_000.0, 100, 1, "pass", 1.05),
        # 260 kJ -> 2600 W -> u = 1.10 > 1.05 -> fail_high
        ("fail_high exclusive", "excl", 260_000.0, 100, 1, "fail_high", 1.10),
        # 30 kJ -> 300 W -> u = -0.05: NOT < 0-eps -> boundary pass
        ("pass at u=-eps boundary", "excl", 30_000.0, 100, 1, "pass", -0.05),
        # 20 kJ -> 200 W -> u = -0.10 < -0.05, exclusive -> fail_low
        ("fail_low exclusive", "excl", 20_000.0, 100, 1, "fail_low", -0.10),
        # same on shared partition -> informational, not a failure
        ("below_idle_shared", "shr", 20_000.0, 100, 1, "below_idle_shared", -0.10),
        # above-max is impossible even on shared partitions
        ("fail_high on shared", "shr", 260_000.0, 100, 1, "fail_high", 1.10),
        # 2-node job: 480 kJ / (100 s * 2) = 2400 W -> u = 1.0 -> pass
        ("pass multi-node share", "excl", 480_000.0, 100, 2, "pass", 1.00),
        # wraparound wins over everything (here: also zero duration)
        ("wraparound precedence", "excl", WRAP, 0, 1, "wraparound", None),
        # missing energy wins over zero duration
        ("missing_energy precedence", "excl", None, 0, 1, "missing_energy", None),
        # zero energy wins over zero duration
        ("zero_energy precedence", "excl", 0.0, 0, 1, "zero_energy", None),
        ("zero_duration (elapsed=0)", "excl", 1_000.0, 0, 1, "zero_duration", None),
        ("zero_duration (elapsed null)", "excl", 1_000.0, None, 1, "zero_duration", None),
        # genoa deliberately has no envelope in this test dict
        ("missing_envelope", "genoa", 1_000.0, 100, 1, "missing_envelope", None),
        # nodes_required null -> u NaN -> defensive catch-all
        ("not_computable (nodes null)", "excl", 1_000.0, 100, None, "not_computable", None),
    ]

    df = pd.DataFrame({
        "job_id": [str(i) for i in range(len(cases))],
        "partition": [c[1] for c in cases],
        "energy_joules": pd.array([c[2] for c in cases], dtype="Float64"),
        "elapsed_seconds": pd.array([c[3] for c in cases], dtype="Int64"),
        "nodes_required": pd.array([c[4] for c in cases], dtype="Int64"),
    })
    result = qc1_envelope_check(df, envelopes, epsilon=0.05)

    failures = 0
    for i, (name, _p, _e, _t, _n, want_verdict, want_u) in enumerate(cases):
        got_verdict = str(result.loc[i, "qc1_verdict"])
        got_u = result.loc[i, "qc1_u"]
        ok = got_verdict == want_verdict
        if want_u is not None:
            ok = ok and math.isfinite(got_u) and abs(got_u - want_u) < 1e-9
        status = "PASS" if ok else "FAIL"
        u_str = f"{got_u:+.3f}" if math.isfinite(got_u) else "  NaN"
        print(f"    [{status}] qc1: {name:<32s} verdict={got_verdict:<18s} "
              f"(want {want_verdict:<18s}) u={u_str}")
        if not ok:
            failures += 1
    return failures


def _run_synthetic_qc2_tests() -> int:
    """Known-answer tests exercising every QC2 verdict. Returns #failures.

    SyntheticPowerSource: n1=500 W, n2=500 W. Window 1000->1100 (100 s), so
    trace energy is 50 kJ per node.
    """
    source = SyntheticPowerSource({"n1": 500.0, "n2": 500.0})

    # (case name, node_list, start, end, sacct energy J,
    #  expected verdict, expected trace J or None)
    cases = [
        # trace = 2 nodes * 50 kJ = 100 kJ; sacct 105 kJ -> rel = 0.05 <= 0.10
        ("agree", ["n1", "n2"], 1000.0, 1100.0, 105_000.0, "agree", 100_000.0),
        # rel error exactly 0.10 -> still agree (<=)
        ("agree at rel_tol boundary", ["n1", "n2"], 1000.0, 1100.0, 110_000.0,
         "agree", 100_000.0),
        # sacct 50 kJ vs trace 100 kJ -> rel = 0.50, sacct low
        ("disagree_sacct_low", ["n1", "n2"], 1000.0, 1100.0, 50_000.0,
         "disagree_sacct_low", 100_000.0),
        # sacct 200 kJ vs trace 100 kJ -> rel = 1.00, sacct high
        ("disagree_sacct_high", ["n1", "n2"], 1000.0, 1100.0, 200_000.0,
         "disagree_sacct_high", 100_000.0),
        # unknown node -> source returns None -> whole job missing_trace
        ("missing_trace (unknown node)", ["n1", "nX"], 1000.0, 1100.0,
         100_000.0, "missing_trace", None),
        # no start time -> window invalid -> missing_trace
        ("missing_trace (no window)", ["n1"], None, 1100.0, 100_000.0,
         "missing_trace", None),
        # empty node list -> missing_trace (never a fake zero-energy trace)
        ("missing_trace (empty nodes)", [], 1000.0, 1100.0, 100_000.0,
         "missing_trace", None),
        # trace fine, sacct energy null -> missing_energy
        ("missing_energy", ["n1"], 1000.0, 1100.0, None, "missing_energy",
         50_000.0),
        # precedence: both sides missing -> missing_trace wins
        ("missing_trace precedence", ["nX"], 1000.0, 1100.0, None,
         "missing_trace", None),
    ]

    df = pd.DataFrame({
        "job_id": [str(i) for i in range(len(cases))],
        "node_list": [c[1] for c in cases],
        "start_time": pd.array([c[2] for c in cases], dtype="Float64"),
        "end_time": pd.array([c[3] for c in cases], dtype="Float64"),
        "energy_joules": pd.array([c[4] for c in cases], dtype="Float64"),
    })
    result = qc2_cross_source(df, source, rel_tol=0.10)

    failures = 0
    for i, (name, _nl, _t0, _t1, _e, want_verdict, want_trace) in enumerate(cases):
        got_verdict = str(result.loc[i, "qc2_verdict"])
        got_trace = result.loc[i, "qc2_trace_energy_joules"]
        ok = got_verdict == want_verdict
        if want_trace is None:
            ok = ok and not math.isfinite(got_trace)
        else:
            ok = ok and math.isfinite(got_trace) and abs(got_trace - want_trace) < 1e-6
        status = "PASS" if ok else "FAIL"
        trace_str = f"{got_trace:,.0f}J" if math.isfinite(got_trace) else "NaN"
        print(f"    [{status}] qc2: {name:<32s} verdict={got_verdict:<20s} "
              f"(want {want_verdict:<20s}) trace={trace_str}")
        if not ok:
            failures += 1
    return failures


def _run_synthetic_qc3_tests() -> int:
    """Known-answer tests exercising every QC3 check and every counter_reset
    subclass, on streams with hand-planted defects at known indices. Returns
    #failures. Hand computations are in each scenario's comments.
    """
    failures = 0

    def check(name, got, want):
        nonlocal failures
        if isinstance(got, np.ndarray):
            got = got.tolist()
        if isinstance(want, np.ndarray):
            want = want.tolist()
        if isinstance(want, float) and isinstance(got, float):
            ok = (math.isnan(got) and math.isnan(want)) or got == want
        else:
            ok = got == want
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] qc3: {name:<44s} got={got!r} (want {want!r})")
        if not ok:
            failures += 1

    # --- A. clean stream: 100 samples @ 60 s, exactly 500 W ----------------
    n = 100
    t = 1000.0 + 60.0 * np.arange(n)
    e = 1.0e6 + 30000.0 * np.arange(n)                    # 30 kJ / 60 s = 500 W
    p = np.where(np.arange(n) % 2 == 0, 480.0, 520.0)     # varies: no stuck run
    r = qc3_check_stream(t, e, p, device_p_max=1000.0)
    check("A clean: has_defects", r.has_defects, False)
    check("A clean: all 6 checks ran", len(r.checks_run), 6)
    check("A clean: median_dt_s", r.median_dt_s, 60.0)
    check("A clean: stuck_run_count", r.stuck_run_count, 0)

    # --- B. timestamp defects ----------------------------------------------
    # t[50] = t[49]-10 -> decreasing pair at idx 50 (dt[50]=130 <= 5*60: no gap)
    # t[70] = t[69]    -> exact duplicate at idx 70 (dt[70]=120: no gap)
    # t[90:] += 3600   -> dt[89] = 3660 > 300 -> ONE gap at idx 90
    t = 1000.0 + 60.0 * np.arange(n)
    t[50] = t[49] - 10.0
    t[70] = t[69]
    t[90:] += 3600.0
    r = qc3_check_stream(t)
    check("B ts: decreasing idx", r.ts_decreasing_idx, [50])
    check("B ts: duplicate idx", r.ts_duplicate_idx, [70])
    check("B ts: gap idx", r.gap_idx, [90])
    check("B ts: gap_max_s", r.gap_max_s, 3660.0)
    check("B ts: gap_total_s", r.gap_total_s, 3660.0)
    check("B ts: median_dt_s still 60", r.median_dt_s, 60.0)
    check("B ts: only t-checks ran", list(r.checks_run),
          ["timestamp_nonmonotonic", "sampling_gap"])

    # --- C. counter resets: one defect per subclass, p_max=1000 ------------
    # Baseline dE=+30 kJ (500 W). Planted (offending sample = i+1):
    #   dE[19] = -100        -> jitter at idx 20 (|dE| <= p_max*dt = 60 kJ)
    #   e[40]  = 5.0         -> reset_to_zero at idx 40 (old ~2.17 MJ, new 5 J)
    #                           side effect: dE[40] huge positive -> epc_high 41
    #   dE[59] = -2^63       -> u64_wrap at idx 60 (< -2^62)
    #   dE[79] = -1e6        -> rollback_other at idx 80 (not small, old < 0
    #                           after the wrap so the reset_to_zero old>0
    #                           guard is exercised)
    # epc_low expected at [40, 60, 80]: implied W = -36165, -1.5e17, -16667,
    # all < -eps*p_max = -50; idx 20 implied -1.67 W is NOT epc_low.
    de = np.full(n - 1, 30000.0)
    de[19] = -100.0
    de[59] = -float(2**63)
    de[79] = -1.0e6
    e = 1.0e6 + np.concatenate([[0.0], np.cumsum(de)])
    e[40] = 5.0
    t = 60.0 * np.arange(n)
    r = qc3_check_stream(t, e, device_p_max=1000.0)
    check("C reset: reset_idx", r.reset_idx, [20, 40, 60, 80])
    check("C reset: jitter idx", r.reset_class_idx["jitter"], [20])
    check("C reset: reset_to_zero idx", r.reset_class_idx["reset_to_zero"], [40])
    check("C reset: u64_wrap idx", r.reset_class_idx["u64_wrap"], [60])
    check("C reset: rollback_other idx", r.reset_class_idx["rollback_other"], [80])
    check("C reset: epc_low idx", r.epc_low_idx, [40, 60, 80])
    check("C reset: epc_high idx (post-reset jump)", r.epc_high_idx, [41])

    # --- D. energy/power consistency + power envelope + stuck --------------
    # p_max=1000, eps=0.05 -> power bounds (-50, 1050), exclusive.
    #   dE[24] = 90000 -> implied 1500 W -> epc_high at idx 25
    #   dE[49] = -12000 -> implied -200 W -> epc_low at idx 50; |dE| <= 60 kJ
    #                      so the SAME interval is also a jitter reset
    #                      (documented overlap between the two checks)
    #   dE[70] = 63000 -> implied exactly 1050.0 W: 1050.0 is NOT >
    #                      1000*(1+0.05) (which rounds up in float64) ->
    #                      boundary NOT flagged
    #   p[10]=1200 -> pow_high; p[11]=-100 -> pow_low
    #   p[12]=1050.0, p[13]=-50.0 -> exact bounds, NOT flagged
    #   p[30:59] = 350.0 -> 29 identical: below stuck_min_run=30, not flagged
    #   p[60:90] = 400.0 -> 30 identical: ONE stuck run (60, 30, 400.0)
    nn = 120
    t = 60.0 * np.arange(nn)
    de = np.full(nn - 1, 30000.0)
    de[24] = 90000.0
    de[49] = -12000.0
    de[70] = 63000.0
    e = 1.0e6 + np.concatenate([[0.0], np.cumsum(de)])
    p = np.where(np.arange(nn) % 2 == 0, 490.0, 510.0)
    p[10] = 1200.0
    p[11] = -100.0
    p[12] = 1050.0
    p[13] = -50.0
    p[30:59] = 350.0
    p[60:90] = 400.0
    r = qc3_check_stream(t, e, p, device_p_max=1000.0)
    check("D epc: high idx (boundary excluded)", r.epc_high_idx, [25])
    check("D epc: low idx", r.epc_low_idx, [50])
    check("D epc: same interval also jitter reset",
          r.reset_class_idx["jitter"], [50])
    check("D pow: high idx (boundary excluded)", r.pow_high_idx, [10])
    check("D pow: low idx (boundary excluded)", r.pow_low_idx, [11])
    check("D stuck: runs (29-run ignored)", list(r.stuck_runs),
          [(60, 30, 400.0)])
    check("D stuck: sample count", r.stuck_sample_count, 30)
    check("D stuck: informational, so 29+30 runs but has_defects from "
          "epc/pow only", r.has_defects, True)

    # --- E. skippable checks + fallback jitter threshold (no p_max) --------
    #   dE[9]  = -5000  -> |dE| <= fallback 1e4 -> jitter at idx 10
    #   dE[19] = -20000 -> > 1e4, new/old ~ 1 -> rollback_other at idx 20
    t = 60.0 * np.arange(50)
    de = np.full(49, 30000.0)
    de[9] = -5000.0
    de[19] = -20000.0
    e = 1.0e7 + np.concatenate([[0.0], np.cumsum(de)])
    r = qc3_check_stream(t, energy_cum=e)  # no power_w, no device_p_max
    check("E fallback: checks_run", list(r.checks_run),
          ["timestamp_nonmonotonic", "sampling_gap", "counter_reset"])
    check("E fallback: jitter idx (<=1e4 J)", r.reset_class_idx["jitter"], [10])
    check("E fallback: rollback_other idx",
          r.reset_class_idx["rollback_other"], [20])
    check("E fallback: epc skipped (no p_max)",
          r.epc_high_count + r.epc_low_count, 0)

    # --- F. gap boundary: threshold is strict > gap_factor*median ----------
    # dts: 47x60 s, one exactly 300 (=5*60, NOT a gap), one 301 (gap idx 21)
    dts = np.full(49, 60.0)
    dts[10] = 300.0
    dts[20] = 301.0
    t = 100.0 + np.concatenate([[0.0], np.cumsum(dts)])
    r = qc3_check_stream(t)
    check("F gap: boundary 300s not flagged, 301s is", r.gap_idx, [21])
    check("F gap: gap_max_s", r.gap_max_s, 301.0)
    check("F gap: gap_total_s", r.gap_total_s, 301.0)

    # --- G. degenerate single-sample stream --------------------------------
    r = qc3_check_stream(np.array([100.0]), energy_cum=np.array([5.0]),
                         power_w=np.array([1.0]), device_p_max=100.0)
    check("G degenerate: n_samples", r.n_samples, 1)
    check("G degenerate: has_defects", r.has_defects, False)
    check("G degenerate: median_dt_s is NaN", r.median_dt_s, float("nan"))

    return failures


if __name__ == "__main__":
    import sys
    import time

    DUMP_PATH = (
        "/opt/data/hpc/adastra/"
        "sacct_dump-UTC-02feb-26may-2026.MixGENOA+MI250.dump"
    )

    print("=" * 78)
    print("REDI HPC power-data QC (QC1/QC2/QC3) — verification report")
    print("=" * 78)

    # --- [0] Synthetic known-answer tests --------------------------------
    print("\n[0] Synthetic known-answer tests")
    print("  QC1 (envelope idle=400 W / max=2400 W, eps=0.05):")
    qc1_failures = _run_synthetic_qc1_tests()
    print("  QC2 (SyntheticPowerSource n1/n2 @ 500 W, rel_tol=0.10):")
    qc2_failures = _run_synthetic_qc2_tests()
    print("  QC3 (stream checks: gap_factor=5, stuck_min_run=30, eps=0.05):")
    qc3_failures = _run_synthetic_qc3_tests()
    total_failures = qc1_failures + qc2_failures + qc3_failures
    if total_failures:
        print(f"\n    {total_failures} synthetic test case(s) FAILED — aborting.")
        sys.exit(1)
    print("\n    All synthetic QC1/QC2/QC3 known-answer cases PASSED.")

    # --- [1] QC1 on the real Adastra sacct dump --------------------------
    print(f"\n[1] QC1 on real sacct dump: {DUMP_PATH}")
    print("    " + "!" * 70)
    print("    ! CAVEAT: envelopes are PROVISIONAL (derived from RAPS")
    print("    ! adastraMI250.yaml, not CINES-confirmed), and cover the mi250")
    print("    ! partitions ONLY. All genoa / genoa-shared jobs are verdict")
    print("    ! 'missing_envelope' pending the CINES envelope confirmation")
    print("    ! (D8 request). Do not quote these numbers without that caveat.")
    print("    " + "!" * 70)

    from redi.domains.hpc.sacct_loader import parse_sacct_dump

    t0 = time.time()
    try:
        jobs = parse_sacct_dump(DUMP_PATH)
    except FileNotFoundError:
        print(f"    Real dump not found at {DUMP_PATH}; skipping real-data "
              f"report (synthetic tests all passed).")
        sys.exit(0)
    t_parse = time.time() - t0

    t0 = time.time()
    qc = qc1_envelope_check(jobs, PROVISIONAL_ADASTRA_ENVELOPES, epsilon=0.05)
    t_qc = time.time() - t0
    print(f"\n    Parsed {len(jobs):,} jobs in {t_parse:.2f}s; "
          f"QC1 ran in {t_qc:.3f}s (epsilon=0.05)")
    for p, e in PROVISIONAL_ADASTRA_ENVELOPES.items():
        print(f"    envelope[{p}]: idle={e.idle_w:.2f} W, max={e.max_w:.2f} W, "
              f"shared={e.shared}")
        print(f"      source: {e.source}")

    # Overall verdict counts
    print("\n[2] QC1 verdict counts — overall")
    counts = qc["qc1_verdict"].value_counts()
    for verdict in QC1_VERDICTS:
        n = int(counts.get(verdict, 0))
        if n or verdict != "not_computable":
            print(f"    {verdict:<18s} {n:>9,}  ({100.0 * n / len(qc):6.2f}%)")

    # Per-partition verdict counts
    print("\n[3] QC1 verdict counts — per partition (count / % of partition)")
    tab = pd.crosstab(qc["partition"], qc["qc1_verdict"])
    for partition in tab.index:
        row = tab.loc[partition]
        total = int(row.sum())
        print(f"    {partition} (n={total:,}):")
        for verdict in QC1_VERDICTS:
            n = int(row.get(verdict, 0))
            if n:
                print(f"        {verdict:<18s} {n:>9,}  ({100.0 * n / total:6.2f}%)")

    # Envelope-failure fraction among energy-bearing mi250 jobs — the first
    # quantitative check of Gerard's ">50% incorrect" claim.
    print("\n[4] Envelope-failure fraction among ENERGY-BEARING mi250 jobs")
    print("    (provisional-envelope-based check of Gerard's '>50%' claim;")
    print("    energy-bearing = energy present, not null, not u64-wraparound)")
    for parts in (["mi250"], ["mi250-shared"], ["mi250", "mi250-shared"]):
        sub = qc[qc["partition"].isin(parts)]
        verdicts = sub["qc1_verdict"].astype(str)
        bearing = ~verdicts.isin(["missing_energy", "wraparound"])
        n_bearing = int(bearing.sum())
        vb = verdicts[bearing]
        n_fail = int(vb.isin(["fail_high", "fail_low"]).sum())
        n_zero = int((vb == "zero_energy").sum())
        n_below_shared = int((vb == "below_idle_shared").sum())
        n_zero_dur = int((vb == "zero_duration").sum())
        label = "+".join(parts)
        print(f"    {label} (energy-bearing n={n_bearing:,}):")
        print(f"        hard envelope failures (fail_high+fail_low): "
              f"{n_fail:,}  ({100.0 * n_fail / max(n_bearing, 1):.2f}%)")
        print(f"        zero_energy: {n_zero:,} ({100.0 * n_zero / max(n_bearing, 1):.2f}%)   "
              f"below_idle_shared (informational): {n_below_shared:,} "
              f"({100.0 * n_below_shared / max(n_bearing, 1):.2f}%)   "
              f"zero_duration: {n_zero_dur:,}")

    # Ten most extreme fail_high jobs
    print("\n[5] Ten most extreme fail_high jobs (by qc1_u)")
    fh = qc[qc["qc1_verdict"] == "fail_high"].sort_values("qc1_u", ascending=False)
    if fh.empty:
        print("    (none)")
    else:
        cols = ["job_id", "state", "partition", "elapsed_seconds",
                "nodes_required", "qc1_avg_power_w", "qc1_u"]
        top = fh.head(10)[cols].copy()
        top["qc1_avg_power_w"] = top["qc1_avg_power_w"].map(lambda x: f"{x:,.0f}")
        top["qc1_u"] = top["qc1_u"].map(lambda x: f"{x:,.2f}")
        print(top.to_string(index=False))

    # Distribution of u among pass jobs, per partition
    print("\n[6] qc1_u distribution among 'pass' jobs, per partition")
    passed = qc[qc["qc1_verdict"] == "pass"]
    for partition, sub in passed.groupby("partition", observed=True):
        u = sub["qc1_u"].dropna()
        if u.empty:
            continue
        print(f"    {partition}: n={len(u):,}  min={u.min():+.3f}  "
              f"p25={u.quantile(0.25):+.3f}  median={u.median():+.3f}  "
              f"p75={u.quantile(0.75):+.3f}  max={u.max():+.3f}")
    print("    (genoa/genoa-shared absent by construction: no envelope yet -> "
          "no pass verdicts)")

    print("\n" + "=" * 78)
    print("Reminder: mi250 numbers above use PROVISIONAL RAPS-derived envelopes;")
    print("genoa partitions are entirely unevaluated (missing_envelope) pending")
    print("CINES-confirmed idle/max values (plan §5, D8).")
    print("=" * 78)
