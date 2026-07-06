"""
REDI power-data-quality detectors for HPC job accounting (QC1 / QC2).

This module implements W4 from notes/uc-demonstration-plan2.md §3 ("Automated
power-data QC: detect + repair"): the *detection* half, restricted to the two
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

Entry points:
  - `qc1_envelope_check(df, envelopes, epsilon)` — pure function.
  - `qc2_cross_source(df, source, rel_tol)` — pure function.
  - `PowerEnvelopeQC` — a PREPROCESS-stage `PipelineStep` wrapping QC1.
    NOT registered in any get_*_steps() list — wiring is a separate change,
    same policy as `LoadAdastraSacctDump` in hpc_sacct_loader.py.
  - `python -m redi.domains.hpc_power_qc` — synthetic known-answer tests
    (exit nonzero on any failure) followed by a QC1 verification report on
    the real Adastra sacct dump.

Input contract: the DataFrame produced by
`redi.domains.hpc_sacct_loader.parse_sacct_dump` (columns: partition,
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
    from ..pipeline import PipelineStep, PipelineStage, PipelineContext

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
# 5. Self-tests (synthetic known-answer) + real-data verification report
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


if __name__ == "__main__":
    import sys
    import time

    DUMP_PATH = (
        "/opt/data/hpc/adastra/"
        "sacct_dump-UTC-02feb-26may-2026.MixGENOA+MI250.dump"
    )

    print("=" * 78)
    print("REDI HPC power-data QC (QC1/QC2) — verification report")
    print("=" * 78)

    # --- [0] Synthetic known-answer tests --------------------------------
    print("\n[0] Synthetic known-answer tests")
    print("  QC1 (envelope idle=400 W / max=2400 W, eps=0.05):")
    qc1_failures = _run_synthetic_qc1_tests()
    print("  QC2 (SyntheticPowerSource n1/n2 @ 500 W, rel_tol=0.10):")
    qc2_failures = _run_synthetic_qc2_tests()
    total_failures = qc1_failures + qc2_failures
    if total_failures:
        print(f"\n    {total_failures} synthetic test case(s) FAILED — aborting.")
        sys.exit(1)
    print("\n    All synthetic QC1/QC2 known-answer cases PASSED.")

    # --- [1] QC1 on the real Adastra sacct dump --------------------------
    print(f"\n[1] QC1 on real sacct dump: {DUMP_PATH}")
    print("    " + "!" * 70)
    print("    ! CAVEAT: envelopes are PROVISIONAL (derived from RAPS")
    print("    ! adastraMI250.yaml, not CINES-confirmed), and cover the mi250")
    print("    ! partitions ONLY. All genoa / genoa-shared jobs are verdict")
    print("    ! 'missing_envelope' pending the CINES envelope confirmation")
    print("    ! (D8 request). Do not quote these numbers without that caveat.")
    print("    " + "!" * 70)

    from redi.domains.hpc_sacct_loader import parse_sacct_dump

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
