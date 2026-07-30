"""
Corrupt-and-recover V&V suite for the REDI power-QC detectors (QC1).

This module implements the *detection* half of the plan's §3 "V&V of the QC
system itself" (notes/uc-demonstration-plan2.md): take jobs that pass all
available QC (verified-good), inject synthetic corruption matched to the
observed taxonomy, and measure detector precision/recall per corruption type.
The repair-error half (R1 exact recompute / R2 imputation MAE) needs the
pm_counters ingester (W3, blocked on CINES schema confirmation, D8) and is
deliberately absent here.

Design:

  - **Clean substrate** (`build_clean_substrate`): real jobs from the Adastra
    sacct dump that get QC1 verdict `pass` on the exclusive `mi250` partition
    (both envelope bounds enforceable there; on `*-shared` partitions the
    below-idle bound is informational only). A seeded deterministic subsample
    of those is the experiment population — so every injected corruption is
    applied to a job KNOWN to be envelope-clean beforehand, and ground truth
    is exact by construction.

  - **Corruption injectors** (`CORRUPTION_MODES`): one pure function per
    observed/plausible corruption mode, each `(df, rng, fraction) ->
    (df_corrupted, ground_truth_mask)`. Modes match what the corpus has
    actually shown (u64 counter wraparound, blank/zero ConsumedEnergyRaw)
    plus plausible unit/metadata errors (J<->mJ scaling, double-counted
    nodes, zeroed timestamps). The E3.1 corpus build reuses these injectors
    for the paper's QC table.

  - **Experiment protocol** (`run_vv_suite`): for each mode independently,
    corrupt a fraction (default 10%) of the population, run QC1, and score
    recall (corrupted jobs whose verdict is not `pass`), verdict attribution
    (which taxonomy category caught them), and the false-positive rate on
    the uncorrupted remainder (expected ~0 by construction — the substrate
    passed QC1 before injection). Repeated over 3 seeds; report mean ± std.

  - **Baseline** ("Gerard's rule as quoted"): the same protocol under the
    naive reading of the CINES rule — epsilon = 0 (no tolerance band) and
    node sharing ignored (below-idle is a failure on every partition).
    Two comparisons: (a) recall difference on injected corruption, and
    (b) the false-positive cost on the FULL real `mi250-shared` partition —
    how many real jobs the naive rule flags that the shared-aware rule does
    not. That over-flagging is what the refinement buys.

  - **Honest limits** (`qc1_detection_boundary`): QC1 sees only the per-node
    average power, so any corruption that keeps ū inside [0−ε, 1+ε] is
    UNDETECTABLE by QC1 in principle — small multiplicative scalings
    (`partial_scale`), node undercounts on low-power jobs, and compensating
    errors (energy and elapsed scaled together). This function quantifies
    the detection boundary on the real substrate (which scale factors keep
    a job in-envelope), so the recall numbers are calibrated rather than
    oversold. QC2 (cross-source, pm_counters) is the designed catch for
    that class, pending W3/D8.

Entry points:
  - `build_clean_substrate(jobs, envelopes, epsilon, ...)` — pure function.
  - `CORRUPTION_MODES` — ordered {mode name -> injector} registry.
  - `run_vv_suite(substrate, envelopes, ...)` — full seeded protocol,
    returns nested per-mode/per-seed scores for ours + naive rule.
  - `qc1_detection_boundary(substrate, envelope, epsilon)` — limits math.
  - `python -m redi.domains.hpc.qc_validation` — runs the whole suite
    against the real Adastra dump and prints a structured report; exits
    nonzero if any structural check fails.

Input contract: the DataFrame produced by
`redi.domains.hpc.sacct_loader.parse_sacct_dump` (columns: partition,
energy_joules [Float64], elapsed_seconds [Int64], nodes_required [Int64],
...), as consumed by `redi.domains.hpc.power_qc.qc1_envelope_check`.
"""

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .power_qc import (
    PROVISIONAL_ADASTRA_ENVELOPES,
    QC1_VERDICTS,
    PartitionEnvelope,
    qc1_envelope_check,
)

logger = logging.getLogger(__name__)

# Defaults for the experiment protocol (plan §3 V&V).
DEFAULT_EPSILON = 0.05          # ours: QC1's tolerance band
DEFAULT_FRACTION = 0.10         # fraction of the population corrupted per run
DEFAULT_SEEDS = (0, 1, 2)       # 3 seeds -> mean +/- std
DEFAULT_SUBSTRATE_SIZE = 5000   # deterministic subsample of the ~25k pass jobs
SUBSTRATE_PARTITION = "mi250"   # exclusive partition: both bounds enforceable
SUBSTRATE_SEED = 42             # substrate subsample seed (fixed, not swept)

# Verdicts that do NOT count as "flagged" when scoring recall / FP. On the
# exclusive substrate `below_idle_shared` cannot occur, but keep it explicit
# so the scorer stays correct if pointed at a shared-partition population.
NON_FLAGGING_VERDICTS = ("pass", "below_idle_shared")

# Where each mode is EXPECTED to land in the QC1 verdict taxonomy (used for
# the attribution commentary; the attribution matrix reports what actually
# happened, including escapes to `pass`).
EXPECTED_VERDICT = {
    "unit_scale_high": "fail_high",
    "unit_scale_low": "fail_low",
    "partial_scale": "fail_high",
    "zeroed": "zero_energy",
    "blanked": "missing_energy",
    "u64_wrap": "wraparound",
    "elapsed_zero": "zero_duration",
    "undercount_nodes": "fail_high",
}

# Modes where 100% recall is EXPECTED by construction on an envelope-clean
# substrate (see each injector's docstring); the structural checks in
# __main__ hard-fail if any of these drop below 1.0. partial_scale and
# undercount_nodes are excluded: their recall is bounded by the QC1
# detection boundary (see qc1_detection_boundary) and partial recall there
# is the honest, expected result.
DETERMINISTIC_MODES = (
    "unit_scale_high", "unit_scale_low", "zeroed", "blanked",
    "u64_wrap", "elapsed_zero",
)


# ---------------------------------------------------------------------------
# 1. Clean substrate
# ---------------------------------------------------------------------------
def build_clean_substrate(
    jobs: pd.DataFrame,
    envelopes: Dict[str, PartitionEnvelope] = PROVISIONAL_ADASTRA_ENVELOPES,
    epsilon: float = DEFAULT_EPSILON,
    partition: str = SUBSTRATE_PARTITION,
    n: int = DEFAULT_SUBSTRATE_SIZE,
    seed: int = SUBSTRATE_SEED,
) -> pd.DataFrame:
    """Verified-good experiment population: QC1-pass jobs, subsampled.

    Runs QC1 over `jobs`, keeps rows with verdict `pass` on `partition`
    (default `mi250` — exclusive, so both envelope bounds are enforceable
    and `pass` genuinely means in-envelope), then draws a deterministic
    seeded subsample of `n` rows as the experiment population.

    The returned frame has a fresh RangeIndex (positions == labels, which
    the injectors and scorer rely on) and the qc1_* columns dropped —
    qc1_envelope_check re-adds them on every scored run.

    Args:
        jobs: sacct-loader DataFrame (parse_sacct_dump output).
        envelopes: partition -> PartitionEnvelope for the screening QC1 run.
        epsilon: tolerance band for the screening run (same value the suite
            uses for "our" rule, so `pass` here means pass under our rule).
        partition: which partition to draw the substrate from.
        n: subsample size (capped at the available pass population).
        seed: subsample seed — fixed by default so the population is the
            SAME across the suite's corruption seeds.

    Returns:
        DataFrame of len <= n verified-good jobs, RangeIndex.
    """
    qc = qc1_envelope_check(jobs, envelopes, epsilon=epsilon)
    clean = qc[
        (qc["partition"] == partition)
        & (qc["qc1_verdict"].astype(str) == "pass")
    ]
    logger.info(
        f"build_clean_substrate: {len(clean):,} QC1-pass jobs on "
        f"'{partition}' out of {len(jobs):,} total"
    )
    take = min(n, len(clean))
    sample = clean.sample(n=take, random_state=seed)
    drop = [c for c in sample.columns if c.startswith("qc1_")]
    return sample.drop(columns=drop).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Corruption injectors
#
# Contract shared by every injector:
#   inject_<mode>(df, rng, fraction) -> (df_corrupted, ground_truth_mask)
#     - df is NOT mutated; a copy is returned.
#     - df must have a RangeIndex (build_clean_substrate guarantees this).
#     - ground_truth_mask is a bool ndarray of len(df); True = corrupted.
#     - rng is a np.random.Generator: row selection and any random
#       parameters draw from it, so (population, rng state) fully
#       determines the outcome.
# ---------------------------------------------------------------------------
def _select_rows(
    n_rows: int,
    rng: np.random.Generator,
    fraction: float,
    eligible: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pick ~fraction*n_rows distinct row positions, uniformly at random.

    Args:
        n_rows: population size.
        rng: seeded generator.
        fraction: target corrupted fraction of the FULL population.
        eligible: optional bool mask restricting which rows may be picked
            (for modes that are only physically meaningful on some rows,
            e.g. undercount_nodes needs nodes_required >= 2). If fewer
            eligible rows exist than the target count, all eligible rows
            are taken (the caller's mask then reflects the smaller count).

    Returns:
        Sorted int ndarray of selected row positions.
    """
    target = max(1, int(round(fraction * n_rows)))
    if eligible is None:
        pool = np.arange(n_rows)
    else:
        pool = np.flatnonzero(eligible)
    take = min(target, pool.size)
    if take < target:
        logger.warning(
            f"_select_rows: only {pool.size} eligible rows for a target of "
            f"{target}; corrupting {take}"
        )
    return np.sort(rng.choice(pool, size=take, replace=False))


def _mask(n_rows: int, positions: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    mask[positions] = True
    return mask


def inject_unit_scale_high(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = DEFAULT_FRACTION
) -> Tuple[pd.DataFrame, np.ndarray]:
    """energy x1000 — a J<->mJ-style unit error (reported in mJ, read as J).

    Expected verdict `fail_high`; recall should be 1.0 on this substrate:
    a pass job implies per-node power >= idle − ε·span (~472 W here), so
    x1000 lands orders of magnitude above any node envelope, while staying
    far below the 2^63 wraparound sentinel (max substrate energy ~1e10 J).
    """
    out = df.copy()
    pos = _select_rows(len(out), rng, fraction)
    out.loc[pos, "energy_joules"] = out.loc[pos, "energy_joules"] * 1000.0
    return out, _mask(len(out), pos)


def inject_unit_scale_low(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = DEFAULT_FRACTION
) -> Tuple[pd.DataFrame, np.ndarray]:
    """energy /1000 — the opposite unit error (reported in kJ, read as J).

    Expected verdict `fail_low` (exclusive partition): a pass job's per-node
    power drops to < 2.4 W, far below idle, but stays > 0 so it does not
    collide with the `zero_energy` category. Recall should be 1.0.
    """
    out = df.copy()
    pos = _select_rows(len(out), rng, fraction)
    out.loc[pos, "energy_joules"] = out.loc[pos, "energy_joules"] / 1000.0
    return out, _mask(len(out), pos)


def inject_partial_scale(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = DEFAULT_FRACTION
) -> Tuple[pd.DataFrame, np.ndarray]:
    """energy x U(2, 10) per row — a subtler miscount (e.g. energy summed
    over double-counted node entries, or a partially duplicated accounting
    window).

    Expected verdict `fail_high` WHEN caught — but a factor s only trips
    QC1 if s x avg_power exceeds max + ε·span, so low-power jobs escape
    even at large factors. Partial recall here is the honest, expected
    result; qc1_detection_boundary quantifies it analytically and QC2
    (cross-source) is the designed catch (plan §3).
    """
    out = df.copy()
    pos = _select_rows(len(out), rng, fraction)
    factors = rng.uniform(2.0, 10.0, size=pos.size)
    out.loc[pos, "energy_joules"] = out.loc[pos, "energy_joules"] * factors
    return out, _mask(len(out), pos)


def inject_zeroed(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = DEFAULT_FRACTION
) -> Tuple[pd.DataFrame, np.ndarray]:
    """energy set to exactly 0 — observed in the corpus (e.g. NODE_FAIL
    jobs report ConsumedEnergyRaw=0). Expected verdict `zero_energy`;
    recall 1.0 (the zero check precedes envelope math)."""
    out = df.copy()
    pos = _select_rows(len(out), rng, fraction)
    out.loc[pos, "energy_joules"] = 0.0
    return out, _mask(len(out), pos)


def inject_blanked(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = DEFAULT_FRACTION
) -> Tuple[pd.DataFrame, np.ndarray]:
    """energy set to null — observed in the corpus (blank ConsumedEnergyRaw
    fields; ~21k such jobs on mi250 alone). Expected verdict
    `missing_energy`; recall 1.0."""
    out = df.copy()
    pos = _select_rows(len(out), rng, fraction)
    out.loc[pos, "energy_joules"] = pd.NA
    return out, _mask(len(out), pos)


def inject_u64_wrap(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = DEFAULT_FRACTION
) -> Tuple[pd.DataFrame, np.ndarray]:
    """energy set to 2^64 − X, X ~ U(1e6, 1e10) — the OBSERVED wrapped
    unsigned-64-bit counter signature (e.g. 18446744061398995278 in the
    real dump: a counter that went slightly negative and wrapped).

    Expected verdict `wraparound` (energy > 2^63); recall 1.0. X stays
    well above float64's ~2048 J ulp near 2^64, so distinct values survive
    the Float64 representation.
    """
    out = df.copy()
    pos = _select_rows(len(out), rng, fraction)
    x = rng.uniform(1.0e6, 1.0e10, size=pos.size)
    out.loc[pos, "energy_joules"] = float(2**64) - x
    return out, _mask(len(out), pos)


def inject_elapsed_zero(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = DEFAULT_FRACTION
) -> Tuple[pd.DataFrame, np.ndarray]:
    """elapsed_seconds set to 0 — timestamp corruption (start==end, e.g. a
    clock/accounting glitch), with energy left intact. Expected verdict
    `zero_duration` (the substrate's energies are positive and non-wrapped,
    so the higher-precedence energy checks do not fire); recall 1.0."""
    out = df.copy()
    pos = _select_rows(len(out), rng, fraction)
    out.loc[pos, "elapsed_seconds"] = 0
    return out, _mask(len(out), pos)


def inject_undercount_nodes(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = DEFAULT_FRACTION
) -> Tuple[pd.DataFrame, np.ndarray]:
    """nodes_required halved (integer div), energy unchanged — metadata
    corruption that inflates the implied per-node power by ~2x.

    Only rows with nodes_required >= 2 are eligible (a 1-node job cannot
    be halved; halving to 0 would surface as `not_computable`, testing the
    wrong thing). If fewer than fraction*n rows are eligible, all eligible
    rows are corrupted and the ground-truth mask reflects that.

    Expected verdict `fail_high` WHEN caught — doubling only trips QC1 if
    2 x avg_power > max + ε·span, i.e. the job was drawing more than about
    half the envelope; low-utilization jobs escape. Partial recall is the
    honest, expected result (see qc1_detection_boundary).
    """
    out = df.copy()
    nodes = out["nodes_required"]
    eligible = (nodes.notna() & (nodes >= 2)).to_numpy(dtype=bool)
    pos = _select_rows(len(out), rng, fraction, eligible=eligible)
    out.loc[pos, "nodes_required"] = out.loc[pos, "nodes_required"] // 2
    return out, _mask(len(out), pos)


# Ordered registry: mode name -> injector. Order is the report order and
# feeds the per-mode rng stream (seeded as [seed, mode_index]).
CORRUPTION_MODES: Dict[str, Callable] = {
    "unit_scale_high": inject_unit_scale_high,
    "unit_scale_low": inject_unit_scale_low,
    "partial_scale": inject_partial_scale,
    "zeroed": inject_zeroed,
    "blanked": inject_blanked,
    "u64_wrap": inject_u64_wrap,
    "elapsed_zero": inject_elapsed_zero,
    "undercount_nodes": inject_undercount_nodes,
}


# ---------------------------------------------------------------------------
# 3. Scoring + experiment protocol
# ---------------------------------------------------------------------------
def naive_envelopes(
    envelopes: Dict[str, PartitionEnvelope]
) -> Dict[str, PartitionEnvelope]:
    """The baseline's envelope config: Gerard's rule *as quoted*.

    Same idle/max numbers, but node sharing ignored — every partition is
    treated as exclusive, so below-idle is a hard failure everywhere.
    Combined with epsilon=0 at the call site, this is the naive reading of
    "energy is incorrect when the average power derived from it is lower
    or higher than the idle or maximum power of the nodes used by the job".
    """
    return {
        p: PartitionEnvelope(
            idle_w=e.idle_w, max_w=e.max_w, shared=False,
            source=e.source + " [naive baseline: sharing ignored]",
        )
        for p, e in envelopes.items()
    }


def score_qc1_run(
    qc: pd.DataFrame,
    truth_mask: np.ndarray,
    non_flagging: Sequence[str] = NON_FLAGGING_VERDICTS,
) -> Dict:
    """Score one QC1 run against the known corruption ground truth.

    Args:
        qc: qc1_envelope_check output over the (partially corrupted)
            population; must be positionally aligned with truth_mask.
        truth_mask: bool ndarray, True where the row was corrupted.
        non_flagging: verdicts that count as NOT flagged.

    Returns:
        dict with:
          - n / n_corrupted / n_clean: population sizes;
          - n_caught, recall: corrupted rows flagged (verdict outside
            `non_flagging`) and their fraction;
          - n_false_pos, fp_rate: clean rows flagged, and their fraction
            of the clean rows (expected ~0: the substrate passed QC1
            before injection, so under "our" rule any FP is a bug);
          - verdicts_corrupted: verdict -> count over corrupted rows only
            (the attribution row; `pass` entries are the escapes).
    """
    verdict = qc["qc1_verdict"].astype(str).to_numpy()
    flagged = ~np.isin(verdict, list(non_flagging))
    n_corrupted = int(truth_mask.sum())
    n_clean = int((~truth_mask).sum())
    n_caught = int(flagged[truth_mask].sum())
    n_false_pos = int(flagged[~truth_mask].sum())
    counts = pd.Series(verdict[truth_mask]).value_counts()
    return {
        "n": len(qc),
        "n_corrupted": n_corrupted,
        "n_clean": n_clean,
        "n_caught": n_caught,
        "recall": n_caught / n_corrupted if n_corrupted else float("nan"),
        "n_false_pos": n_false_pos,
        "fp_rate": n_false_pos / n_clean if n_clean else float("nan"),
        "verdicts_corrupted": {k: int(v) for k, v in counts.items()},
    }


def run_vv_suite(
    substrate: pd.DataFrame,
    envelopes: Dict[str, PartitionEnvelope] = PROVISIONAL_ADASTRA_ENVELOPES,
    epsilon: float = DEFAULT_EPSILON,
    fraction: float = DEFAULT_FRACTION,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> Dict[str, List[Dict]]:
    """Full corrupt-and-recover protocol over every mode and seed.

    For each (mode, seed): corrupt `fraction` of the substrate with a
    generator seeded as [seed, mode_index] (decorrelates modes within a
    seed while keeping everything reproducible), run QC1 once under OUR
    rule (given envelopes, given epsilon) and once under the NAIVE
    baseline (same idle/max, sharing ignored, epsilon=0) on the SAME
    corrupted frame, and score both against the injection ground truth.

    Returns:
        {mode: [ {"seed": s, "ours": score, "naive": score}, ... ]} with
        scores as returned by score_qc1_run.
    """
    naive = naive_envelopes(envelopes)
    results: Dict[str, List[Dict]] = {}
    for mode_idx, (mode, injector) in enumerate(CORRUPTION_MODES.items()):
        per_seed = []
        for seed in seeds:
            rng = np.random.default_rng([seed, mode_idx])
            corrupted, mask = injector(substrate, rng, fraction)
            ours = score_qc1_run(
                qc1_envelope_check(corrupted, envelopes, epsilon=epsilon),
                mask,
            )
            base = score_qc1_run(
                qc1_envelope_check(corrupted, naive, epsilon=0.0),
                mask,
            )
            per_seed.append({"seed": seed, "ours": ours, "naive": base})
        results[mode] = per_seed
        mean_recall = float(np.mean([r["ours"]["recall"] for r in per_seed]))
        logger.info(
            f"run_vv_suite: {mode}: recall(ours) mean={mean_recall:.3f} "
            f"over {len(per_seed)} seed(s)"
        )
    return results


# ---------------------------------------------------------------------------
# 4. Honest limits: the QC1 detection boundary
# ---------------------------------------------------------------------------
def qc1_detection_boundary(
    substrate: pd.DataFrame,
    envelope: PartitionEnvelope,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict:
    """Quantify what multiplicative corruption QC1 CANNOT see, on this
    substrate.

    QC1 accepts a job iff its per-node average power P lies in
    [idle − ε·span, max + ε·span] (span = max − idle). A multiplicative
    energy error s therefore escapes iff s·P stays inside that band —
    per job the undetectable window is s ∈ [band_lo/P, band_hi/P].

    Returns a dict with:
      - band_lo_w / band_hi_w: the accepted power band, W;
      - median_power_w and the median job's undetectable scale window
        (scale_lo, scale_hi);
      - escape_frac: {factor: fraction of substrate jobs for which
        energy x factor stays in-band} for representative factors;
      - partial_scale_predicted_recall: analytic expected recall of the
        `partial_scale` injector (factor ~ U(2,10)): per job,
        P(detect) = P(s > band_hi/P) clipped to [0,1], averaged;
      - undercount_predicted_recall: analytic expected recall of
        `undercount_nodes` on its eligible rows (nodes >= 2): halving N
        doubles P, caught iff 2P > band_hi;
      - undercount_eligible_frac: fraction of substrate rows eligible.
    """
    energy = substrate["energy_joules"].astype("Float64").to_numpy(
        dtype="float64", na_value=np.nan)
    elapsed = substrate["elapsed_seconds"].astype("Float64").to_numpy(
        dtype="float64", na_value=np.nan)
    nodes = substrate["nodes_required"].astype("Float64").to_numpy(
        dtype="float64", na_value=np.nan)
    power = energy / (elapsed * nodes)

    span = envelope.max_w - envelope.idle_w
    band_lo = envelope.idle_w - epsilon * span
    band_hi = envelope.max_w + epsilon * span

    median_p = float(np.nanmedian(power))
    escape_frac = {}
    for s in (0.1, 0.5, 2.0, 3.0, 5.0, 10.0):
        inside = (s * power >= band_lo) & (s * power <= band_hi)
        escape_frac[s] = float(np.nanmean(inside))

    # partial_scale, s ~ U(2, 10): detected iff s > band_hi / P (the low
    # edge is unreachable for s >= 2 on an in-band job: 2*P >= 2*band_lo
    # > band_lo). Per job: P(detect) = (10 - max(2, band_hi/P)) / 8.
    s_min_detect = np.clip(band_hi / power, 2.0, 10.0)
    partial_pred = float(np.nanmean((10.0 - s_min_detect) / 8.0))

    eligible = nodes >= 2
    if eligible.any():
        undercount_pred = float(np.nanmean(2.0 * power[eligible] > band_hi))
    else:
        undercount_pred = float("nan")

    return {
        "band_lo_w": band_lo,
        "band_hi_w": band_hi,
        "median_power_w": median_p,
        "median_scale_window": (band_lo / median_p, band_hi / median_p),
        "escape_frac": escape_frac,
        "partial_scale_predicted_recall": partial_pred,
        "undercount_predicted_recall": undercount_pred,
        "undercount_eligible_frac": float(np.mean(eligible)),
    }


# ---------------------------------------------------------------------------
# 5. Injector contract self-check
# ---------------------------------------------------------------------------
def _verify_injector_contracts(substrate: pd.DataFrame) -> List[str]:
    """Check every injector's contract on the real substrate. Returns a
    list of violation strings (empty = all good).

    Verified per mode (seed 0): (a) the original frame is not mutated;
    (b) rows OUTSIDE the ground-truth mask are byte-identical on the
    columns QC1 consumes; (c) rows INSIDE the mask differ from the
    original on at least one QC1 input column; (d) the corrupted count is
    positive and <= the requested fraction of the population.
    """
    problems: List[str] = []
    qc_cols = ["energy_joules", "elapsed_seconds", "nodes_required"]
    before = substrate[qc_cols].copy()
    n = len(substrate)
    target = max(1, int(round(DEFAULT_FRACTION * n)))
    for mode_idx, (mode, injector) in enumerate(CORRUPTION_MODES.items()):
        rng = np.random.default_rng([0, mode_idx])
        corrupted, mask = injector(substrate, rng, DEFAULT_FRACTION)
        if not substrate[qc_cols].equals(before):
            problems.append(f"{mode}: mutated the input frame")
        if len(corrupted) != n or mask.shape != (n,):
            problems.append(f"{mode}: shape mismatch")
            continue
        n_corr = int(mask.sum())
        if not (0 < n_corr <= target):
            problems.append(
                f"{mode}: corrupted {n_corr} rows (want 1..{target})")
        same = corrupted[qc_cols].reset_index(drop=True).eq(
            before.reset_index(drop=True))
        # Nullable eq() returns NA where either side is NA; treat NA==NA
        # as equal by also comparing null placement.
        null_match = corrupted[qc_cols].isna().reset_index(drop=True).eq(
            before.isna().reset_index(drop=True))
        row_equal = (same.fillna(True).all(axis=1)
                     & null_match.all(axis=1)).to_numpy()
        if not row_equal[~mask].all():
            problems.append(f"{mode}: touched rows outside the truth mask")
        if row_equal[mask].any():
            problems.append(f"{mode}: mask marks untouched rows")
    return problems


# ---------------------------------------------------------------------------
# 6. Report helpers
# ---------------------------------------------------------------------------
def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype="float64")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _fmt_pct(mean: float, std: float) -> str:
    return f"{100.0 * mean:6.2f}% ± {100.0 * std:5.2f}"


if __name__ == "__main__":
    import sys
    import time

    DUMP_PATH = (
        "/opt/data/hpc/adastra/"
        "sacct_dump-UTC-02feb-26may-2026.MixGENOA+MI250.dump"
    )

    t_start = time.time()
    structural_failures: List[str] = []

    print("=" * 78)
    print("REDI power-QC corrupt-and-recover V&V suite (plan §3, QC1)")
    print("=" * 78)
    print(f"  protocol: fraction={DEFAULT_FRACTION:.0%} corrupted per mode, "
          f"seeds={list(DEFAULT_SEEDS)}, ours eps={DEFAULT_EPSILON}, "
          f"naive baseline eps=0 + sharing ignored")
    print("  CAVEAT: envelopes are PROVISIONAL (RAPS-derived, not CINES-")
    print("  confirmed, D8 pending); recall/FP numbers are conditional on them.")

    # --- [0] Load real dump + build the clean substrate --------------------
    from redi.domains.hpc.sacct_loader import parse_sacct_dump

    print(f"\n[0] Clean substrate from real dump: {DUMP_PATH}")
    t0 = time.time()
    try:
        jobs = parse_sacct_dump(DUMP_PATH)
    except FileNotFoundError:
        print(f"    Real dump not found at {DUMP_PATH}; cannot run the "
              f"corrupt-and-recover suite without the real substrate.")
        sys.exit(1)
    t_parse = time.time() - t0

    t0 = time.time()
    substrate = build_clean_substrate(jobs, PROVISIONAL_ADASTRA_ENVELOPES,
                                      epsilon=DEFAULT_EPSILON)
    t_substrate = time.time() - t0
    print(f"    parsed {len(jobs):,} jobs in {t_parse:.2f}s; substrate = "
          f"{len(substrate):,} QC1-pass '{SUBSTRATE_PARTITION}' jobs "
          f"(subsample seed {SUBSTRATE_SEED}) in {t_substrate:.2f}s")
    if len(substrate) < 1000:
        structural_failures.append(
            f"substrate too small: {len(substrate)} < 1000")

    contract_problems = _verify_injector_contracts(substrate)
    if contract_problems:
        structural_failures.extend(contract_problems)
        for p in contract_problems:
            print(f"    [FAIL] injector contract: {p}")
    else:
        print(f"    injector contract check: all {len(CORRUPTION_MODES)} "
              f"modes touch exactly their masked rows [OK]")

    # --- [1] Run the suite --------------------------------------------------
    t0 = time.time()
    results = run_vv_suite(substrate, PROVISIONAL_ADASTRA_ENVELOPES,
                           epsilon=DEFAULT_EPSILON,
                           fraction=DEFAULT_FRACTION, seeds=DEFAULT_SEEDS)
    t_suite = time.time() - t0

    print(f"\n[1] Per-mode recall, OUR rule (eps={DEFAULT_EPSILON}) — "
          f"{len(DEFAULT_SEEDS)}-seed mean ± std")
    print(f"    {'mode':<18s} {'recall':>16s}   {'n_corr/seed':>11s}   "
          f"expected verdict")
    for mode, per_seed in results.items():
        recalls = [r["ours"]["recall"] for r in per_seed]
        m, s = _mean_std(recalls)
        n_corr = per_seed[0]["ours"]["n_corrupted"]
        print(f"    {mode:<18s} {_fmt_pct(m, s):>16s}   {n_corr:>11,}   "
              f"{EXPECTED_VERDICT[mode]}")
        if mode in DETERMINISTIC_MODES and m < 1.0:
            structural_failures.append(
                f"{mode}: recall {m:.4f} < 1.0 (deterministic mode)")

    # --- [2] Verdict attribution matrix -------------------------------------
    print(f"\n[2] Verdict attribution — corrupted rows only, summed over "
          f"{len(DEFAULT_SEEDS)} seeds ('pass' = escaped)")
    attribution = pd.DataFrame(
        0, index=list(CORRUPTION_MODES), columns=QC1_VERDICTS, dtype=int)
    for mode, per_seed in results.items():
        for r in per_seed:
            for verdict, count in r["ours"]["verdicts_corrupted"].items():
                attribution.loc[mode, verdict] += count
    nonzero_cols = [c for c in attribution.columns
                    if attribution[c].sum() > 0]
    print("    " + attribution[nonzero_cols].to_string().replace(
        "\n", "\n    "))
    for mode in CORRUPTION_MODES:
        caught_cols = [c for c in nonzero_cols
                       if c not in NON_FLAGGING_VERDICTS
                       and attribution.loc[mode, c] > 0]
        stray = [c for c in caught_cols if c != EXPECTED_VERDICT[mode]]
        if stray:
            structural_failures.append(
                f"{mode}: caught under unexpected verdict(s) {stray} "
                f"(expected {EXPECTED_VERDICT[mode]})")

    # --- [3] False positives -------------------------------------------------
    print(f"\n[3] False-positive rate on the UNCORRUPTED remainder "
          f"(~0 by construction for our rule)")
    print(f"    {'mode':<18s} {'FP ours':>16s} {'FP naive (eps=0)':>18s}")
    for mode, per_seed in results.items():
        fp_ours = [r["ours"]["fp_rate"] for r in per_seed]
        fp_naive = [r["naive"]["fp_rate"] for r in per_seed]
        mo, so = _mean_std(fp_ours)
        mn, sn = _mean_std(fp_naive)
        print(f"    {mode:<18s} {_fmt_pct(mo, so):>16s} "
              f"{_fmt_pct(mn, sn):>18s}")
        if mo > 0.01:
            structural_failures.append(
                f"{mode}: our-rule FP rate {mo:.4f} > 1%")

    # --- [4] Gerard baseline comparison --------------------------------------
    print("\n[4] Baseline: Gerard's rule as quoted (eps=0, sharing ignored)")
    print("    (a) recall difference on injected corruption "
          "(naive − ours, percentage points):")
    for mode, per_seed in results.items():
        m_ours, _ = _mean_std([r["ours"]["recall"] for r in per_seed])
        m_naive, _ = _mean_std([r["naive"]["recall"] for r in per_seed])
        delta_pp = 100.0 * (m_naive - m_ours)
        note = "" if abs(delta_pp) < 1e-9 else \
            "  <- eps=0 catches borderline scalings ours tolerates"
        print(f"        {mode:<18s} ours={100 * m_ours:6.2f}%  "
              f"naive={100 * m_naive:6.2f}%  delta={delta_pp:+6.2f}pp{note}")

    print("    (b) false-positive cost on the FULL real mi250-shared "
          "partition:")
    shared = jobs[jobs["partition"] == "mi250-shared"]
    qc_ours = qc1_envelope_check(shared, PROVISIONAL_ADASTRA_ENVELOPES,
                                 epsilon=DEFAULT_EPSILON)
    qc_naive = qc1_envelope_check(
        shared, naive_envelopes(PROVISIONAL_ADASTRA_ENVELOPES), epsilon=0.0)
    env_fail = ("fail_high", "fail_low")
    ours_fail = qc_ours["qc1_verdict"].astype(str).isin(env_fail).to_numpy()
    naive_fail = qc_naive["qc1_verdict"].astype(str).isin(env_fail).to_numpy()
    over = naive_fail & ~ours_fail
    n_shared = len(shared)
    print(f"        mi250-shared jobs: {n_shared:,}")
    print(f"        envelope failures — naive: {int(naive_fail.sum()):,}  "
          f"ours: {int(ours_fail.sum()):,}")
    print(f"        over-flagged by naive only: {int(over.sum()):,} "
          f"({100.0 * over.sum() / max(n_shared, 1):.2f}% of the partition)")
    over_break = qc_ours.loc[over, "qc1_verdict"].astype(str).value_counts()
    for verdict, count in over_break.items():
        why = {
            "below_idle_shared": "legit node-share below idle "
                                 "(sharing-aware rule keeps them)",
            "pass": "inside the eps tolerance band",
        }.get(verdict, "")
        print(f"            our verdict {verdict:<18s} {count:>7,}  {why}")

    # --- [5] Honest limits ----------------------------------------------------
    print("\n[5] Honest limits: the QC1 detection boundary "
          "(what QC1 CANNOT see)")
    boundary = qc1_detection_boundary(
        substrate, PROVISIONAL_ADASTRA_ENVELOPES[SUBSTRATE_PARTITION],
        epsilon=DEFAULT_EPSILON)
    lo_s, hi_s = boundary["median_scale_window"]
    print(f"    accepted per-node power band: "
          f"[{boundary['band_lo_w']:.1f}, {boundary['band_hi_w']:.1f}] W "
          f"(idle/max ± eps*span)")
    print(f"    median substrate job draws {boundary['median_power_w']:.0f} "
          f"W/node -> energy scale factors in [{lo_s:.2f}, {hi_s:.2f}] are "
          f"UNDETECTABLE for it")
    print("    fraction of substrate jobs still in-band after energy x s:")
    for s, frac in boundary["escape_frac"].items():
        print(f"        s = {s:>5.1f}: {100.0 * frac:6.2f}% escape QC1")
    partial_measured, _ = _mean_std(
        [r["ours"]["recall"] for r in results["partial_scale"]])
    under_measured, _ = _mean_std(
        [r["ours"]["recall"] for r in results["undercount_nodes"]])
    print(f"    partial_scale (s~U(2,10)):   predicted recall "
          f"{100.0 * boundary['partial_scale_predicted_recall']:6.2f}%  "
          f"measured {100.0 * partial_measured:6.2f}%")
    print(f"    undercount_nodes (P doubles): predicted recall "
          f"{100.0 * boundary['undercount_predicted_recall']:6.2f}%  "
          f"measured {100.0 * under_measured:6.2f}%  "
          f"(eligible rows: "
          f"{100.0 * boundary['undercount_eligible_frac']:.1f}% of substrate)")
    print("    Also invisible to QC1 in principle: compensating errors "
          "(energy and elapsed")
    print("    scaled together), and per-node imbalance hidden by the "
          "job-level average.")
    print("    QC2 (cross-source vs. pm_counters counters) is the designed "
          "catch for this")
    print("    class — pending the W3 ingester (CINES schema, D8).")
    if abs(partial_measured
           - boundary["partial_scale_predicted_recall"]) > 0.05:
        structural_failures.append(
            f"partial_scale measured recall {partial_measured:.3f} deviates "
            f">5pp from analytic prediction "
            f"{boundary['partial_scale_predicted_recall']:.3f}")
    if abs(under_measured
           - boundary["undercount_predicted_recall"]) > 0.05:
        structural_failures.append(
            f"undercount_nodes measured recall {under_measured:.3f} deviates "
            f">5pp from analytic prediction "
            f"{boundary['undercount_predicted_recall']:.3f}")

    # --- [6] Structural checks + runtime --------------------------------------
    t_total = time.time() - t_start
    print(f"\n[6] Runtime: parse {t_parse:.2f}s, substrate {t_substrate:.2f}s, "
          f"suite ({len(CORRUPTION_MODES)} modes x {len(DEFAULT_SEEDS)} seeds "
          f"x 2 rules) {t_suite:.2f}s, total {t_total:.2f}s")
    if structural_failures:
        print(f"\n    {len(structural_failures)} STRUCTURAL CHECK(S) FAILED:")
        for f in structural_failures:
            print(f"      - {f}")
        print("=" * 78)
        sys.exit(1)
    print("    All structural checks passed: substrate size, injector "
          "contracts,")
    print("    deterministic-mode recall = 100%, our-rule FP <= 1%, verdict")
    print("    attribution as expected, measured recall within 5pp of the")
    print("    analytic detection boundary.")
    print("=" * 78)
    sys.exit(0)
