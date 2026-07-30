"""
Cross-system replay casting: slot-wise nearest-neighbor casting, with
replacement and tiling.

Generalizes what was previously a paper-specific script (fixed to a
Marconi100/PM100 source and a Frontier target) into an arbitrary
--source-system/--target-system pairing, using RAPS's own per-node power
envelope for each system (`job_mapper.load_raps_system_params`) instead of
hand-copied constants, and `swf_loader.parse_swf` for schedule-skeleton
extraction instead of a hand-rolled line parser.

For each real job slot (size, start, duration) in a target-system day, picks
the source-system job whose own (duration, whole-node dynamic intensity) is
nearest to (that slot's duration, the target-system's busy-node intensity),
selecting with replacement (a tail-limited intensity band means reuse is
expected and disclosed, not a bug).

Design note: when the target day comes from a schedule-only trace (no
per-job power telemetry), the match criterion's "profile" is operationalized
as a single target intensity scalar, not a per-job target trace shape. Every
selected job's OWN real trace is replayed untouched (tiled if too short,
never scaled); only which real trace gets used, and how many nodes it is
declared to run on, changes.

Whole-node intensity (not GPU-residual) is used for both ranking and
matching: it is measured directly from the source system's own per-node
power trace, avoiding the residual method's completeness/noise caveats for
this selection step. Downstream replay still uses each job's real
cpu_trace/gpu_trace, unmodified except the source-to-target CPUs-per-node
renorm applied in `apply`.

Currently only the Marconi100/PM100 parquet schema is implemented as a
source-job loader (`rank`'s --pm100 argument, columns job_id,
num_nodes_alloc, run_time, node_power_consumption): it is presently the only
dataset in this line of work with real per-node power *traces* rather than a
single per-job scalar total. Extending `rank` to another source dataset's
column schema is a separate, not-yet-done change.

Usage (four stages; --source-system/--target-system default to marconi100
and frontier, matching this tool's original fixed pairing):

    python -m redi.domains.hpc.cross_replay rank --pm100 .../p_m_100.parquet \\
        --out ranking.parquet

    python -m redi.domains.hpc.cross_replay extract-day \\
        --swf .../frontier-2025.swf --day-index-in-file 294 \\
        --out day_target.npz

    python -m redi.domains.hpc.cross_replay match --ranking ranking.parquet \\
        --target day_target.npz --pm100 .../p_m_100.parquet \\
        --out-subset matched_subset.parquet --out-matches matches.parquet

    # then run REDI on matched_subset.parquet, producing raps_jobs.npz

    python -m redi.domains.hpc.cross_replay apply --redi-npz raps_jobs.npz \\
        --matches matches.parquet --out cast_day.npz
"""

import argparse

import numpy as np
import pandas as pd

from .job_mapper import load_raps_system_params
from .swf_loader import parse_swf

MAX_TILE_REPEATS = 20  # log+cap rather than silently allowing pathological tiling

# Fallback match target when --target-intensity isn't given: midpoint of a
# "busy nodes at ~55-60% of dynamic envelope" band, observed for a Frontier
# day at the 60%-of-peak average. This default is Frontier-specific by
# construction (it is only correct when the target day itself runs near that
# average); for another target system, or an off-average day, compute a
# day-specific value with the `target-intensity` subcommand instead.
DEFAULT_TARGET_INTENSITY = 0.575


def _require_system_params(system_name):
    params = load_raps_system_params(system_name)
    if params is None:
        raise SystemExit(
            f"could not load RAPS system config for {system_name!r} (RAPS not "
            f"installed, or no such system config): this module needs the real "
            f"per-node power envelope, not a guessed one"
        )
    return params


def _node_envelope(power_config):
    """Whole-node idle and max power (W), summed across CPU + GPU + mem + NIC + NVMe."""
    cpu, gpu = power_config["cpu"], power_config["gpu"]
    idle = (cpu["per_node"] * cpu["idle"] + gpu["per_node"] * gpu["idle"]
            + power_config["power_mem"]
            + power_config["nics_per_node"] * power_config["power_nic"]
            + power_config["power_nvme"])
    max_ = (cpu["per_node"] * cpu["max"] + gpu["per_node"] * gpu["max"]
            + power_config["power_mem"]
            + power_config["nics_per_node"] * power_config["power_nic"]
            + power_config["power_nvme"])
    return idle, max_


def cmd_target_intensity(args):
    """Node-hour-weighted average busy-node dynamic intensity of a real
    target-system day, computed directly from its own measured
    cpu_trace/gpu_trace (in the target system's own idle/max envelope units).
    Use the printed value as `match`'s --target-intensity for this specific
    day, instead of the generic DEFAULT_TARGET_INTENSITY constant."""
    params = _require_system_params(args.target_system)
    power_config = params["power_config"]
    cpu, gpu = power_config["cpu"], power_config["gpu"]
    node_idle, node_max = _node_envelope(power_config)

    # cpu_trace is treated as a node-normalized aggregate CPU utilization
    # fraction (uniform across the system's per-node CPU count); gpu_trace is
    # treated as an already node-aggregate busy-GPU-equivalent count (0 to
    # gpu["per_node"]), matching the convention this formula was validated
    # against for Frontier (cpus_per_node == 1, so the two conventions
    # coincide there). Neither has been checked against a target system where
    # cpus_per_node > 1.
    cpu_idle_total = cpu["per_node"] * cpu["idle"]
    cpu_max_total = cpu["per_node"] * cpu["max"]

    d = np.load(args.npz, allow_pickle=True)
    jobs = d["jobs"]
    num, den, n_used = 0.0, 0.0, 0
    for j in jobs:
        st = float(j["start_time"])
        if not (0 <= st < 86400):
            continue
        n = int(j["nodes_required"])
        tq = float(j["trace_quanta"])
        c = np.asarray(j["cpu_trace"], dtype=float)
        g = np.asarray(j["gpu_trace"], dtype=float)
        m = min(len(c), len(g))
        if m == 0 or n <= 0:
            continue
        c, g = c[:m], g[:m]
        p_node = (cpu_idle_total + c * (cpu_max_total - cpu_idle_total)) \
            + (gpu["per_node"] * gpu["idle"] + g * (gpu["max"] - gpu["idle"]))
        x = (p_node - node_idle) / (node_max - node_idle)
        w = n * tq
        num += np.sum(x) * w
        den += m * w
        n_used += 1
    intensity = num / den
    print(f"jobs used: {n_used}")
    print(f"node-hour-weighted busy-node intensity: {intensity:.4f}")


def cmd_rank(args):
    params = _require_system_params(args.source_system)
    power_config = params["power_config"]
    trace_quanta = params["trace_quanta"]
    node_idle, node_max = _node_envelope(power_config)

    df = pd.read_parquet(args.pm100, columns=["job_id", "num_nodes_alloc", "run_time", "node_power_consumption"])
    avg_intensity = np.full(len(df), np.nan)
    duration_s = np.full(len(df), np.nan)

    for i, row in enumerate(df.itertuples(index=False)):
        n = int(row.num_nodes_alloc)
        node = np.asarray(row.node_power_consumption, dtype=float)
        if n <= 0 or node.size == 0:
            continue
        x = (node / n - node_idle) / (node_max - node_idle)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        avg_intensity[i] = np.mean(x)
        duration_s[i] = node.size * trace_quanta  # real trace length, not the run_time field

    df["avg_intensity"] = avg_intensity
    df["duration_s"] = duration_s
    df[["job_id", "num_nodes_alloc", "duration_s", "avg_intensity"]].to_parquet(args.out)
    valid = np.isfinite(avg_intensity) & np.isfinite(duration_s) & (duration_s > 0)
    print(f"ranked {valid.sum()} of {len(df)} jobs -> {args.out}")
    print(f"avg_intensity: mean={np.nanmean(avg_intensity):.3f} median={np.nanmedian(avg_intensity):.3f}")


def cmd_extract_day(args):
    df = parse_swf(args.swf)
    day_idx = np.floor_divide(df["submit_time_rel"].to_numpy(dtype=float), 86400).astype(int)
    runtime = df["runtime"].to_numpy(dtype=float)
    nodes = df["nodes_required_canonical"].to_numpy(dtype=float)
    mask = (
        (day_idx == args.day_index_in_file)
        & df["descriptor_parseable"].to_numpy()
        & df["runtime"].notna().to_numpy() & (runtime > 0)
        & df["nodes_required_canonical"].notna().to_numpy() & (nodes > 0)
    )
    sub = df.loc[mask]
    if sub.empty:
        raise SystemExit(f"no jobs found for day-index {args.day_index_in_file} in {args.swf}")

    submit_rel = sub["submit_time_rel"].to_numpy(dtype=float)
    sizes = sub["nodes_required_canonical"].to_numpy(dtype=int)
    tods = submit_rel % 86400.0
    durs = sub["runtime"].to_numpy(dtype=float)

    order = np.argsort(tods)  # chronological, for readability
    sizes, tods, durs = sizes[order], tods[order], durs[order]

    np.savez(args.out, target_sizes=sizes, target_tods=tods, target_durations=durs)
    print(f"day-index {args.day_index_in_file}: {len(sizes)} slots, "
          f"mean size {sizes.mean():.1f}, mean duration {durs.mean()/3600:.2f}h")
    print(f"saved -> {args.out}")


def cmd_extract_day_npz(args):
    """Extract the target-day schedule skeleton from a real (not SWF-derived)
    REDI/RAPS job-telemetry npz, i.e. a private real day used for a
    self-replay gate. Preferred over the SWF file when available: genuinely
    measured, no "Computer: Synthetic" provenance question.

    Only jobs whose start_time falls in [0, 86400) are treated as slots for
    this day (carry-over jobs starting before 0 are the target day's own
    initial occupancy, not something we're reconstructing)."""
    d = np.load(args.npz, allow_pickle=True)
    jobs = d["jobs"]
    slots = []
    for j in jobs:
        st = float(j["start_time"])
        et = float(j["end_time"])
        n = int(j["nodes_required"])
        if not (0 <= st < 86400) or et <= st or n <= 0:
            continue
        slots.append((n, st, et - st))

    if not slots:
        raise SystemExit(f"no in-window jobs found in {args.npz}")

    slots.sort(key=lambda s: s[1])
    sizes = np.array([s[0] for s in slots])
    tods = np.array([s[1] for s in slots])
    durs = np.array([s[2] for s in slots])
    np.savez(args.out, target_sizes=sizes, target_tods=tods, target_durations=durs)
    print(f"{args.npz}: {len(slots)} in-window slots (of {len(jobs)} total jobs), "
          f"mean size {sizes.mean():.1f}, mean duration {durs.mean()/3600:.2f}h")
    print(f"saved -> {args.out}")


def _robust_z(x, med, mad):
    return (x - med) / (1.4826 * mad if mad > 0 else 1.0)


def cmd_match(args):
    target = np.load(args.target)
    target_sizes = target["target_sizes"]
    target_tods = target["target_tods"]
    target_durations = target["target_durations"]
    n_slots = len(target_sizes)
    target_intensity = args.target_intensity if args.target_intensity is not None else DEFAULT_TARGET_INTENSITY
    print(f"matching against target intensity {target_intensity:.4f}")

    ranking = pd.read_parquet(args.ranking).dropna(subset=["avg_intensity", "duration_s"])
    ranking = ranking[ranking["duration_s"] > 0].reset_index(drop=True)

    log_dur = np.log(ranking["duration_s"].to_numpy())
    intensity = ranking["avg_intensity"].to_numpy()

    dur_med, dur_mad = np.median(log_dur), np.median(np.abs(log_dur - np.median(log_dur)))
    int_med, int_mad = np.median(intensity), np.median(np.abs(intensity - np.median(intensity)))

    pool = np.column_stack([_robust_z(log_dur, dur_med, dur_mad), _robust_z(intensity, int_med, int_mad)])
    job_ids = ranking["job_id"].astype(str).to_numpy()
    job_durations = ranking["duration_s"].to_numpy()

    target_log_dur = _robust_z(np.log(target_durations), dur_med, dur_mad)
    target_int = _robust_z(np.full(n_slots, target_intensity), int_med, int_mad)
    targets = np.column_stack([target_log_dur, target_int])

    matches = []
    n_capped = 0
    for slot_idx in range(n_slots):
        d = np.sum((pool - targets[slot_idx]) ** 2, axis=1)
        best = np.argmin(d)
        job_id = job_ids[best]
        job_dur = float(job_durations[best])
        tile_repeats = max(1, int(np.ceil(target_durations[slot_idx] / job_dur)))
        if tile_repeats > MAX_TILE_REPEATS:
            n_capped += 1
            tile_repeats = MAX_TILE_REPEATS
        matches.append((slot_idx, int(target_sizes[slot_idx]), float(target_tods[slot_idx]),
                         float(target_durations[slot_idx]), job_id, job_dur,
                         float(intensity[best]), tile_repeats))

    matches_df = pd.DataFrame(matches, columns=[
        "slot_idx", "target_size", "target_tod", "target_duration_s",
        "job_id", "job_duration_s", "job_avg_intensity", "tile_repeats"])
    matches_df.to_parquet(args.out_matches)

    unique_job_ids = set(matches_df["job_id"])
    print(f"matched {n_slots} slots to {len(unique_job_ids)} unique PM100 jobs "
          f"(reuse factor {n_slots/len(unique_job_ids):.2f}x)")
    if n_capped:
        print(f"WARNING: {n_capped} of {n_slots} slots capped at {MAX_TILE_REPEATS}x tiling "
              f"(target duration >> matched job duration)")

    df = pd.read_parquet(args.pm100)
    df["job_id_str"] = df["job_id"].astype(str)
    df_sel = df[df["job_id_str"].isin(unique_job_ids)].drop(columns=["job_id_str"]).reset_index(drop=True)
    df_sel.to_parquet(args.out_subset)
    print(f"selected PM100 job subset ({len(df_sel)} rows) -> {args.out_subset}")
    print(f"matches table -> {args.out_matches}")
    print(f"next: redi run -c examples/hpc/marconi100.yaml {args.out_subset} --output <dir> --style simple")


def _tile_trace(trace, repeats, target_len):
    arr = np.asarray(trace, dtype=float)
    if arr.size == 0:
        return arr
    tiled = np.tile(arr, repeats)
    return tiled[:target_len]


def cmd_apply(args):
    source_params = _require_system_params(args.source_system)
    target_params = _require_system_params(args.target_system)
    trace_quanta = source_params["trace_quanta"]

    source_cpus_per_node = source_params["power_config"]["cpu"]["per_node"]
    target_cpus_per_node = target_params["power_config"]["cpu"]["per_node"]
    cpu_renorm = target_cpus_per_node / source_cpus_per_node

    source_gpus_per_node = source_params["power_config"]["gpu"]["per_node"]
    target_gpus_per_node = target_params["power_config"]["gpu"]["per_node"]
    if source_gpus_per_node != target_gpus_per_node:
        print(f"WARNING: source system has {source_gpus_per_node} GPUs/node, target has "
              f"{target_gpus_per_node} (only cpu_trace is renormalized here; gpu_trace is "
              f"replayed as-is, which was only validated for a source/target pair where "
              f"GPUs/node happen to match)")

    matches = pd.read_parquet(args.matches)
    d = np.load(args.redi_npz, allow_pickle=True)
    jobs = d["jobs"]
    extra = {k: d[k] for k in d.files if k != "jobs"}

    template_by_id = {str(j["id"]): j for j in jobs}
    missing = set(matches["job_id"]) - set(template_by_id)
    if missing:
        raise SystemExit(f"{len(missing)} matched job_ids not found in REDI npz (ingestion subset mismatch): "
                          f"{sorted(missing)[:5]}...")

    out_jobs = []
    total_node_hours = 0.0
    for row in matches.itertuples(index=False):
        template = template_by_id[row.job_id]
        j = dict(template)  # shallow copy is enough; we overwrite every mutated field below

        target_len = int(round(row.target_duration_s / trace_quanta))
        for trace_key in ("cpu_trace", "gpu_trace"):
            tr = j.get(trace_key)
            if isinstance(tr, (list, np.ndarray)) and len(tr) > 0:
                tiled = _tile_trace(tr, row.tile_repeats, target_len)
                if trace_key == "cpu_trace":
                    tiled = tiled * cpu_renorm
                j[trace_key] = tiled.tolist()

        j["id"] = f"{row.job_id}_slot{row.slot_idx}"
        j["nodes_required"] = int(row.target_size)
        j["scheduled_nodes"] = []
        j["start_time"] = float(row.target_tod)
        j["end_time"] = float(row.target_tod) + float(row.target_duration_s)
        j["submit_time"] = float(row.target_tod)
        out_jobs.append(j)
        total_node_hours += row.target_size * row.target_duration_s / 3600.0

    out_jobs = np.array(out_jobs, dtype=object)
    np.savez(args.out, jobs=out_jobs, **extra)
    max_end_h = max(float(j["end_time"]) for j in out_jobs) / 3600.0
    print(f"built {len(out_jobs)} cast job-slots -> {args.out}")
    print(f"max end_time: {max_end_h:.2f}h, total node-hours demanded: {total_node_hours:,.0f}")
    print(f"replay with: python main.py run --system {args.target_system} --replay {args.out} "
          f"--time {int(max_end_h) + 1}h --time-delta 20s --scheduler default --policy replay --noui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_rank = sub.add_parser("rank")
    p_rank.add_argument("--pm100", required=True)
    p_rank.add_argument("--source-system", default="marconi100",
                         help="RAPS system name whose power envelope defines whole-node "
                              "dynamic intensity for ranking (default: marconi100)")
    p_rank.add_argument("--out", required=True)
    p_rank.set_defaults(func=cmd_rank)

    p_extract = sub.add_parser("extract-day")
    p_extract.add_argument("--swf", required=True)
    p_extract.add_argument("--day-index-in-file", type=int, required=True)
    p_extract.add_argument("--out", required=True)
    p_extract.set_defaults(func=cmd_extract_day)

    p_extract_npz = sub.add_parser("extract-day-npz")
    p_extract_npz.add_argument("--npz", required=True)
    p_extract_npz.add_argument("--out", required=True)
    p_extract_npz.set_defaults(func=cmd_extract_day_npz)

    p_target_intensity = sub.add_parser("target-intensity")
    p_target_intensity.add_argument("--npz", required=True)
    p_target_intensity.add_argument("--target-system", default="frontier",
                                     help="RAPS system name whose power envelope the npz's "
                                          "cpu_trace/gpu_trace are already normalized to "
                                          "(default: frontier)")
    p_target_intensity.set_defaults(func=cmd_target_intensity)

    p_match = sub.add_parser("match")
    p_match.add_argument("--ranking", required=True)
    p_match.add_argument("--target", required=True)
    p_match.add_argument("--pm100", required=True)
    p_match.add_argument("--out-subset", required=True)
    p_match.add_argument("--out-matches", required=True)
    p_match.add_argument("--target-intensity", type=float, default=None,
                          help="override the match target intensity "
                               f"(default: {DEFAULT_TARGET_INTENSITY} generic constant)")
    p_match.set_defaults(func=cmd_match)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--redi-npz", required=True)
    p_apply.add_argument("--matches", required=True)
    p_apply.add_argument("--source-system", default="marconi100",
                          help="RAPS system name the matched jobs' traces came from "
                               "(for trace_quanta and CPU renorm; default: marconi100)")
    p_apply.add_argument("--target-system", default="frontier",
                          help="RAPS system name being replayed onto "
                               "(for CPU renorm and the printed replay command; default: frontier)")
    p_apply.add_argument("--out", required=True)
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)
