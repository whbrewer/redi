"""
Dragon HPC parallel backend.

Uses the Dragon distributed runtime (https://dragonhpc.org) to execute
REDI across cores and nodes.  Dragon is a drop-in replacement for Python
multiprocessing that scales transparently from a laptop to thousands of
HPC nodes over Slingshot / InfiniBand / Ethernet.

Two execution modes (dragon_mode in ParallelConfig):

  subprocess  (default)
      Each Dragon worker calls ``redi run <file>`` as a subprocess.
      Same execution model as LocalBackend but without a GNU parallel
      dependency, and the same code scales to multi-node Dragon launches.

  inprocess   (experimental)
      Each Dragon worker imports and runs the REDI pipeline directly,
      avoiding subprocess overhead.  Large intermediate arrays live in
      Dragon managed memory instead of being serialised through OS pipes.
      Requires Dragon >= 0.9 with managed memory support.

Installation:
    pip install dragonhpc          # CPU-only
    pip install dragonhpc[gpu]     # GPU-aware channels

Multi-node launch (Slingshot / Frontier / Perlmutter):
    # Allocate nodes, then:
    dragon dragon_runner.py --input /data/nc/ --output ./out --workers 64

    # Or via SLURM (generate scripts with --generate-only):
    redi run data/*.nc --parallel --backend dragon --slurm-nodes 4
"""

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

from .base import ParallelBackend

if TYPE_CHECKING:
    from ..executor import ParallelConfig, ParallelResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level worker functions (must be picklable for Pool.map)
# ---------------------------------------------------------------------------


def _subprocess_worker(args: tuple) -> dict:
    """
    Run ``redi run <file>`` as a subprocess inside a Dragon worker process.

    Returns a plain dict so it is always serialisable back to the
    coordinator regardless of Dragon managed-memory availability.
    """
    file_path, output_dir, extra_args, style = args
    file_path = Path(file_path)
    file_out = Path(output_dir) / file_path.stem

    cmd = ["redi", "run", str(file_path), "--style", style, "-o", str(file_out)]
    cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            return {"success": True, "file": str(file_path), "output": str(file_out)}
        return {
            "success": False,
            "file": str(file_path),
            "error": result.stderr.strip() or f"exit code {result.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "file": str(file_path), "error": "timeout (3600 s)"}
    except Exception as exc:
        return {"success": False, "file": str(file_path), "error": str(exc)}


def _inprocess_worker(args: tuple) -> dict:
    """
    Run the REDI pipeline in-process inside a Dragon worker.

    Calls redi.cli.run_pipeline() directly, avoiding subprocess overhead.
    When Dragon managed memory is available the pipeline's intermediate
    numpy arrays are stored there and are not pickled through OS pipes.
    """
    file_path, output_dir, domain, stages, output_format = args
    file_path = Path(file_path)
    file_out = Path(output_dir) / file_path.stem

    try:
        from redi.cli import run_pipeline  # type: ignore[import]

        run_pipeline(
            input_path=str(file_path),
            output_dir=str(file_out),
            domain=domain,
            stages=stages.split(",") if stages else None,
            output_format=output_format,
            style="simple",
        )
        return {"success": True, "file": str(file_path), "output": str(file_out)}
    except Exception as exc:
        return {"success": False, "file": str(file_path), "error": str(exc)}


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class DragonBackend(ParallelBackend):
    """
    Execute REDI in parallel using the Dragon distributed runtime.

    Wraps Dragon's ``multiprocessing.Pool`` — the same API as stdlib
    multiprocessing, but backed by Dragon channels and managed memory so
    it scales transparently across nodes.

    Falls back to ``multiprocessing.Pool`` (with a warning) when Dragon
    is not installed, making the backend usable for local testing.
    """

    @property
    def name(self) -> str:
        return "dragon"

    def check_available(self) -> bool:
        """Return True when the ``dragon`` package is importable."""
        return importlib.util.find_spec("dragon") is not None

    # ------------------------------------------------------------------
    # Pool factory — returns either dragon.multiprocessing.Pool or
    # stdlib multiprocessing.Pool, with appropriate start-method setup.
    # ------------------------------------------------------------------

    def _pool(self, workers: int):
        """Return an open Pool configured for Dragon or stdlib."""
        if self.check_available():
            import dragon  # noqa: F401  (side-effect: registers Dragon runtime)
            import dragon.multiprocessing as dmp

            # Dragon's start method must be set before the first Pool.
            try:
                dmp.set_start_method("dragon")
            except RuntimeError:
                pass  # already set in this process

            logger.info("Using Dragon multiprocessing pool (%d workers)", workers)
            return dmp.Pool(workers)
        else:
            import multiprocessing as mp

            logger.warning(
                "dragon package not found — falling back to stdlib multiprocessing. "
                "Install with: pip install dragonhpc"
            )
            ctx = mp.get_context("spawn")
            return ctx.Pool(workers)

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(
        self,
        file_batches: List[List[Path]],
        config: "ParallelConfig",
    ) -> "ParallelResult":
        """
        Distribute REDI over files using Dragon's Pool.

        Args:
            file_batches: Batches produced by the executor's distributor.
                          Flattened here because Dragon's scheduler handles
                          work balancing natively.
            config: Parallel configuration.

        Returns:
            ParallelResult with per-file success/failure counts.
        """
        from ..executor import ParallelResult

        all_files: List[Path] = [f for batch in file_batches for f in batch]
        total = len(all_files)
        if not all_files:
            return ParallelResult(success=True, total_files=0, completed=0, failed=0)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mode = getattr(config, "dragon_mode", "subprocess")
        logger.info(
            "Dragon backend: %d files, %d workers, mode=%s",
            total, config.workers, mode,
        )

        if mode == "inprocess":
            worker_fn = _inprocess_worker
            domain = getattr(config, "dragon_domain", None) or "auto"
            stages = getattr(config, "dragon_stages", "") or ""
            fmt = getattr(config, "dragon_output_format", "npz") or "npz"
            work_args = [
                (str(f), str(output_dir), domain, stages, fmt)
                for f in all_files
            ]
        else:
            worker_fn = _subprocess_worker
            work_args = [
                (str(f), str(output_dir), list(config.redi_args), config.style)
                for f in all_files
            ]

        errors: List[str] = []
        completed = 0
        output_dirs: List[Path] = []

        with self._pool(config.workers) as pool:
            results = pool.map(worker_fn, work_args)

        for r in results:
            if r["success"]:
                completed += 1
                output_dirs.append(Path(r["output"]))
            else:
                errors.append(f"{r['file']}: {r['error']}")

        failed = total - completed
        return ParallelResult(
            success=failed == 0,
            total_files=total,
            completed=completed,
            failed=failed,
            output_dirs=output_dirs,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # generate_scripts
    # ------------------------------------------------------------------

    def generate_scripts(
        self,
        input_paths: List[Path],
        script_dir: Path,
        config: "ParallelConfig",
    ) -> Dict[str, Path]:
        """
        Generate Dragon launcher artefacts.

        Creates:
          dragon_runner.py   — worker script launched with ``dragon``
          run_dragon.sh      — convenience wrapper (local or SLURM)
          input_files.txt    — file list consumed by dragon_runner.py

        Args:
            input_paths: Input file paths.
            script_dir: Directory where artefacts are written.
            config: Parallel configuration.

        Returns:
            Mapping of artefact names to their paths.
        """
        script_dir = Path(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)

        # ---- file list -----------------------------------------------
        file_list_path = script_dir / "input_files.txt"
        file_list_path.write_text("\n".join(str(p) for p in input_paths) + "\n")

        # ---- dragon_runner.py ----------------------------------------
        mode = getattr(config, "dragon_mode", "subprocess")
        redi_args_repr = repr(list(config.redi_args))
        domain = getattr(config, "dragon_domain", None) or "auto"
        stages = getattr(config, "dragon_stages", "") or ""
        fmt = getattr(config, "dragon_output_format", "npz") or "npz"

        runner_path = script_dir / "dragon_runner.py"
        runner_path.write_text(
            f"""\
#!/usr/bin/env python
\"\"\"
REDI Dragon runner — generated by ``redi --parallel --backend dragon``.

Launch:
    dragon dragon_runner.py              # local multi-core
    dragon dragon_runner.py --workers N  # explicit worker count

Multi-node (Dragon must already be allocated across nodes):
    dragon --nodes {getattr(config, "dragon_nodes", 1)} dragon_runner.py
\"\"\"

import argparse
import sys
from pathlib import Path

import dragon                             # noqa: F401  (starts Dragon runtime)
import dragon.multiprocessing as dmp

try:
    dmp.set_start_method("dragon")
except RuntimeError:
    pass

# Import worker functions from the installed redi package
from redi.parallel.backends.dragon import (
    _subprocess_worker,
    _inprocess_worker,
)


def main() -> None:
    p = argparse.ArgumentParser(description="REDI Dragon parallel runner")
    p.add_argument("--filelist", default={str(file_list_path)!r},
                   help="Text file with one input path per line")
    p.add_argument("--output", default={str(config.output_dir)!r},
                   help="Base output directory")
    p.add_argument("--workers", type=int, default={config.workers},
                   help="Dragon pool size (default: %(default)s)")
    p.add_argument("--mode", default={mode!r},
                   choices=["subprocess", "inprocess"],
                   help="Execution mode (default: %(default)s)")
    args = p.parse_args()

    file_list = Path(args.filelist).read_text().splitlines()
    file_list = [f for f in file_list if f.strip()]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "inprocess":
        worker_fn = _inprocess_worker
        work_args = [
            (f, str(output_dir), {domain!r}, {stages!r}, {fmt!r})
            for f in file_list
        ]
    else:
        worker_fn = _subprocess_worker
        work_args = [
            (f, str(output_dir), {redi_args_repr}, "simple")
            for f in file_list
        ]

    print(f"Dragon pool: {{args.workers}} workers, {{len(file_list)}} files, mode={{args.mode}}")
    with dmp.Pool(args.workers) as pool:
        results = pool.map(worker_fn, work_args)

    ok = sum(1 for r in results if r["success"])
    fail = len(results) - ok
    print(f"Done: {{ok}} succeeded, {{fail}} failed")
    if fail:
        for r in results:
            if not r["success"]:
                print(f"  FAIL {{r['file']}}: {{r['error']}}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
        )
        runner_path.chmod(0o755)

        # ---- run_dragon.sh -------------------------------------------
        dragon_nodes = getattr(config, "dragon_nodes", 1)
        launcher = getattr(config, "dragon_launcher", "local")

        if launcher == "slurm":
            account_line = (
                f"#SBATCH --account={config.slurm_account}"
                if config.slurm_account
                else "#SBATCH --account=<YOUR_ACCOUNT>"
            )
            run_sh_content = f"""\
#!/bin/bash
#SBATCH --job-name=redi-dragon
#SBATCH --nodes={dragon_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --time={config.slurm_time}
#SBATCH --partition={config.slurm_partition}
{account_line}
#SBATCH --output=redi-dragon-%j.out
#SBATCH --error=redi-dragon-%j.err

# Load modules / activate environment
{"".join(f"module load {m}{chr(10)}" for m in config.slurm_modules)}
conda activate {config.slurm_conda_env}

# Dragon multi-node launch (Slingshot/InfiniBand)
dragon --nodes {dragon_nodes} {runner_path} --workers {config.workers}
"""
        else:
            run_sh_content = f"""\
#!/bin/bash
# REDI Dragon local launcher — generated by redi --parallel --backend dragon

set -e

WORKERS={config.workers}
RUNNER="{runner_path}"
OUTPUT="{config.output_dir}"

# Activate environment if needed
# source /opt/venvs/drai/bin/activate

echo "Launching Dragon pool with $WORKERS workers..."
dragon "$RUNNER" --workers "$WORKERS" --output "$OUTPUT"
echo "Done."
"""

        run_sh_path = script_dir / "run_dragon.sh"
        run_sh_path.write_text(run_sh_content)
        run_sh_path.chmod(0o755)

        return {
            "dragon_runner.py": runner_path,
            "run_dragon.sh": run_sh_path,
            "input_files.txt": file_list_path,
        }
