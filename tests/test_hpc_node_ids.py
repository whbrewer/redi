"""Node-name ids must not depend on the Python process (PYTHONHASHSEED)."""
import os
import subprocess
import sys
from pathlib import Path

from redi.domains.hpc.job_mapper import _parse_node_list, _stable_node_id, convert_node_names

ROOT = str(Path(__file__).resolve().parent.parent)
CODE = ("from redi.domains.hpc.job_mapper import _parse_node_list, convert_node_names;"
        "print(_parse_node_list('x4720c2s2b0n0,x4720c6s5b0n0'), convert_node_names(['r1-n2','r3-n4'], 'mit_supercloud', None))")


def test_ids_identical_across_processes_with_different_hash_seeds():
    outs = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": ROOT}
        outs.add(subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=env).stdout)
    assert len(outs) == 1 and outs.pop().strip()


def test_ids_are_stable_distinct_ints_in_range():
    names = [f"node{i:04d}" for i in range(2000)]
    ids = [_stable_node_id(n) for n in names]
    assert ids == [_stable_node_id(n) for n in names]
    assert len(set(ids)) == len(ids) and all(0 <= i < 10 ** 9 for i in ids)
    assert _parse_node_list("a,b") == [_stable_node_id("a"), _stable_node_id("b")]
    assert convert_node_names(["a", "b"], "unknown", None) == [_stable_node_id("a"), _stable_node_id("b")]
