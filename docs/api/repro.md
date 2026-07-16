# redi.repro

Reproducibility logging for REDI. Tracks commands run during a session
and appends them to a `repro.sh` script in the output directory,
including full environment metadata (host, Python version, git commit
hash, working directory).

**Source:** [`redi/repro.py`](../../redi/repro.py)
