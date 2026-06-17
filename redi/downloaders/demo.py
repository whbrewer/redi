"""
Demo dataset downloader for ``redi demo``.

Each entry in DEMOS is a small, no-auth-required public dataset (or a
synthetic generator) that exercises a different REDI domain.
"""

import csv
import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEMOS: Dict[str, dict] = {
    "general": {
        "name": "UCI Wine Quality",
        "description": "White wine physicochemical properties + quality scores (4,898 samples)",
        "domain": "general",
        "size": "~84 KB",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        "filename": "wine_quality.csv",
        "format": "csv",
    },
    "hpc": {
        "name": "Synthetic HPC Job Log",
        "description": "Simulated HPC job telemetry (5,000 jobs across 4 partitions)",
        "domain": "hpc",
        "size": "~300 KB",
        "source": "generated",
        "url": None,
        "filename": "hpc_jobs.csv",
        "format": "csv",
    },
    "climate": {
        "name": "NOAA GHCND — Honolulu Airport",
        "description": "Daily temperature & precipitation observations 2010–2023",
        "domain": "climate",
        "size": "~400 KB",
        "source": "NOAA National Centers for Environmental Information",
        "url": "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USC00519281.csv",
        "filename": "ghcnd_honolulu.csv",
        "format": "csv",
    },
    "geospatial": {
        "name": "EPA Annual Air Quality Index by County (2023)",
        "description": "AQI statistics for ~3,000 US counties",
        "domain": "geospatial",
        "size": "~250 KB",
        "source": "US EPA Air Quality System",
        "url": "https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2023.zip",
        "filename": "epa_aqi_county_2023.csv",
        "format": "csv_in_zip",
    },
    "materials": {
        "name": "Synthetic Crystal Structure Dataset",
        "description": "Simulated material properties (formula, lattice, band gap, 2,000 entries)",
        "domain": "materials",
        "size": "~150 KB",
        "source": "generated",
        "url": None,
        "filename": "materials_sample.csv",
        "format": "csv",
    },
}

DEFAULT_DOMAIN = "general"


def list_demos() -> Dict[str, dict]:
    return DEMOS


def download_demo(domain: str, output_dir: Path) -> Tuple[Path, dict]:
    """Download or generate demo data for *domain* into *output_dir*.

    Returns ``(path_to_file, demo_metadata_dict)``.
    """
    if domain not in DEMOS:
        available = ", ".join(DEMOS.keys())
        raise ValueError(f"No demo for domain '{domain}'. Available: {available}")

    meta = DEMOS[domain]
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / meta["filename"]

    if meta["url"] is None:
        _generate_synthetic(domain, dest)
    elif meta["format"] == "csv_in_zip":
        _download_zip_csv(meta["url"], dest)
    else:
        _download_url(meta["url"], dest)

    return dest, meta


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _download_url(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "redi-demo/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def _download_zip_csv(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "redi-demo/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in zip from {url}")
        dest.write_bytes(zf.read(csv_names[0]))


# ---------------------------------------------------------------------------
# Synthetic generators (for domains with no freely available public logs)
# ---------------------------------------------------------------------------

def _generate_synthetic(domain: str, dest: Path) -> None:
    import random
    rng = random.Random(42)
    if domain == "hpc":
        _gen_hpc(dest, rng)
    elif domain == "materials":
        _gen_materials(dest, rng)
    else:
        raise ValueError(f"No synthetic generator for domain '{domain}'")


def _gen_hpc(dest: Path, rng) -> None:
    partitions = ["gpu", "cpu", "bigmem", "debug"]
    users = [f"user{i:03d}" for i in range(50)]
    rows = []
    for i in range(5000):
        nodes = rng.choice([1, 2, 4, 8, 16, 32])
        req_walltime = rng.choice([3600, 7200, 14400, 28800, 86400])
        actual_walltime = int(req_walltime * rng.uniform(0.1, 1.0))
        wait = max(0, int(rng.gauss(300, 600)))
        rows.append({
            "job_id": 100000 + i,
            "user": rng.choice(users),
            "partition": rng.choice(partitions),
            "nodes": nodes,
            "cpus": nodes * 56,
            "req_walltime": req_walltime,
            "walltime": actual_walltime,
            "queue_time": wait,
            "submit_time": 1700000000 + i * 120,
            "start_time": 1700000000 + i * 120 + wait,
            "exit_code": 0 if rng.random() > 0.05 else 1,
        })
    _write_csv(dest, rows)


def _gen_materials(dest: Path, rng) -> None:
    elements = [
        "Li", "Na", "K", "Mg", "Ca", "Al", "Si", "Fe", "Cu", "Zn",
        "Ti", "Mn", "Co", "Ni", "Ga", "Ge", "Sr", "Zr", "Nb", "Mo",
    ]
    lattice_systems = [
        "cubic", "tetragonal", "orthorhombic", "hexagonal", "monoclinic", "triclinic",
    ]
    rows = []
    for i in range(2000):
        e1, e2 = rng.choice(elements), rng.choice(elements)
        n1, n2 = rng.randint(1, 4), rng.randint(1, 4)
        formula = f"{e1}{n1}{e2}{n2}" if n2 > 1 else f"{e1}{n1}{e2}"
        a = round(rng.uniform(3.0, 10.0), 4)
        b = round(rng.uniform(3.0, 10.0), 4)
        c = round(rng.uniform(3.0, 10.0), 4)
        rows.append({
            "material_id": f"mp-{10000 + i}",
            "formula": formula,
            "lattice_system": rng.choice(lattice_systems),
            "a": a, "b": b, "c": c,
            "volume": round(a * b * c, 3),
            "nsites": n1 + n2,
            "band_gap": round(max(0.0, rng.gauss(1.5, 1.2)), 3),
            "formation_energy_per_atom": round(rng.gauss(-1.2, 0.8), 4),
            "energy_above_hull": round(max(0.0, rng.gauss(0.05, 0.08)), 4),
            "is_stable": 1 if rng.random() > 0.3 else 0,
        })
    _write_csv(dest, rows)


def _write_csv(dest: Path, rows: list) -> None:
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
