#!/usr/bin/env python3
"""Download a time subset of TaiESM1 CMIP6 3D variables from ESGF via OPeNDAP.

Saves each variable as a local NetCDF file following the same naming convention
as the full ClimaX downloads:
  /opt/drai/CMIP6/TaiESM1/raw/<name>/<name>_<year_str>_raw.nc

Usage:
  python scripts/download_cmip6_subset.py [--ntimes 100]
"""

import argparse
import os
import netCDF4 as nc4
import numpy as np
from pathlib import Path

ESGF_BASE = (
    "https://esgf.ceda.ac.uk/thredds/dodsC/esg_cmip6/CMIP6/CMIP/"
    "AS-RCEC/TaiESM1/historical/r1i1p1f1/6hrPlevPt"
)
YEAR_STR = "186001010600-187001010000"
OUT_BASE = Path("/opt/drai/CMIP6/TaiESM1/raw")

# (cmip_name, era_name)
VARIABLES = [
    ("zg",  "geopotential"),
    ("ta",  "temperature"),
    ("hus", "specific_humidity"),
    ("ua",  "u_component_of_wind"),
    ("va",  "v_component_of_wind"),
]


def download_variable(cmip_name, out_name, ntimes, overwrite=False):
    remote_fname = (
        f"{cmip_name}_6hrPlevPt_TaiESM1_historical_r1i1p1f1_gn_{YEAR_STR}.nc"
    )
    url = f"{ESGF_BASE}/{cmip_name}/gn/v20201112/{remote_fname}"
    out_dir = OUT_BASE / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}_{YEAR_STR}_raw.nc"

    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path} already exists")
        return out_path

    print(f"  Opening remote: {url}")
    src = nc4.Dataset(url)

    actual_ntimes = min(ntimes, src.dimensions["time"].size)
    print(f"  Downloading {actual_ntimes}/{src.dimensions['time'].size} timesteps ...")

    # Create output file
    dst = nc4.Dataset(out_path, "w")

    # Copy global attributes
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    dst.subset_note = f"Time subset: first {actual_ntimes} steps of {YEAR_STR}"

    # Copy dimensions (subset time, keep everything else)
    for name, dim in src.dimensions.items():
        size = actual_ntimes if name == "time" else len(dim)
        dst.createDimension(name, size)

    # Copy variables
    for name, var in src.variables.items():
        out_var = dst.createVariable(
            name, var.datatype, var.dimensions,
            zlib=True, complevel=4
        )
        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        if "time" in var.dimensions:
            time_idx = var.dimensions.index("time")
            slices = tuple(
                slice(0, actual_ntimes) if i == time_idx else slice(None)
                for i in range(len(var.dimensions))
            )
            out_var[:] = var[slices]
        else:
            out_var[:] = var[:]

    src.close()
    dst.close()
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntimes", type=int, default=100,
                        help="Number of timesteps to download (default: 100)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even if file exists")
    args = parser.parse_args()

    print(f"Downloading {args.ntimes} timesteps for {len(VARIABLES)} variables\n")
    for cmip_name, out_name in VARIABLES:
        print(f"[{cmip_name} -> {out_name}]")
        download_variable(cmip_name, out_name, args.ntimes, overwrite=args.overwrite)
        print()
    print("Done.")


if __name__ == "__main__":
    main()
