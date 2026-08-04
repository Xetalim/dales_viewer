import glob
import os
from pathlib import Path

import numpy as np
import xarray as xr

from preprocessors import profiles


def open_dataset_optional(path, postprocess=None):
    """Open a dataset if it exists, optionally applying postprocess."""
    path = Path(path)
    try:
        if path.is_file():
            candidate_path = path
        else:
            matches = sorted(glob.glob(path.as_posix()))
            if not matches:
                return None
            candidate_path = Path(matches[0])

        ds = xr.open_dataset(
            candidate_path,
            chunks={"time": 16, "xt": -1, "yt": -1, "xm": -1, "ym": -1},
        )
    except FileNotFoundError:
        return None
    return postprocess(ds) if postprocess is not None else ds


def open_mfdataset_optional(pattern, postprocess=None):
    """Open a multi-file dataset from a glob pattern if matches exist."""
    try:
        ds = xr.open_mfdataset(
            glob.glob(pattern),
            chunks={"time": 16, "xt": -1, "yt": -1, "xm": -1, "ym": -1},
        )
    except (FileNotFoundError, OSError):
        return None
    return postprocess(ds) if postprocess is not None else ds


def virt_residual(ds):
    ds["residual"] = -ds["radbal"] + -ds["shf"] + -ds["lhf"] + ds["ghf"]
    return ds


def load_point_nc_files(output_path, prefix, preprocess_fn=None):
    """Load all <prefix>.<xi>.<yj>.nc files and return a label->dataset dict."""
    output_path = Path(output_path)
    result = {}

    for search_dir in [output_path, output_path / "run_001"]:
        if not search_dir.exists():
            continue
        for file_path in sorted(search_dir.glob(f"{prefix}.*.nc")):
            try:
                ds = xr.open_dataset(file_path)
                for index in ds.index:
                    locx = ds["xt"].sel(index=index).values
                    locy = ds["yt"].sel(index=index).values
                    label = f"x={float(locx):.0f}m, y={float(locy):.0f}m"
                    result[label] = (
                        preprocess_fn(ds.sel(index=index))
                        if preprocess_fn is not None
                        else ds.sel(index=index)
                    )
            except (FileNotFoundError, OSError, ValueError):
                continue

    return result or None


def load_profiles(output_path):
    ds_profiles = open_dataset_optional(
        Path(output_path) / "run_001" / "profiles.001.nc"
    )
    return profiles(ds_profiles) if ds_profiles is not None else None


def load_openboundaries(output_path):
    output_path = Path(output_path)
    ob_files = sorted((output_path / "input").glob("openboundaries.inp.*.nc"))
    return xr.open_dataset(ob_files[0]) if ob_files else None


def prepare_forcings(ds):
    if ds is None:
        return None
    if "ug_timedep" in ds and "vg_timedep" in ds:
        ds["uv_timedep"] = np.sqrt(ds["ug_timedep"] ** 2 + ds["vg_timedep"] ** 2)
    return ds
