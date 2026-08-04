from pathlib import Path

import numpy as np
import xarray as xr

from data_loading import open_dataset_optional, open_mfdataset_optional


def _write_dataset(path: Path) -> xr.Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({"foo": ("x", np.arange(3))}, coords={"x": [0, 1, 2]})
    ds.to_netcdf(path)
    return ds


def test_open_dataset_optional_opens_exact_file(tmp_path):
    path = tmp_path / "crossxy.nc"
    expected = _write_dataset(path)

    ds = open_dataset_optional(path)

    assert ds is not None
    xr.testing.assert_identical(ds, expected)


def test_open_dataset_optional_opens_glob_pattern(tmp_path):
    path = tmp_path / "run_001" / "crossxy.001.001.nc"
    expected = _write_dataset(path)
    pattern = tmp_path / "run_001" / "crossxy.*.001.nc"

    ds = open_dataset_optional(pattern)

    assert ds is not None
    xr.testing.assert_identical(ds, expected)


def test_open_mfdataset_optional_opens_glob_pattern(tmp_path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)

    ds0 = xr.Dataset({"foo": ("time", [1.0])}, coords={"time": [0]})
    ds1 = xr.Dataset({"foo": ("time", [2.0])}, coords={"time": [1]})
    ds0.to_netcdf(run_dir / "crossxy.0001.001.nc")
    ds1.to_netcdf(run_dir / "crossxy.0002.001.nc")

    ds = open_mfdataset_optional((run_dir / "crossxy.*.001.nc").as_posix())

    assert ds is not None
    assert "time" in ds.dims
    assert int(ds.dims["time"]) == 2
    np.testing.assert_array_equal(ds["time"].values, np.array([0, 1]))
