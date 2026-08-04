from preprocessors import cape, crosses, profiles

from data_loading import (
    load_openboundaries,
    load_point_nc_files,
    load_profiles,
    open_dataset_optional,
    open_mfdataset_optional,
    prepare_forcings,
    virt_residual,
)
from make_basic_panels import (
    make_2d_xy_panel,
    make_init_profile_panel,
    make_tmser_panel,
)
from make_colstat_panel import make_colstat_panel
from make_fielddump_slice_panel import make_fielddump_slice_panel
from make_heat_panel_with_clim_controls import make_heat_panel_with_clim_controls
from make_lsm_panel import make_lsm_panel
from make_meanstd_or_horizontal_panel import make_meanstd_or_horizontal_panel
from make_slice_panel import make_slice_panel
from make_slrbcross_panel import make_slrbcross_panel
from make_virtualmeasurement_panel import make_virtualmeasurement_panel


def make_loader_registry(output_path):
    """Return a safe, preregistered loader function registry."""
    return {
        "load_profiles": lambda: load_profiles(output_path),
        "load_tmser": lambda: open_dataset_optional(
            output_path / "run_001" / "tmser.001.nc"
        ),
        "load_cape": lambda: open_dataset_optional(output_path / "cape.nc")
        or open_dataset_optional(output_path / "run_001" / "cape.001.nc"),
        "load_surfcross": lambda: open_dataset_optional(output_path / "surfcross.nc")
        or open_dataset_optional(output_path / "run_001" / "surfcross.001.nc"),
        "load_sampling": lambda: open_dataset_optional(
            output_path / "run_001" / "sampling.001.nc", postprocess=profiles
        ),
        "load_samptend": lambda: open_dataset_optional(
            output_path / "run_001" / "samptend.001.nc", postprocess=profiles
        ),
        "load_crossxz": lambda: open_dataset_optional(output_path / "crossxz.nc")
        or open_mfdataset_optional(
            (output_path / "run_001" / "crossxz.*.001.nc").as_posix()
        ),
        "load_crossyz": lambda: open_dataset_optional(output_path / "crossyz.nc")
        or open_mfdataset_optional(
            (output_path / "run_001" / "crossyz.*.001.nc").as_posix()
        ),
        "load_crossxy": lambda: open_dataset_optional(output_path / "crossxy.nc")
        or open_mfdataset_optional(
            (output_path / "run_001" / "crossxy.*.001.nc").as_posix()
        ),
        "load_fielddump": lambda: open_dataset_optional(output_path / "fielddump.nc")
        or open_dataset_optional(output_path / "run_001" / "fielddump.001.nc"),
        "load_samptend_all": lambda: open_dataset_optional(
            output_path / "run_001" / "samptend_all.001.nc"
        ),
        "load_all_dump": lambda: open_dataset_optional(output_path / "all_dump.nc"),
        "load_openboundaries": lambda: load_openboundaries(output_path),
        "load_lsm": lambda: open_dataset_optional(
            output_path / "input" / "lsm.inp_001.nc"
        ),
        "load_inslurb": lambda: open_dataset_optional(
            output_path / "input" / "inslurb.001.nc"
        ),
        "load_init": lambda: open_dataset_optional(
            output_path / "input" / "init.001.nc"
        ),
        "load_forcings": lambda: prepare_forcings(
            open_dataset_optional(output_path / "input" / "forcings.001.nc")
        ),
        "load_slrbcross": lambda: open_dataset_optional(output_path / "slrbcross.nc")
        or open_mfdataset_optional(
            (output_path / "run_001").as_posix() + "/slurbcross_*.nc"
        ),
        "load_radfield": lambda: open_dataset_optional(output_path / "radfield.nc")
        or open_dataset_optional(output_path / "run_001" / "radfield.001.nc"),
        "load_virtualmeasurement": lambda: load_point_nc_files(
            output_path, "virtualmeasurement", preprocess_fn=virt_residual
        ),
        "load_colstat": lambda: load_point_nc_files(output_path, "colstat"),
    }


def make_builder_registry(ds_cache):
    """Return a safe, preregistered builder function registry."""
    return {
        "build_heat_panel": make_heat_panel_with_clim_controls,
        "build_tmser_panel": make_tmser_panel,
        "build_cape_panel": lambda d: make_meanstd_or_horizontal_panel(
            d,
            cape,
            slice_dim="time",
            toggle_label="CAPE view",
            ds_lsm=ds_cache.get("lsm"),
        ),
        "build_surfcross_panel": lambda d: make_meanstd_or_horizontal_panel(
            d,
            crosses,
            slice_dim="time",
            toggle_label="Surfcross view",
            ds_lsm=ds_cache.get("lsm"),
        ),
        "build_slice_yt_panel": lambda d: make_slice_panel(d, "yt"),
        "build_slice_xt_panel": lambda d: make_slice_panel(d, "xt"),
        "build_slice_zt_panel": lambda d: make_slice_panel(d, "zt"),
        "build_fielddump_panel": make_fielddump_slice_panel,
        "build_slice_time_panel": lambda d: make_slice_panel(d, "time"),
        "build_lsm_panel": make_lsm_panel,
        "build_2d_xy_panel": make_2d_xy_panel,
        "build_init_profile_panel": make_init_profile_panel,
        "build_slrbcross_panel": lambda d: make_slrbcross_panel(d, ds_cache.get("lsm")),
        "build_radfield_panel": lambda d: make_meanstd_or_horizontal_panel(
            d,
            cape,
            slice_dim="time",
            toggle_label="Radfield view",
            ds_lsm=ds_cache.get("lsm"),
        ),
        "build_virtualmeasurement_panel": make_virtualmeasurement_panel,
        "build_colstat_panel": make_colstat_panel,
    }
