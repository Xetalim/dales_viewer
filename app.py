import xarray as xr
import panel as pn
import pathlib
import numpy as np
import glob
import os
import holoviews as hv
from holoviews import streams
import hvplot.xarray  # noqa: F401
import param
import logging

from _meanstd_overlay import _meanstd_overlay
from controllers import (
    ClimController,
)
from helpers import (
    _compute_slice_clim,
    _make_clim_controls,
    apply_horizontal_mask,
    build_cover_mask,
    determine_clim_and_cmap,
    get_label_to_var,
    catchall,
)
from make_fielddump_slice_panel import make_fielddump_slice_panel
from make_heat_panel_with_clim_controls import make_heat_panel_with_clim_controls
from make_meanstd_or_horizontal_panel import make_meanstd_or_horizontal_panel
from make_slice_panel import make_slice_panel
from preprocessors import cape, crosses, profiles

hv.extension("bokeh")

# Ensure that panning/zooming in one plot does not affect others by
# disabling axis sharing globally. Individual panels can still override
# this per-element if needed.
hv.opts.defaults(
    hv.opts.Image(shared_axes=False, axiswise=True),
    hv.opts.QuadMesh(shared_axes=False, axiswise=True),
    hv.opts.Curve(shared_axes=False, axiswise=True),
    hv.opts.HeatMap(shared_axes=False, axiswise=True),
)

FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


def _open_dataset_optional(path, postprocess=None):
    """Open a dataset if it exists, optionally apply a postprocess function."""

    try:
        if os.path.isfile(path):
            ds = xr.open_dataset(path, chunks={"time": "auto"})
        else:
            return None
    except FileNotFoundError:
        return None
    return postprocess(ds) if postprocess is not None else ds


def _open_mfdataset_optional(path, postprocess=None):
    """Open a dataset if it exists, optionally apply a postprocess function."""
    try:
        ds = xr.open_mfdataset(glob.glob(path), chunks={"time": "auto"})
    except FileNotFoundError, OSError:
        return None
    return postprocess(ds) if postprocess is not None else ds


def _load_datasets(output_path):
    """Load all datasets from *output_path* and return them in a dict."""
    ds = {}

    ds["fielddump"] = _open_dataset_optional(
        output_path / "fielddump.nc"
    ) or _open_dataset_optional(output_path / "run_001" / "fielddump.001.nc")
    ds["crossxz"] = _open_dataset_optional(output_path / "crossxz.nc")
    ds["crossyz"] = _open_dataset_optional(output_path / "crossyz.nc")
    ds["crossxy"] = _open_dataset_optional(output_path / "crossxy.nc")
    ds_profiles = _open_dataset_optional(output_path / "run_001" / "profiles.001.nc")
    ds["prof"] = profiles(ds_profiles) if ds_profiles is not None else None
    ds["cape_raw"] = _open_dataset_optional(
        output_path / "cape.nc"
    ) or _open_dataset_optional(output_path / "run_001" / "cape.001.nc")
    ds["radfield_raw"] = _open_dataset_optional(
        output_path / "radfield.nc"
    ) or _open_dataset_optional(output_path / "run_001" / "radfield.001.nc")
    ds["crosses_raw"] = _open_dataset_optional(
        output_path / "surfcross.nc"
    ) or _open_dataset_optional(output_path / "run_001" / "surfcross.001.nc")
    ds["tmser"] = _open_dataset_optional(output_path / "run_001" / "tmser.001.nc")
    ds["samptend"] = _open_dataset_optional(
        output_path / "run_001" / "samptend.001.nc", postprocess=profiles
    )
    ds["sampling"] = _open_dataset_optional(
        output_path / "run_001" / "sampling.001.nc", postprocess=profiles
    )
    ds["dump"] = _open_dataset_optional(output_path / "all_dump.nc")

    try:
        ob_input_dir = output_path / "input"
        ob_files = sorted(ob_input_dir.glob("openboundaries.inp.*.nc"))
        ds["openboundaries"] = xr.open_dataset(ob_files[0]) if ob_files else None
    except FileNotFoundError:
        ds["openboundaries"] = None

    input_dir = output_path / "input"
    ds["lsm"] = _open_dataset_optional(input_dir / "lsm.inp_001.nc")
    ds["inslurb"] = _open_dataset_optional(input_dir / "inslurb.001.nc")
    ds["init"] = _open_dataset_optional(input_dir / "init.001.nc")
    ds["forcings"] = _open_dataset_optional(input_dir / "forcings.001.nc")
    ds["slrbcross"] = _open_dataset_optional(
        output_path / "slrbcross.nc"
    ) or _open_mfdataset_optional(
        (output_path / "run_001").as_posix() + "/slurbcross_*.nc"
    )

    return ds


# ---- Input file panel builders ----


_SLRB_SUFFIXES = [
    "wall_a",
    "wall_b",
    "win_a",
    "win_b",
    "road",
    "roof",
    "wall",
    "win",
    "urb",
    "can",
    "facade",
]


def _parse_slrb_categories(ds):
    """Group SLURB cross variables by surface-type suffix."""
    categories = {s: [] for s in _SLRB_SUFFIXES}
    categories["other"] = []

    for v in ds.data_vars:
        matched = False
        for suffix in _SLRB_SUFFIXES:
            if v.endswith(f"_{suffix}") or v == suffix:
                categories[suffix].append(v)
                matched = True
                break
        if not matched:
            categories["other"].append(v)

    # Remove empty categories
    return {k: sorted(v) for k, v in categories.items() if v}


class SLRBController(param.Parameterized):
    category = param.ObjectSelector(default=None, objects=[], label="Category")
    variable = param.ObjectSelector(default=None, objects=[], label="Variable")
    time_index = param.Integer(default=0, bounds=(0, 0), label="Time index")
    zts_index = param.Integer(default=0, bounds=(0, 0), label="zts layer")
    view = param.ObjectSelector(
        default="mean/std", objects=["mean/std", "horizontal"], label="View"
    )
    auto = param.Boolean(default=True, doc="Automatically set clim from current view")
    trigger = param.Integer(default=0, doc="Manual trigger to recompute clim from view")
    mask_cover = param.Boolean(
        default=False,
        doc="Mask (x, y) where cover_slb == 0 in lsm.inp_001.nc",
    )


def _build_slrbcross_cover_mask(ds_slrb, ds_lsm):
    """Return a (xt, yt)-shaped mask from lsm cover_slb, or None.

    The mask is 1 where cover_slb > 0 and NaN where cover_slb == 0 so that
    multiplying will turn those locations into NaN in all fields.
    """
    return build_cover_mask(ds_slrb, ds_lsm)


def make_slrbcross_panel(ds, ds_lsm=None):
    """Panel for slrbcross.nc with category filtering."""

    categories = _parse_slrb_categories(ds)
    cat_names = list(categories.keys())

    if not cat_names:
        return pn.pane.Markdown("No variables", sizing_mode="stretch_width")

    # Optional cover_slb-based mask from lsm.inp_001.nc
    cover_mask = _build_slrbcross_cover_mask(ds, ds_lsm)

    # Pre-compute mean and std over horizontal dims for all variables
    horiz_dims = [d for d in ("xt", "yt") if d in ds.dims]
    ds_mean = ds.mean(dim=horiz_dims, keep_attrs=True) if horiz_dims else ds
    ds_std = ds.std(dim=horiz_dims, keep_attrs=True) if horiz_dims else None

    ds_masked = None
    ds_mean_masked = None
    ds_std_masked = None
    if cover_mask is not None and horiz_dims:
        ds_masked = apply_horizontal_mask(ds, cover_mask)
        ds_mean_masked = ds_masked.mean(dim=horiz_dims, keep_attrs=True, skipna=True)
        ds_std_masked = ds_masked.std(dim=horiz_dims, keep_attrs=True, skipna=True)

    controller = SLRBController()
    controller.param["category"].objects = cat_names
    controller.category = cat_names[0]

    default_vars = categories[cat_names[0]]
    controller.param["variable"].objects = default_vars
    controller.variable = default_vars[0] if default_vars else None

    if "time" in ds.dims:
        n_time = int(ds.sizes["time"])
        controller.param["time_index"].bounds = (0, max(n_time - 1, 0))
    if "zts" in ds.dims:
        n_zts = int(ds.sizes["zts"])
        controller.param["zts_index"].bounds = (0, max(n_zts - 1, 0))

    def _update_vars(event):
        cat = controller.category
        var_list = categories.get(cat, [])
        controller.param["variable"].objects = var_list
        controller.variable = var_list[0] if var_list else None

    controller.param.watch(_update_vars, "category")

    # --- Mean/std DynamicMap ---
    ms_param_stream = streams.Params(
        controller,
        parameters=["category", "variable", "zts_index", "view", "mask_cover"],
    )

    @catchall
    def ms_fn(**kwargs):
        var_name = controller.variable
        zts_idx = controller.zts_index

        # Choose masked or unmasked statistics based on availability and toggle
        use_masked = cover_mask is not None and controller.mask_cover

        cur_ds_mean = (
            ds_mean_masked if use_masked and ds_mean_masked is not None else ds_mean
        )
        cur_ds_std = (
            ds_std_masked if use_masked and ds_std_masked is not None else ds_std
        )

        if var_name is None or var_name not in cur_ds_mean:
            return hv.Curve([])

        da_m = cur_ds_mean[var_name]
        da_s = cur_ds_std[var_name]

        if "zts" in da_m.dims:
            da_m = da_m.isel(zts=zts_idx)
            da_s = da_s.isel(zts=zts_idx)

        if "time" not in da_m.dims:
            return hv.Curve([])

        # Fix the plot width so it doesn't grow and overlap
        return _meanstd_overlay(da_m, da_s, title=var_name).opts(width=400)

    ms_dmap = hv.DynamicMap(ms_fn, streams=[ms_param_stream]).opts(framewise=True)
    ms_plot = pn.panel(ms_dmap, sizing_mode="stretch_width")

    # --- Horizontal slice DynamicMap (with clim from view) ---
    hz_range_stream = streams.RangeXY()
    hz_param_stream = streams.Params(
        controller,
        parameters=[
            "category",
            "variable",
            "time_index",
            "zts_index",
            "view",
            "auto",
            "trigger",
            "mask_cover",
        ],
    )

    @catchall
    def hz_fn(x_range=None, y_range=None, **kwargs):
        var_name = controller.variable
        t_idx = controller.time_index
        zts_idx = controller.zts_index
        auto = controller.auto
        trigger = controller.trigger

        base_ds = (
            ds_masked
            if (
                cover_mask is not None
                and ds_masked is not None
                and controller.mask_cover
            )
            else ds
        )

        if var_name is None or var_name not in base_ds:
            return hv.Curve([])

        da = base_ds[var_name]
        sel = {}
        if "time" in da.dims:
            sel["time"] = t_idx
        if "zts" in da.dims:
            sel["zts"] = zts_idx

        sliced = da.isel(sel, drop=True)

        remaining = [d for d in sliced.dims]
        if len(remaining) < 2:
            return hv.Curve([])

        xdim = "xt" if "xt" in remaining else remaining[0]
        ydim = "yt" if "yt" in remaining else remaining[1]
        if xdim == ydim:
            ydim = remaining[0] if remaining[0] != xdim else remaining[1]

        vmin, vmax = _compute_slice_clim(
            da, sliced, xdim, ydim, x_range, y_range, auto, trigger
        )

        clim, cmap = determine_clim_and_cmap(vmin, vmax)

        title = var_name
        if "time" in da.dims and "time" in ds.coords:
            t_val = str(ds.time.isel(time=t_idx).values)[:19]
            title += f" @ {t_val}"
        if "zts" in da.dims:
            title += f", zts idx {zts_idx}"

        # Use a fixed width to keep the panel from expanding too wide
        return sliced.hvplot(
            x=xdim,
            y=ydim,
            cmap=cmap,
            clim=clim,
            colorbar=True,
            title=title,
            height=300,
            width=400,
        )

    hz_dmap = hv.DynamicMap(hz_fn, streams=[hz_range_stream, hz_param_stream]).opts(
        framewise=False,
        shared_axes=False,
        axiswise=True,
    )
    hz_range_stream.source = hz_dmap
    hz_plot = pn.panel(hz_dmap, sizing_mode="stretch_width")

    # Toggle between views by swapping Column contents
    plot_area = pn.Column(ms_plot, sizing_mode="stretch_width")

    def _toggle_view(*events):
        hz_range_stream.event(x_range=None, y_range=None)
        if controller.view == "mean/std":
            plot_area.objects = [ms_plot]
        else:
            plot_area.objects = [hz_plot]

    controller.param.watch(_toggle_view, ["view"])

    cat_select = pn.widgets.Select.from_param(
        controller.param.category, name="Category"
    )
    var_select = pn.widgets.Select.from_param(
        controller.param.variable, name="Variable"
    )
    view_select = pn.widgets.RadioButtonGroup.from_param(
        controller.param.view, name="View", button_type="default"
    )
    time_slider = pn.widgets.IntSlider.from_param(
        controller.param.time_index, name="Time index"
    )
    zts_slider = pn.widgets.IntSlider.from_param(
        controller.param.zts_index, name="zts layer"
    )
    zts_slider.visible = "zts" in ds.dims

    auto_checkbox, button_view, button_global = _make_clim_controls(controller)

    # Toggle button to enable/disable cover_slb-based masking
    mask_toggle = pn.widgets.Toggle.from_param(
        controller.param.mask_cover,
        name="Mask where cover_slb = 0",
        button_type="primary",
    )
    # If no mask information is available, disable the toggle
    mask_toggle.disabled = cover_mask is None

    controls = pn.Column(
        cat_select,
        var_select,
        view_select,
        time_slider,
        zts_slider,
        mask_toggle,
        auto_checkbox,
        button_view,
        button_global,
        width=250,
    )

    return pn.Row(plot_area, controls, sizing_mode="stretch_width")


class LSMController(param.Parameterized):
    category = param.ObjectSelector(default=None, objects=[], label="Land use category")
    variable = param.ObjectSelector(default=None, objects=[], label="Variable")
    z_index = param.Integer(default=0, bounds=(0, 0), label="Soil layer index")


def _read_lu_suffixes(ds):
    """Read land-use short names from the lushort char variable."""
    if "lushort" not in ds:
        return []
    raw = ds["lushort"].values  # shape (nlu, str3) or similar
    suffixes = []
    for row in raw:
        if hasattr(row, "tobytes"):
            s = row.tobytes().decode("utf-8", errors="ignore").strip("\x00").strip()
        else:
            s = str(row).strip("\x00").strip()
        if s:
            suffixes.append(s)
    return suffixes


def _parse_lsm_categories(ds):
    """Group LSM variables by land-use suffix (read from lushort) and soil."""
    lu_suffixes = _read_lu_suffixes(ds)
    categories = {s: {} for s in lu_suffixes}
    categories["soil"] = {}

    for v in ds.data_vars:
        if ds[v].dtype.kind in ("S", "U", "O"):
            continue
        matched = False
        for suffix in lu_suffixes:
            if v.endswith(f"_{suffix}"):
                base = v[: -(len(suffix) + 1)]
                categories[suffix][base] = v
                matched = True
                break
        if not matched and "z" in ds[v].dims:
            categories["soil"][v] = v

    return categories


def make_lsm_panel(ds):
    categories = _parse_lsm_categories(ds)

    cat_names = [c for c in categories if categories[c]]  # only non-empty
    controller = LSMController()
    controller.param["category"].objects = cat_names
    controller.category = cat_names[0] if cat_names else None

    default_vars = list(categories[cat_names[0]].keys()) if cat_names else []
    controller.param["variable"].objects = default_vars
    if default_vars:
        controller.variable = default_vars[0]

    if "z" in ds.dims:
        n_z = int(ds.sizes["z"])
        controller.param["z_index"].bounds = (0, max(n_z - 1, 0))

    def _update_vars(event):
        cat = controller.category
        var_list = list(categories.get(cat, {}).keys())
        controller.param["variable"].objects = var_list
        controller.variable = var_list[0] if var_list else None

    controller.param.watch(_update_vars, "category")

    param_stream = streams.Params(
        controller, parameters=["category", "variable", "z_index"]
    )

    @catchall
    def plot_fn(**kwargs):
        cat = controller.category
        var_base = controller.variable
        z_idx = controller.z_index

        if var_base is None:
            return hv.Curve([])

        cat_vars = categories.get(cat, {})
        var_name = cat_vars.get(var_base, var_base)

        if var_name not in ds:
            return hv.Curve([])

        da = ds[var_name]
        if "z" in da.dims:
            da = da.isel(z=z_idx)

        long_name = da.attrs.get("long_name", "")
        title = f"{var_name}: {long_name}" if long_name else var_name

        vmin, vmax = float(da.min(skipna=True)), float(da.max(skipna=True))
        clim, cmap = determine_clim_and_cmap(vmin, vmax)

        return da.hvplot(
            x="x",
            y="y",
            cmap=cmap,
            clim=clim,
            colorbar=True,
            title=title,
            height=300,
            responsive=True,
        )

    dmap = hv.DynamicMap(plot_fn, streams=[param_stream]).opts(
        framewise=False,
        shared_axes=False,
        axiswise=True,
    )

    plot = pn.panel(dmap, sizing_mode="stretch_width")

    cat_select = pn.widgets.Select.from_param(
        controller.param.category, name="Land use category"
    )
    var_select = pn.widgets.Select.from_param(
        controller.param.variable, name="Variable"
    )
    z_slider = pn.widgets.IntSlider.from_param(
        controller.param.z_index, name="Soil layer (z)"
    )
    z_slider.visible = controller.category == "soil"

    def _toggle_z(event):
        z_slider.visible = controller.category == "soil"

    controller.param.watch(_toggle_z, "category")

    controls = pn.Column(cat_select, var_select, z_slider, sizing_mode="stretch_width")

    return pn.Row(plot, controls, sizing_mode="stretch_width")


def make_2d_xy_panel(ds):
    """Simple 2D (x, y) field selector panel."""
    label_to_var = get_label_to_var(ds)
    labels = list(label_to_var.keys())

    if not labels:
        return pn.pane.Markdown("No variables", sizing_mode="stretch_width")

    controller = ClimController(label=labels[0])
    controller.param["label"].objects = labels

    param_stream = streams.Params(controller, parameters=["label"])

    @catchall
    def plot_fn(**kwargs):
        label = controller.label
        if label is None:
            return hv.Curve([])

        var = label_to_var[label]
        da = ds[var]

        vmin, vmax = float(da.min(skipna=True)), float(da.max(skipna=True))
        clim, cmap = determine_clim_and_cmap(vmin, vmax)

        return da.hvplot(
            x="x",
            y="y",
            cmap=cmap,
            clim=clim,
            colorbar=True,
            title=label,
            height=300,
            responsive=True,
        )

    dmap = hv.DynamicMap(plot_fn, streams=[param_stream]).opts(
        framewise=False,
        shared_axes=False,
        axiswise=True,
    )

    plot = pn.panel(dmap, sizing_mode="stretch_width")
    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")
    controls = pn.Column(var_select, sizing_mode="stretch_width")

    return pn.Row(plot, controls, sizing_mode="stretch_width")


def make_init_profile_panel(ds):
    """1D profile panel for init.001.nc (variable vs zh)."""
    # Only keep genuinely 1D variables to avoid crashing on time-dependent
    # fields such as ua_nudge(time, zh).
    full_label_to_var = get_label_to_var(ds)
    label_to_var = {}
    for label, var in full_label_to_var.items():
        da = ds[var]
        if da.ndim == 1:
            label_to_var[label] = var

    labels = list(label_to_var.keys())

    if not labels:
        return pn.pane.Markdown(
            "No 1D profile variables in init file",
            sizing_mode="stretch_width",
        )

    controller = ClimController(label=labels[0])
    controller.param["label"].objects = labels

    param_stream = streams.Params(controller, parameters=["label"])

    @catchall
    def plot_fn(**kwargs):
        label = controller.label
        if label is None:
            return hv.Curve([])

        var = label_to_var[label]
        da = ds[var]

        ydim = "zh" if "zh" in da.dims else list(da.dims)[0]
        z_vals = ds[ydim].values
        x_vals = da.values

        return hv.Curve((x_vals, z_vals), var, ydim).opts(
            xlabel=label,
            ylabel=f"{ydim} [m]",
            title=label,
            height=300,
            responsive=True,
        )

    dmap = hv.DynamicMap(plot_fn, streams=[param_stream]).opts(
        framewise=True,
    )

    plot = pn.panel(dmap, sizing_mode="stretch_width")
    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")
    controls = pn.Column(var_select, sizing_mode="stretch_width")

    return pn.Row(plot, controls, sizing_mode="stretch_width")


def make_tmser_panel(ds):
    """Simple 1D time-series panel for tmser.001.nc.

    Each variable is assumed to be primarily a function of time, and is
    rendered as a 1D curve instead of a time–height heatmap.
    """

    label_to_var = get_label_to_var(ds)
    labels = list(label_to_var.keys())

    if not labels:
        return pn.pane.Markdown("No variables", sizing_mode="stretch_width")

    controller = ClimController(label=labels[0])
    controller.param["label"].objects = labels

    param_stream = streams.Params(controller, parameters=["label"])

    @catchall
    def plot_fn(**kwargs):
        label = controller.label
        if label is None:
            return hv.Curve([])

        var = label_to_var[label]
        da = ds[var]

        # Prefer explicit time dimension when available; otherwise fall back
        # to the first dimension.
        if "time" in da.dims:
            return da.hvplot(
                x="time",
                title=label,
                height=300,
                responsive=True,
            )
        elif da.ndim >= 1:
            dim = da.dims[0]
            return da.hvplot(
                x=dim,
                title=label,
                height=300,
                responsive=True,
            )
        else:
            return hv.Curve([])

    dmap = hv.DynamicMap(plot_fn, streams=[param_stream]).opts(framewise=True)

    plot = pn.panel(dmap, sizing_mode="stretch_width")
    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")
    controls = pn.Column(var_select, sizing_mode="stretch_width")

    return pn.Row(plot, controls, sizing_mode="stretch_width")


def make_app(folder):
    output_path = pathlib.Path(folder)
    ds = _load_datasets(output_path)

    grid = pn.GridSpec(sizing_mode="stretch_width")

    def _p(msg):
        return pn.panel(msg, sizing_mode="stretch_width")

    grid[0, 0] = (
        make_heat_panel_with_clim_controls(ds["prof"])
        if ds["prof"] is not None
        else _p("profiles.001.nc not loaded")
    )
    grid[0, 1] = (
        make_tmser_panel(ds["tmser"])
        if ds["tmser"] is not None
        else _p("tmser.001.nc not loaded")
    )
    grid[1, 0] = (
        make_meanstd_or_horizontal_panel(
            ds["cape_raw"],
            cape,
            slice_dim="time",
            toggle_label="CAPE view",
            ds_lsm=ds["lsm"],
        )
        if ds["cape_raw"] is not None
        else _p("cape.nc not loaded")
    )
    grid[1, 1] = (
        make_meanstd_or_horizontal_panel(
            ds["crosses_raw"],
            crosses,
            slice_dim="time",
            toggle_label="Surfcross view",
            ds_lsm=ds["lsm"],
        )
        if ds["crosses_raw"] is not None
        else _p("surfcross.nc not loaded")
    )
    grid[2, 0] = (
        make_heat_panel_with_clim_controls(ds["sampling"])
        if ds["sampling"] is not None
        else _p("sampling.001.nc not loaded")
    )
    grid[2, 1] = (
        make_heat_panel_with_clim_controls(ds["samptend"])
        if ds["samptend"] is not None
        else _p("samptend.001.nc not loaded")
    )

    grid[3, 0] = (
        make_slice_panel(ds["crossxz"], "yt")
        if ds["crossxz"] is not None
        else _p("crossxz.nc not loaded")
    )
    grid[3, 1] = (
        make_slice_panel(ds["crossyz"], "xt")
        if ds["crossyz"] is not None
        else _p("crossyz.nc not loaded")
    )
    grid[4, 0] = (
        make_slice_panel(ds["crossxy"], "zt")
        if ds["crossxy"] is not None
        else _p("crossxy.nc not loaded")
    )
    grid[4, 1] = (
        make_fielddump_slice_panel(ds["fielddump"])
        if ds["fielddump"] is not None
        else _p("fielddump.nc not loaded")
    )
    grid[5, 0] = (
        make_fielddump_slice_panel(ds["dump"])
        if ds["dump"] is not None
        else _p("all_dump.nc not loaded")
    )
    grid[5, 1] = (
        make_slice_panel(ds["openboundaries"], "time")
        if ds["openboundaries"] is not None
        else _p("openboundaries.inp.*.nc not loaded")
    )

    radfield_pane = (
        make_meanstd_or_horizontal_panel(
            ds["radfield_raw"],
            cape,
            slice_dim="time",
            toggle_label="Radfield view",
            ds_lsm=ds["lsm"],
        )
        if ds["radfield_raw"] is not None
        else _p("radfield.nc not loaded")
    )

    grid[6, 0] = (
        pn.Column(
            pn.pane.Markdown("### lsm.inp_001.nc"),
            make_lsm_panel(ds["lsm"]),
            sizing_mode="stretch_width",
        )
        if ds["lsm"] is not None
        else _p("lsm.inp_001.nc not loaded")
    )
    grid[6, 1] = (
        pn.Column(
            pn.pane.Markdown("### inslurb.001.nc"),
            make_2d_xy_panel(ds["inslurb"]),
            sizing_mode="stretch_width",
        )
        if ds["inslurb"] is not None
        else _p("inslurb.001.nc not loaded")
    )
    grid[7, 0] = (
        pn.Column(
            pn.pane.Markdown("### init.001.nc"),
            make_init_profile_panel(ds["init"]),
            sizing_mode="stretch_width",
        )
        if ds["init"] is not None
        else _p("init.001.nc not loaded")
    )

    if ds["forcings"] is not None:
        ds["forcings"]["uv_timedep"] = np.sqrt(
            ds["forcings"]["ug_timedep"] ** 2 + ds["forcings"]["vg_timedep"] ** 2
        )
        grid[7, 1] = pn.Column(
            pn.pane.Markdown("### forcings.001.nc"),
            make_heat_panel_with_clim_controls(ds["forcings"]),
            sizing_mode="stretch_width",
        )
    else:
        grid[7, 1] = _p("forcings.001.nc not loaded")

    grid[8, 0] = (
        pn.Column(
            pn.pane.Markdown("### slrbcross.nc"),
            make_slrbcross_panel(ds["slrbcross"], ds["lsm"]),
            sizing_mode="stretch_width",
        )
        if ds["slrbcross"] is not None
        else _p("slrbcross.nc not loaded")
    )
    grid[8, 1] = radfield_pane

    header = pn.Row(
        pn.pane.Markdown(f"# {output_path.resolve()}"),
        sizing_mode="stretch_width",
    )

    return pn.Column(
        header,
        grid,
        sizing_mode="stretch_width",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DALES viewer")
    parser.add_argument(
        "folder", nargs="?", default=".", help="Path to DALES output folder"
    )
    parser.add_argument("--port", type=int, default=5007)
    args = parser.parse_args()

    pn.config.console_output = "accumulate"

    import traceback, sys

    def _print_exception(ex):
        traceback.print_exception(type(ex), ex, ex.__traceback__, file=sys.stderr)

    pn.extension(exception_handler=_print_exception)

    app = make_app(args.folder)
    pn.serve(app, show=True, port=args.port, admin=True)
