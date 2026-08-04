from _meanstd_overlay import _meanstd_overlay
from helpers import (
    _make_clim_controls,
    append_indexed_dim_to_title,
    apply_horizontal_mask,
    build_cover_mask,
    catchall,
    make_plot_with_controls_layout,
    plot_2d_heatmap,
    slice_to_2d,
)

import holoviews as hv
import panel as pn
import param
from holoviews import streams

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
    mask_cover_positive = param.Boolean(
        default=False,
        doc="Mask (x, y) where cover_slb > 0 in lsm.inp_001.nc",
    )


def make_slrbcross_panel(ds, ds_lsm=None):
    """Panel for slrbcross.nc with category filtering, mean/std and masking."""

    categories = _parse_slrb_categories(ds)
    cat_names = list(categories.keys())

    if not cat_names:
        return pn.pane.Markdown("No variables", sizing_mode="stretch_width")

    cover_mask = build_cover_mask(ds, ds_lsm)
    cover_mask_positive = build_cover_mask(ds, ds_lsm, keep_zero=True)

    horiz_dims = [d for d in ("xt", "yt") if d in ds.dims]
    ds_mean = ds.mean(dim=horiz_dims, keep_attrs=True) if horiz_dims else ds
    ds_std = ds.std(dim=horiz_dims, keep_attrs=True) if horiz_dims else None

    ds_masked = None
    ds_masked_positive = None
    ds_mean_masked = None
    ds_std_masked = None
    ds_mean_masked_positive = None
    ds_std_masked_positive = None
    if cover_mask is not None and horiz_dims:
        ds_masked = apply_horizontal_mask(ds, cover_mask)
        ds_mean_masked = ds_masked.mean(dim=horiz_dims, keep_attrs=True, skipna=True)
        ds_std_masked = ds_masked.std(dim=horiz_dims, keep_attrs=True, skipna=True)
    if cover_mask_positive is not None and horiz_dims:
        ds_masked_positive = apply_horizontal_mask(ds, cover_mask_positive)
        ds_mean_masked_positive = ds_masked_positive.mean(
            dim=horiz_dims, keep_attrs=True, skipna=True
        )
        ds_std_masked_positive = ds_masked_positive.std(
            dim=horiz_dims, keep_attrs=True, skipna=True
        )

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

    def _ensure_exclusive(event):
        if event.new:
            if event.name == "mask_cover":
                controller.mask_cover_positive = False
            elif event.name == "mask_cover_positive":
                controller.mask_cover = False

    controller.param.watch(_ensure_exclusive, ["mask_cover", "mask_cover_positive"])

    def _update_vars(_event):
        cat = controller.category
        var_list = categories.get(cat, [])
        controller.param["variable"].objects = var_list
        controller.variable = var_list[0] if var_list else None

    controller.param.watch(_update_vars, "category")

    # --- Mean/std DynamicMap ---
    ms_param_stream = streams.Params(
        controller,
        parameters=[
            "category",
            "variable",
            "zts_index",
            "view",
            "mask_cover",
            "mask_cover_positive",
        ],
    )

    @catchall
    def ms_fn(**_kwargs):
        var_name = controller.variable
        zts_idx = controller.zts_index

        if controller.mask_cover and ds_mean_masked is not None:
            cur_ds_mean = ds_mean_masked
            cur_ds_std = ds_std_masked
        elif controller.mask_cover_positive and ds_mean_masked_positive is not None:
            cur_ds_mean = ds_mean_masked_positive
            cur_ds_std = ds_std_masked_positive
        else:
            cur_ds_mean = ds_mean
            cur_ds_std = ds_std

        if var_name is None or var_name not in cur_ds_mean:
            return hv.Curve([])

        da_m = cur_ds_mean[var_name]
        da_s = cur_ds_std[var_name]

        if "zts" in da_m.dims:
            da_m = da_m.isel(zts=zts_idx)
            da_s = da_s.isel(zts=zts_idx)

        if "time" not in da_m.dims:
            return hv.Curve([])

        return _meanstd_overlay(da_m, da_s, title=var_name)

    ms_dmap = hv.DynamicMap(ms_fn, streams=[ms_param_stream]).opts(framewise=True)
    ms_plot = pn.panel(ms_dmap, sizing_mode="stretch_width")

    # --- Horizontal slice DynamicMap ---
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
            "mask_cover_positive",
        ],
    )

    @catchall
    def hz_fn(x_range=None, y_range=None, **_kwargs):
        var_name = controller.variable
        t_idx = controller.time_index
        zts_idx = controller.zts_index

        if controller.mask_cover and ds_masked is not None:
            base_ds = ds_masked
        elif controller.mask_cover_positive and ds_masked_positive is not None:
            base_ds = ds_masked_positive
        else:
            base_ds = ds

        if var_name is None or var_name not in base_ds:
            return hv.Curve([])

        da = base_ds[var_name]
        sliced, xdim, ydim, _sel = slice_to_2d(da, time=t_idx, zts=zts_idx)
        if xdim is None or ydim is None:
            return hv.Curve([])

        title = var_name
        if "time" in da.dims:
            title = append_indexed_dim_to_title(
                title,
                ds,
                "time",
                t_idx,
                separator=" @ ",
                use_coord=True,
                formatter=lambda value: str(value)[:19],
            )
        if "zts" in da.dims:
            title = append_indexed_dim_to_title(
                title,
                ds,
                "zts",
                zts_idx,
                separator=", ",
                use_coord=False,
            )

        return plot_2d_heatmap(
            sliced,
            xdim=xdim,
            ydim=ydim,
            title=title,
            full_da=da,
            x_range=x_range,
            y_range=y_range,
            auto=controller.auto,
            trigger=controller.trigger,
            bounds_source=base_ds,
            # responsive=True,
        )

    hz_dmap = hv.DynamicMap(hz_fn, streams=[hz_range_stream, hz_param_stream]).opts(
        framewise=False,
        shared_axes=False,
        axiswise=True,
    )
    hz_range_stream.source = hz_dmap
    hz_plot = pn.panel(hz_dmap, sizing_mode="stretch_width")

    plot_area = pn.Column(ms_plot, sizing_mode="stretch_width")

    def _toggle_view(*_events):
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

    mask_toggle = pn.widgets.Toggle.from_param(
        controller.param.mask_cover,
        name="Mask where cover_slb = 0",
        button_type="primary",
    )
    mask_positive_toggle = pn.widgets.Toggle.from_param(
        controller.param.mask_cover_positive,
        name="Mask where cover_slb > 0",
        button_type="warning",
    )
    mask_toggle.disabled = cover_mask is None
    mask_positive_toggle.disabled = cover_mask_positive is None

    controls = pn.Column(
        cat_select,
        var_select,
        view_select,
        time_slider,
        zts_slider,
        mask_toggle,
        mask_positive_toggle,
        auto_checkbox,
        button_view,
        button_global,
        sizing_mode="stretch_width",
    )

    return make_plot_with_controls_layout(plot_area, controls)
