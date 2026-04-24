from controllers import _select_plot_dims
from helpers import (
    _build_backend_bounds,
    _compute_slice_clim,
    _make_clim_controls,
    catchall,
    determine_clim_and_cmap,
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


class SlrbHorizontalController(param.Parameterized):
    category = param.ObjectSelector(default=None, objects=[], label="Category")
    variable = param.ObjectSelector(default=None, objects=[], label="Variable")
    time_index = param.Integer(default=0, bounds=(0, 0), label="Time index")
    zts_index = param.Integer(default=0, bounds=(0, 0), label="zts layer")
    auto = param.Boolean(default=True, doc="Automatically set clim from current view")
    trigger = param.Integer(default=0, doc="Manual trigger to recompute clim from view")
    symmetric_cmap = param.Boolean(
        default=False,
        doc="Force symmetric color limits around 0",
    )


def make_slrbcross_panel(ds, ds_lsm=None):
    """Panel for slrbcross.nc — horizontal view only, no masking or mean/std."""

    categories = _parse_slrb_categories(ds)
    cat_names = list(categories.keys())

    if not cat_names:
        return pn.pane.Markdown("No variables", sizing_mode="stretch_width")

    controller = SlrbHorizontalController()
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

    hz_range_stream = streams.RangeXY()
    hz_param_stream = streams.Params(
        controller,
        parameters=[
            "category",
            "variable",
            "time_index",
            "zts_index",
            "auto",
            "trigger",
            "symmetric_cmap",
        ],
    )

    @catchall
    def hz_fn(x_range=None, y_range=None, **kwargs):
        var_name = controller.variable
        t_idx = controller.time_index
        zts_idx = controller.zts_index

        if var_name is None or var_name not in ds:
            return hv.Curve([])

        da = ds[var_name]
        sel = {}
        if "time" in da.dims:
            sel["time"] = t_idx
        if "zts" in da.dims:
            sel["zts"] = zts_idx

        sliced = da.isel(sel, drop=True)

        remaining = list(sliced.dims)
        if len(remaining) < 2:
            return hv.Curve([])

        xdim = "xt" if "xt" in remaining else remaining[0]
        ydim = "yt" if "yt" in remaining else remaining[1]
        if xdim == ydim:
            ydim = remaining[0] if remaining[0] != xdim else remaining[1]

        vmin, vmax = _compute_slice_clim(
            da,
            sliced,
            xdim,
            ydim,
            x_range,
            y_range,
            controller.auto,
            controller.trigger,
        )

        if controller.symmetric_cmap:
            vmax_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vmax_abs, vmax_abs

        clim, cmap = determine_clim_and_cmap(vmin, vmax)

        title = var_name
        if "time" in da.dims and "time" in ds.coords:
            t_val = str(ds.time.isel(time=t_idx).values)[:19]
            title += f" @ {t_val}"
        if "zts" in da.dims:
            title += f", zts idx {zts_idx}"

        plot = sliced.hvplot(
            x=xdim,
            y=ydim,
            cmap=cmap,
            clim=clim,
            colorbar=True,
            title=title,
            height=300,
            width=400,
        )

        backend_opts = _build_backend_bounds(ds, xdim, ydim)
        return plot.opts(backend_opts=backend_opts)

    hz_dmap = hv.DynamicMap(hz_fn, streams=[hz_range_stream, hz_param_stream]).opts(
        framewise=False,
        shared_axes=False,
        axiswise=True,
    )
    hz_range_stream.source = hz_dmap
    hz_plot = pn.panel(hz_dmap, sizing_mode="stretch_width")

    cat_select = pn.widgets.Select.from_param(
        controller.param.category, name="Category"
    )
    var_select = pn.widgets.Select.from_param(
        controller.param.variable, name="Variable"
    )
    time_slider = pn.widgets.IntSlider.from_param(
        controller.param.time_index, name="Time index"
    )
    zts_slider = pn.widgets.IntSlider.from_param(
        controller.param.zts_index, name="zts layer"
    )
    zts_slider.visible = "zts" in ds.dims

    auto_checkbox, button_view, button_global = _make_clim_controls(controller)

    sym_toggle = pn.widgets.Toggle.from_param(
        controller.param.symmetric_cmap,
        name="Symmetric clim around 0",
        button_type="primary",
    )

    controls = pn.Column(
        cat_select,
        var_select,
        time_slider,
        zts_slider,
        sym_toggle,
        auto_checkbox,
        button_view,
        button_global,
        width=250,
    )

    return pn.Row(hz_plot, controls, sizing_mode="stretch_width")
