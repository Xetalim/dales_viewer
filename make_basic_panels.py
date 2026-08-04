import holoviews as hv
import panel as pn
from holoviews import streams

from controllers import ClimController
from helpers import (
    catchall,
    get_label_to_var,
    make_plot_with_controls_layout,
    plot_2d_heatmap,
)


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
    def plot_fn(**_kwargs):
        label = controller.label
        if label is None:
            return hv.Curve([])

        var = label_to_var[label]
        da = ds[var]
        return plot_2d_heatmap(
            da,
            xdim="x",
            ydim="y",
            title=label,
            bounds_source=ds,
        )

    dmap = hv.DynamicMap(plot_fn, streams=[param_stream]).opts(
        framewise=False,
        shared_axes=False,
        axiswise=True,
    )

    plot = pn.panel(dmap, sizing_mode="stretch_width")
    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")
    controls = pn.Column(var_select, sizing_mode="stretch_width")

    return make_plot_with_controls_layout(plot, controls)


def make_init_profile_panel(ds):
    """Profile panel for init-like data.

    Supports both:
    - 1D vertical profiles (rendered as line plots)
    - 2D time-vertical fields (rendered as heatmaps)
    """

    def _detect_vertical_dim(da):
        for dim in ("zh", "zt", "zm", "zf", "zts"):
            if dim in da.dims:
                return dim
        return None

    full_label_to_var = get_label_to_var(ds)
    label_to_var = {}
    for label, var in full_label_to_var.items():
        da = ds[var]
        vdim = _detect_vertical_dim(da)
        if vdim is None:
            continue

        is_1d_profile = da.ndim == 1 and vdim in da.dims
        is_time_vertical = da.ndim == 2 and "time" in da.dims and vdim in da.dims

        if is_1d_profile or is_time_vertical:
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
    def plot_fn(**_kwargs):
        label = controller.label
        if label is None:
            return hv.Curve([])

        var = label_to_var[label]
        da = ds[var]
        ydim = _detect_vertical_dim(da)
        if ydim is None:
            return hv.Curve([])

        if da.ndim == 2 and "time" in da.dims and ydim in da.dims:
            return plot_2d_heatmap(
                da,
                xdim="time",
                ydim=ydim,
                title=label,
                bounds_source=ds,
            )

        z_vals = ds[ydim].values
        x_vals = da.values

        return hv.Curve((x_vals, z_vals), var, ydim).opts(
            xlabel=label,
            ylabel=f"{ydim} [m]",
            title=label,
            height=300,
            responsive=True,
        )

    dmap = hv.DynamicMap(plot_fn, streams=[param_stream]).opts(framewise=True)

    plot = pn.panel(dmap, sizing_mode="stretch_width")
    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")
    controls = pn.Column(var_select, sizing_mode="stretch_width")

    return make_plot_with_controls_layout(plot, controls)


def make_tmser_panel(ds):
    """Simple 1D time-series panel for tmser.001.nc."""
    label_to_var = get_label_to_var(ds)
    labels = list(label_to_var.keys())

    if not labels:
        return pn.pane.Markdown("No variables", sizing_mode="stretch_width")

    controller = ClimController(label=labels[0])
    controller.param["label"].objects = labels

    param_stream = streams.Params(controller, parameters=["label"])

    @catchall
    def plot_fn(**_kwargs):
        label = controller.label
        if label is None:
            return hv.Curve([])

        var = label_to_var[label]
        da = ds[var]

        if "time" in da.dims:
            return da.hvplot(
                x="time",
                title=label,
                height=300,
                responsive=True,
            )
        if da.ndim >= 1:
            dim = da.dims[0]
            return da.hvplot(
                x=dim,
                title=label,
                height=300,
                responsive=True,
            )
        return hv.Curve([])

    dmap = hv.DynamicMap(plot_fn, streams=[param_stream]).opts(framewise=True)

    plot = pn.panel(dmap, sizing_mode="stretch_width")
    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")
    controls = pn.Column(var_select, sizing_mode="stretch_width")

    return make_plot_with_controls_layout(plot, controls)
