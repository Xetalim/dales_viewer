from controllers import _select_plot_dims
from helpers import (
    _build_backend_bounds,
    _compute_slice_clim,
    _make_clim_controls,
    catchall,
    determine_clim_and_cmap,
    get_label_to_var,
)

import holoviews as hv
import panel as pn
import param
from holoviews import streams


class HorizontalController(param.Parameterized):
    label = param.ObjectSelector(default=None, objects=[], label="Variable")
    time_index = param.Integer(default=0, bounds=(0, 0), label="Time index")
    auto = param.Boolean(default=True, doc="Automatically set clim from current view")
    trigger = param.Integer(default=0, doc="Manual trigger to recompute clim from view")
    symmetric_cmap = param.Boolean(
        default=False,
        doc="Force symmetric color limits around 0",
    )


def make_horizontal_panel(
    ds_raw, stat_fn, slice_dim="time", toggle_label="View", ds_lsm=None
):
    """Panel showing raw horizontal fields without masking or mean/std.

    ds_raw: original dataset with horizontal dimensions (e.g. xt, yt) and a slice_dim
            (typically "time") that selects individual horizontal fields.
    stat_fn: unused — kept for API compatibility.
    slice_dim: dimension in ds_raw along which to slice to obtain horizontal maps.
    toggle_label: unused — kept for API compatibility.
    ds_lsm: unused — kept for API compatibility.
    """

    label_to_var = get_label_to_var(ds_raw)
    labels = list(label_to_var.keys())

    if not labels:
        return pn.pane.Markdown("No variables", sizing_mode="stretch_width")

    controller = HorizontalController(label=labels[0])
    controller.param["label"].objects = labels

    if slice_dim in ds_raw.dims:
        n_time = int(ds_raw.sizes[slice_dim])
        controller.param["time_index"].bounds = (0, max(n_time - 1, 0))

    hz_range_stream = streams.RangeXY()
    hz_param_stream = streams.Params(
        controller,
        parameters=[
            "label",
            "time_index",
            "auto",
            "trigger",
            "symmetric_cmap",
        ],
    )

    @catchall
    def hz_fn(x_range=None, y_range=None, **kwargs):
        label = controller.label
        if label is None:
            return hv.Curve([])

        var_name = label_to_var.get(label)
        if var_name is None or var_name not in ds_raw:
            return hv.Curve([])

        da = ds_raw[var_name]
        if slice_dim not in da.dims:
            return hv.Curve([])

        slice_sel = {slice_dim: controller.time_index}
        other_dims = [d for d in da.dims if d not in slice_sel]
        if len(other_dims) != 2:
            return hv.Curve([])

        xdim, ydim = _select_plot_dims(other_dims)
        if xdim is None or ydim is None:
            return hv.Curve([])

        sliced = da.isel(slice_sel, drop=True)

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

        title = label
        if slice_dim in da.dims:
            if slice_dim in ds_raw.coords:
                slice_val = (
                    ds_raw[slice_dim].isel({slice_dim: controller.time_index}).values
                )
                title += f" @ {slice_dim}={str(slice_val)[:19]}"
            else:
                title += f" @ {slice_dim} index {controller.time_index}"

        plot = sliced.hvplot(
            x=xdim,
            y=ydim,
            cmap=cmap,
            clim=clim,
            colorbar=True,
            title=title,
            height=300,
            responsive=True,
        )

        backend_opts = _build_backend_bounds(ds_raw, xdim, ydim)
        return plot.opts(backend_opts=backend_opts)

    hz_dmap = hv.DynamicMap(hz_fn, streams=[hz_range_stream, hz_param_stream]).opts(
        framewise=False,
        shared_axes=False,
        axiswise=True,
    )
    hz_range_stream.source = hz_dmap
    hz_plot = pn.panel(hz_dmap, sizing_mode="stretch_width")

    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")
    time_slider = pn.widgets.IntSlider.from_param(
        controller.param.time_index,
        name=f"{slice_dim} index",
    )
    time_slider.visible = slice_dim in ds_raw.dims

    auto_checkbox, button_view, button_global = _make_clim_controls(controller)

    sym_toggle = pn.widgets.Toggle.from_param(
        controller.param.symmetric_cmap,
        name="Symmetric clim around 0",
        button_type="primary",
    )

    controls = pn.Column(
        var_select,
        time_slider,
        sym_toggle,
        auto_checkbox,
        button_view,
        button_global,
        width=250,
    )

    return pn.Row(hz_plot, controls, sizing_mode="stretch_width")
