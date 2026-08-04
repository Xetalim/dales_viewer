from helpers import (
    _make_clim_controls,
    append_indexed_dim_to_title,
    catchall,
    get_label_to_var,
    make_plot_with_controls_layout,
    plot_2d_heatmap,
    slice_to_2d,
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
    _ = stat_fn, toggle_label, ds_lsm

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
    def hz_fn(x_range=None, y_range=None, **_kwargs):
        label = controller.label
        if label is None:
            return hv.Curve([])

        var_name = label_to_var.get(label)
        if var_name is None or var_name not in ds_raw:
            return hv.Curve([])

        da = ds_raw[var_name]
        if slice_dim not in da.dims:
            return hv.Curve([])

        sliced, xdim, ydim, _slice_sel = slice_to_2d(
            da, **{slice_dim: controller.time_index}
        )
        if xdim is None or ydim is None:
            return hv.Curve([])
        title = append_indexed_dim_to_title(
            label,
            ds_raw,
            slice_dim,
            controller.time_index,
            separator=" @ ",
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
            symmetric_cmap=controller.symmetric_cmap,
            bounds_source=ds_raw,
        )

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
        sizing_mode="stretch_width",
    )

    return make_plot_with_controls_layout(hz_plot, controls)
