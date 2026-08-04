from _meanstd_overlay import _meanstd_overlay
from helpers import (
    _make_clim_controls,
    append_indexed_dim_to_title,
    apply_horizontal_mask,
    build_cover_mask,
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


class MeanStdHorizontalController(param.Parameterized):
    label = param.ObjectSelector(default=None, objects=[], label="Variable")
    view = param.ObjectSelector(
        default="Mean/std over horizontal",
        objects=["Mean/std over horizontal", "Horizontal field"],
        label="View",
    )
    time_index = param.Integer(default=0, bounds=(0, 0), label="Time index")
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
    symmetric_cmap = param.Boolean(
        default=False,
        doc="Force symmetric color limits around 0 for horizontal view",
    )


def make_meanstd_or_horizontal_panel(
    ds_raw, stat_fn, slice_dim="time", toggle_label="View", ds_lsm=None
):
    """Panel to switch between mean/std-over-horizontal and raw horizontal view.

    ds_raw: original dataset with horizontal dimensions (e.g. xt, yt) and a slice_dim
            (typically "time") that selects individual horizontal fields.
    stat_fn: function that takes ds_raw and returns (ds_mean, ds_std), e.g. cape or
             crosses.
    slice_dim: dimension in ds_raw along which to slice to obtain horizontal maps.
    toggle_label: widget label for the view selector.
    """

    label_to_var = get_label_to_var(ds_raw)
    labels = list(label_to_var.keys())

    if not labels:
        return pn.pane.Markdown("No variables", sizing_mode="stretch_width")

    controller = MeanStdHorizontalController(label=labels[0])
    controller.param["label"].objects = labels

    if slice_dim in ds_raw.dims:
        n_time = int(ds_raw.sizes[slice_dim])
        controller.param["time_index"].bounds = (0, max(n_time - 1, 0))

    cover_mask = build_cover_mask(ds_raw, ds_lsm)
    cover_mask_positive = build_cover_mask(ds_raw, ds_lsm, keep_zero=True)
    ds_masked = apply_horizontal_mask(ds_raw, cover_mask)
    ds_masked_positive = apply_horizontal_mask(ds_raw, cover_mask_positive)

    ds_mean, ds_std = stat_fn(ds_raw)
    ds_mean_masked = None
    ds_std_masked = None
    ds_mean_masked_positive = None
    ds_std_masked_positive = None
    if ds_masked is not None:
        ds_mean_masked, ds_std_masked = stat_fn(ds_masked)
    if ds_masked_positive is not None:
        ds_mean_masked_positive, ds_std_masked_positive = stat_fn(ds_masked_positive)

    def _ensure_exclusive(event):
        if event.new:
            if event.name == "mask_cover":
                controller.mask_cover_positive = False
            elif event.name == "mask_cover_positive":
                controller.mask_cover = False

    controller.param.watch(_ensure_exclusive, ["mask_cover", "mask_cover_positive"])

    ms_param_stream = streams.Params(
        controller,
        parameters=["label", "view", "mask_cover", "mask_cover_positive"],
    )

    @catchall
    def ms_fn(**_kwargs):
        label = controller.label
        if controller.mask_cover and ds_mean_masked is not None:
            current_mean = ds_mean_masked
            current_std = ds_std_masked
        elif controller.mask_cover_positive and ds_mean_masked_positive is not None:
            current_mean = ds_mean_masked_positive
            current_std = ds_std_masked_positive
        else:
            current_mean = ds_mean
            current_std = ds_std

        if label is None:
            return hv.Curve([])

        var_name = label_to_var.get(label)
        if (
            var_name is None
            or var_name not in current_mean
            or var_name not in current_std
        ):
            return hv.Curve([])

        plot = _meanstd_overlay(
            current_mean[var_name],
            current_std[var_name],
            title=label,
        )

        if "time" in current_mean[var_name].dims:
            return plot.opts(
                backend_opts={
                    "x_range.bounds": (
                        current_mean.time.min(skipna=True).values,
                        current_mean.time.max(skipna=True).values,
                    )
                }
            )

        return plot

    ms_dmap = hv.DynamicMap(ms_fn, streams=[ms_param_stream]).opts(framewise=False)
    ms_plot = pn.panel(ms_dmap, sizing_mode="stretch_width")

    hz_range_stream = streams.RangeXY()
    hz_param_stream = streams.Params(
        controller,
        parameters=[
            "label",
            "time_index",
            "view",
            "auto",
            "trigger",
            "mask_cover",
            "mask_cover_positive",
            "symmetric_cmap",
        ],
    )

    @catchall
    def hz_fn(x_range=None, y_range=None, **_kwargs):
        label = controller.label
        if controller.mask_cover and ds_masked is not None:
            base_ds = ds_masked
        elif controller.mask_cover_positive and ds_masked_positive is not None:
            base_ds = ds_masked_positive
        else:
            base_ds = ds_raw

        if label is None:
            return hv.Curve([])

        var_name = label_to_var.get(label)
        if var_name is None or var_name not in base_ds:
            return hv.Curve([])

        da = base_ds[var_name]
        if slice_dim not in da.dims:
            return hv.Curve([])

        sliced, xdim, ydim, _slice_sel = slice_to_2d(
            da, **{slice_dim: controller.time_index}
        )
        if xdim is None or ydim is None:
            return hv.Curve([])
        title = append_indexed_dim_to_title(
            label,
            base_ds,
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
            bounds_source=base_ds,
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
        if controller.view == "Mean/std over horizontal":
            plot_area.objects = [ms_plot]
        else:
            plot_area.objects = [hz_plot]

    controller.param.watch(_toggle_view, ["view"])

    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")
    mode_toggle = pn.widgets.RadioButtonGroup.from_param(
        controller.param.view,
        name=toggle_label,
        button_type="default",
    )
    time_slider = pn.widgets.IntSlider.from_param(
        controller.param.time_index,
        name=f"{slice_dim} index",
    )
    time_slider.visible = slice_dim in ds_raw.dims

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
    sym_toggle = pn.widgets.Toggle.from_param(
        controller.param.symmetric_cmap,
        name="Symmetric clim around 0",
        button_type="primary",
    )
    mask_toggle.disabled = cover_mask is None
    mask_positive_toggle.disabled = cover_mask_positive is None

    controls = pn.Column(
        var_select,
        mode_toggle,
        time_slider,
        mask_toggle,
        mask_positive_toggle,
        sym_toggle,
        auto_checkbox,
        button_view,
        button_global,
        sizing_mode="stretch_width",
    )

    return make_plot_with_controls_layout(plot_area, controls)
