from _meanstd_overlay import _meanstd_overlay
from helpers import get_label_to_var, catchall
from controllers import ClimController


import holoviews as hv
import panel as pn
from holoviews import streams


def make_meanstd_panel_with_range_controls(ds_mean, ds_std):
    label_to_var = get_label_to_var(ds_mean)

    labels = list(label_to_var.keys())
    controller = ClimController(label=labels[0] if labels else None)
    controller.param["label"].objects = labels

    param_stream = streams.Params(controller, parameters=["label"])

    @catchall
    def ms_fn(label=None):
        if label is None:
            return hv.Curve([])

        var = label_to_var[label]
        return _meanstd_overlay(
            ds_mean[var],
            ds_std[var],
            title=label,
        ).opts(
            backend_opts={
                "x_range.bounds": (
                    ds_mean.time.min(skipna=True).values,
                    ds_mean.time.max(skipna=True).values,
                )
            }
        )

    ms_dmap = hv.DynamicMap(
        ms_fn,
        streams=[param_stream],
    ).opts(framewise=False)

    plot = pn.panel(ms_dmap, sizing_mode="stretch_width")

    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")

    controls = pn.Column(var_select, sizing_mode="stretch_width")

    return pn.Row(plot, controls, sizing_mode="stretch_width")
