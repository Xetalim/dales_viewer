from helpers import get_label_to_var, catchall
from controllers import ClimController


import holoviews as hv
import panel as pn
from holoviews import streams


def make_1d_panel_with_range_controls(ds):
    label_to_var = get_label_to_var(ds)

    labels = list(label_to_var.keys())
    controller = ClimController(label=labels[0] if labels else None)
    controller.param["label"].objects = labels

    # Only react to variable selection; no auto y-range or rescaling
    param_stream = streams.Params(controller, parameters=["label"])

    @catchall
    def ts_fn(label=None):
        if label is None:
            return hv.Curve([])

        var = label_to_var[label]
        full_da = ds[var]

        curve = full_da.hvplot(
            x="time", title=label, margin=100, height=300, responsive=True
        )
        return curve.opts(
            backend_opts={
                "x_range.bounds": (
                    float(ds.time.min(skipna=True).values),
                    float(ds.time.max(skipna=True).values),
                )
            }
        )

    ts_dmap = hv.DynamicMap(
        ts_fn,
        streams=[param_stream],
    ).opts(framewise=True)

    plot = pn.panel(ts_dmap, sizing_mode="stretch_width")

    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")

    controls = pn.Column(var_select, sizing_mode="stretch_width")

    return pn.Row(plot, controls, sizing_mode="stretch_width")
