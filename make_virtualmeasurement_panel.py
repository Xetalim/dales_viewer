"""Panel for virtualmeasurement.<xi>.<yj>.nc time-series files.

Each file contains 1-D variables (dimension: time only) for a single grid
point.  This panel exposes a location selector (when multiple files are
present) and a variable selector, and renders a time-series curve.
"""

import param
import holoviews as hv
import panel as pn
from holoviews import streams
import hvplot.xarray  # noqa: F401
from helpers import get_label_to_var, make_plot_with_controls_layout


class _VMController(param.Parameterized):
    file_label = param.ObjectSelector(default=None, objects=[], label="Location")
    variable = param.ObjectSelector(default=None, objects=[], label="Variable")


def make_virtualmeasurement_panel(datasets):
    """Panel for a collection of virtualmeasurement.X.Y.nc datasets.

    Parameters
    ----------
    datasets : dict[str, xr.Dataset]
        Mapping of location label (derived from locx/locy global attributes or
        file stem) to the opened dataset.
    """
    if not datasets:
        return pn.pane.Markdown(
            "No virtualmeasurement files found.", sizing_mode="stretch_width"
        )

    file_labels = list(datasets.keys())
    controller = _VMController()
    controller.param["file_label"].objects = file_labels
    controller.file_label = file_labels[0]

    def _current_label_to_var():
        ds = datasets.get(controller.file_label)
        return get_label_to_var(ds) if ds is not None else {}

    label_to_var = _current_label_to_var()
    var_labels = list(label_to_var.keys())
    controller.param["variable"].objects = var_labels
    controller.variable = var_labels[0] if var_labels else None

    def _update_vars(_event):
        ltv = _current_label_to_var()
        new_labels = list(ltv.keys())
        controller.param["variable"].objects = new_labels
        controller.variable = new_labels[0] if new_labels else None

    controller.param.watch(_update_vars, "file_label")

    param_stream = streams.Params(controller, parameters=["file_label", "variable"])

    def plot_fn(**_kwargs):
        ds = datasets.get(controller.file_label)
        if ds is None:
            return hv.Curve([])

        ltv = get_label_to_var(ds)
        var_label = controller.variable
        if var_label is None or var_label not in ltv:
            return hv.Curve([])

        var = ltv[var_label]
        da = ds[var]
        units = da.attrs.get("units", "")

        # Build a title that includes the physical location when available.
        locx = ds.attrs.get("locx", None)
        locy = ds.attrs.get("locy", None)
        if locx is not None and locy is not None:
            loc_str = f"  (x={float(locx):.1f} m, y={float(locy):.1f} m)"
        else:
            loc_str = ""
        title = var_label + loc_str

        # Keep this behavior aligned with the known-working tmser panel.
        if "time" in da.dims:
            return da.hvplot(
                x="time",
                title=title,
                ylabel=units or var,
                height=300,
                responsive=True,
            )

        # Fallback: plot against the first available dimension.
        dim = da.dims[0] if da.dims else None
        if dim is None:
            return hv.Curve([])
        return da.hvplot(
            x=dim,
            title=title,
            ylabel=units or var,
            height=300,
            responsive=True,
        )

    dmap = hv.DynamicMap(plot_fn, streams=[param_stream]).opts(framewise=True)
    plot = pn.panel(dmap, sizing_mode="stretch_width")

    file_select = pn.widgets.Select.from_param(
        controller.param.file_label, name="Location"
    )
    var_select = pn.widgets.Select.from_param(
        controller.param.variable, name="Variable"
    )
    controls = pn.Column(file_select, var_select, sizing_mode="stretch_width")

    return make_plot_with_controls_layout(plot, controls)
