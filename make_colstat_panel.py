"""Panel for colstat.<xi>.<yj>.nc vertical-profile files.

Each file contains 2-D variables (dimensions: time × zf or time × zh) for a
single grid point.  This panel exposes a location selector, a variable
selector, and two view modes:

  heatmap  — time × height colour map with automatic clim control.
  profile  — instantaneous vertical profile (value vs height) at a chosen
              time index.
"""

import param
import holoviews as hv
import panel as pn
from holoviews import streams

from helpers import (
    get_label_to_var,
    _make_clim_controls,
    make_plot_with_controls_layout,
    plot_2d_heatmap,
)


def _detect_colstat_zdim(da):
    """Return the vertical-dimension name for a colstat DataArray.

    Checks for ``zf`` (full levels, tt-grid) first, then ``zh`` (half levels,
    mt-grid), then falls back to the common DALES names ``zt``/``zm``/``zh``.
    Raises ValueError when no recognised vertical dimension is present.
    """
    for dim in ("zf", "zh", "zt", "zm"):
        if dim in da.dims:
            return dim
    raise ValueError(f"No vertical dimension found in variable '{da.name}'")


class _ColstatController(param.Parameterized):
    file_label = param.ObjectSelector(default=None, objects=[], label="Location")
    variable = param.ObjectSelector(default=None, objects=[], label="Variable")
    mode = param.ObjectSelector(
        default="heatmap", objects=["heatmap", "profile"], label="View"
    )
    time_index = param.Integer(default=0, bounds=(0, 0), label="Time index")
    auto = param.Boolean(default=True, doc="Automatically set clim from current view")
    trigger = param.Integer(default=0, doc="Manual trigger to recompute clim")


def make_colstat_panel(datasets):
    """Panel for a collection of colstat.X.Y.nc datasets.

    Parameters
    ----------
    datasets : dict[str, xr.Dataset]
        Mapping of location label (derived from locx/locy global attributes or
        file stem) to the opened dataset.
    """
    if not datasets:
        return pn.pane.Markdown("No colstat files found.", sizing_mode="stretch_width")

    file_labels = list(datasets.keys())
    controller = _ColstatController()
    controller.param["file_label"].objects = file_labels
    controller.file_label = file_labels[0]

    # ------------------------------------------------------------------ helpers
    def _current_ds():
        return datasets.get(controller.file_label)

    def _current_label_to_var():
        ds = _current_ds()
        return get_label_to_var(ds) if ds is not None else {}

    def _n_time(ds):
        if ds is not None and "time" in ds.dims:
            return int(ds.sizes["time"])
        return 1

    def _loc_str(ds):
        locx = ds.attrs.get("locx", None)
        locy = ds.attrs.get("locy", None)
        if locx is not None and locy is not None:
            return f"  (x={float(locx):.1f} m, y={float(locy):.1f} m)"
        return ""

    # ---------------------------------------------------- initial widget state
    label_to_var = _current_label_to_var()
    var_labels = list(label_to_var.keys())
    controller.param["variable"].objects = var_labels
    controller.variable = var_labels[0] if var_labels else None

    ds0 = datasets[file_labels[0]]
    n = _n_time(ds0)
    controller.param["time_index"].bounds = (0, max(n - 1, 0))

    # ------------------------------------------------- react to location change
    def _update_vars(event):
        ds = _current_ds()
        ltv = get_label_to_var(ds) if ds is not None else {}
        new_labels = list(ltv.keys())
        controller.param["variable"].objects = new_labels
        controller.variable = new_labels[0] if new_labels else None
        nt = _n_time(ds)
        controller.param["time_index"].bounds = (0, max(nt - 1, 0))
        controller.time_index = 0

    controller.param.watch(_update_vars, "file_label")

    # ----------------------------------------------------------- heatmap DMap
    range_stream = streams.RangeXY()
    heat_param_stream = streams.Params(
        controller, parameters=["file_label", "variable", "auto", "trigger"]
    )

    def heat_fn(x_range=None, y_range=None, **kwargs):
        try:
            ds = _current_ds()
            if ds is None:
                return hv.Curve([])

            ltv = get_label_to_var(ds)
            var_label = controller.variable
            if var_label is None or var_label not in ltv:
                return hv.Curve([])

            var = ltv[var_label]
            da = ds[var]

            try:
                zdim = _detect_colstat_zdim(da)
            except ValueError:
                # No vertical dimension — render as a plain time series.
                if "time" in da.dims:
                    return da.hvplot(
                        x="time",
                        title=var_label + _loc_str(ds),
                        height=300,
                        responsive=True,
                    )
                return hv.Curve([])

            title = var_label + _loc_str(ds)
            return plot_2d_heatmap(
                da,
                xdim="time",
                ydim=zdim,
                title=title,
                full_da=da,
                x_range=x_range,
                y_range=y_range,
                auto=controller.auto,
                trigger=controller.trigger,
                bounds_source=ds,
            )
        except Exception:
            import traceback, sys

            traceback.print_exc(file=sys.stderr)
            return hv.Curve([])

    heat_dmap = hv.DynamicMap(heat_fn, streams=[range_stream, heat_param_stream]).opts(
        framewise=False
    )
    range_stream.source = heat_dmap
    heat_plot = pn.panel(heat_dmap, sizing_mode="stretch_width")

    # ---------------------------------------------------------- profile DMap
    profile_param_stream = streams.Params(
        controller, parameters=["file_label", "variable", "time_index"]
    )

    def profile_fn(**kwargs):
        try:
            ds = _current_ds()
            if ds is None:
                return hv.Curve([])

            ltv = get_label_to_var(ds)
            var_label = controller.variable
            if var_label is None or var_label not in ltv:
                return hv.Curve([])

            var = ltv[var_label]
            da = ds[var]

            if "time" not in da.dims:
                return hv.Curve([])

            try:
                zdim = _detect_colstat_zdim(da)
            except ValueError:
                return da.hvplot(
                    x="time",
                    title=var_label + _loc_str(ds),
                    height=300,
                    responsive=True,
                )

            n_t = da.sizes.get("time", 1)
            idx = max(0, min(int(controller.time_index), n_t - 1))
            prof = da.isel(time=idx)

            z_vals = ds[zdim].values if zdim in ds.coords else prof[zdim].values
            x_vals = prof.values

            xmin = float(da.min(skipna=True).values)
            xmax = float(da.max(skipna=True).values)
            if xmin == xmax:
                pad = 1.0 if xmin == 0 else 0.01 * abs(xmin)
                xmin -= pad
                xmax += pad

            units = da.attrs.get("units", "")
            xlabel = f"{var} [{units}]" if units else var
            title = var_label + _loc_str(ds)

            return hv.Curve((x_vals, z_vals), var, zdim).opts(
                xlabel=xlabel,
                ylabel=f"{zdim} [m]",
                xlim=(xmin, xmax),
                title=title,
                height=300,
                responsive=True,
            )
        except Exception:
            import traceback, sys

            traceback.print_exc(file=sys.stderr)
            return hv.Curve([])

    profile_dmap = hv.DynamicMap(profile_fn, streams=[profile_param_stream]).opts(
        framewise=True
    )
    profile_plot = pn.panel(profile_dmap, sizing_mode="stretch_width")

    # ---------------------------------------- toggle between heatmap / profile
    plot_area = pn.Column(heat_plot, sizing_mode="stretch_width")

    def _toggle_view(*events):
        range_stream.event(x_range=None, y_range=None)
        if controller.mode == "heatmap":
            plot_area.objects = [heat_plot]
        else:
            plot_area.objects = [profile_plot]

    controller.param.watch(_toggle_view, ["mode"])

    # ----------------------------------------------------------- build widgets
    file_select = pn.widgets.Select.from_param(
        controller.param.file_label, name="Location"
    )
    var_select = pn.widgets.Select.from_param(
        controller.param.variable, name="Variable"
    )
    mode_toggle = pn.widgets.RadioButtonGroup.from_param(
        controller.param.mode, name="View", button_type="default"
    )
    time_slider = pn.widgets.IntSlider.from_param(
        controller.param.time_index, name="Time index"
    )
    time_slider.visible = False  # only meaningful in profile mode

    def _update_time_slider_vis(*events):
        time_slider.visible = controller.mode == "profile"

    controller.param.watch(_update_time_slider_vis, "mode")

    auto_checkbox, button_view, button_global = _make_clim_controls(controller)

    controls = pn.Column(
        file_select,
        var_select,
        mode_toggle,
        time_slider,
        auto_checkbox,
        button_view,
        button_global,
        sizing_mode="stretch_width",
    )

    return make_plot_with_controls_layout(plot_area, controls)
