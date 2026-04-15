from _heatmap_element import _heatmap_element
from _profile_element import _profile_element
from helpers import _make_clim_controls
from controllers import ClimController
from helpers import _detect_vertical_dim, get_label_to_var, catchall


import holoviews as hv
import panel as pn
from holoviews import streams
import numpy as np


@catchall
def make_heat_panel_with_clim_controls(ds):
    label_to_var = get_label_to_var(ds)

    # set up controller with explicit variable selector
    labels = list(label_to_var.keys())
    controller = ClimController(label=labels[0] if labels else None)
    controller.param["label"].objects = labels

    # configure time index bounds for this dataset
    n_time = int(ds.sizes.get("time", len(ds["time"])))
    controller.time_index = 0
    controller.param["time_index"].bounds = (0, max(n_time - 1, 0))

    # Heatmap DynamicMap (uses zoom/clim controls)
    range_stream = streams.RangeXY()
    heat_param_stream = streams.Params(
        controller, parameters=["label", "auto", "trigger", "mode"]
    )

    def heat_fn(x_range=None, y_range=None, **kwargs):
        label = controller.label
        auto = controller.auto
        trigger = controller.trigger

        if label is None:
            return hv.NdOverlay({})

        var = label_to_var[label]
        da = ds[var]

        # If there is no time dimension, fall back to a simple 1D plot
        # instead of attempting a time–height heatmap.
        if "time" not in da.dims:
            # Pure 1D variable: plot against its single dimension.
            if da.ndim == 1:
                dim = da.dims[0]
                return da.hvplot(
                    x=dim,
                    title=label,
                    height=300,
                    responsive=True,
                )

            # If there is a recognizable vertical dimension, treat it as
            # a vertical profile (value vs height).
            try:
                ydim = _detect_vertical_dim(da)
                z_vals = ds[ydim].values if ydim in ds else da[ydim].values
                x_vals = da.values
                return hv.Curve((x_vals, z_vals), var, ydim).opts(
                    xlabel=label,
                    ylabel=ydim,
                    title=label,
                    height=300,
                    responsive=True,
                )
            except Exception:
                # As a last resort, return an empty curve.
                return hv.Curve([])

        # If the variable has a vertical dimension (zt/zm/zh), show a 2D
        # heatmap with clim controls. Otherwise, fall back to a 1D time
        # series (as in the original forcings panel).
        try:
            ydim = _detect_vertical_dim(da)
            has_vertical = True
        except ValueError:
            has_vertical = False

        if not has_vertical:
            return da.hvplot(
                x="time",
                title=label,
                height=300,
                responsive=True,
            ).opts(
                backend_opts={
                    "x_range.bounds": (
                        float(ds.time.min(skipna=True).values),
                        float(ds.time.max(skipna=True).values),
                    )
                }
            )

        full_da = da
        try:
            vmin = float(full_da.min(skipna=True))
            vmax = float(full_da.max(skipna=True))
        except ValueError:
            raise ValueError(
                f"Variable {label} has non-numeric data {full_da.values} {ds}"
            )

        use_view = x_range is not None and y_range is not None and (auto or trigger > 0)
        if use_view:
            t0, t1 = x_range
            z0, z1 = y_range
            view_da = full_da.sel(time=slice(t0, t1))
            if ydim in view_da.dims:
                view_da = view_da.sel({ydim: slice(z0, z1)})
            if view_da.size > 0:
                try:
                    v_view = float(view_da.min(skipna=True))
                    V_view = float(view_da.max(skipna=True))

                    # Only override global clim when the view-based range is
                    # finite and non-degenerate. This avoids clim=(NaN, NaN)
                    # when zooming into fully masked/NaN regions.
                    if np.isfinite(v_view) and np.isfinite(V_view) and v_view != V_view:
                        vmin, vmax = v_view, V_view
                except ValueError:
                    # Fall back to a simple default range if the view cannot
                    # be reduced numerically.
                    vmin = 0.0
                    vmax = 1.0

        return _heatmap_element(ds, full_da, label, ydim, vmin, vmax)

    heat_dmap = hv.DynamicMap(
        heat_fn,
        streams=[range_stream, heat_param_stream],
    ).opts(framewise=False)

    range_stream.source = heat_dmap
    heat_plot = pn.panel(heat_dmap, sizing_mode="stretch_width")

    # Profile DynamicMap (time-scrolling vertical profile)
    profile_param_stream = streams.Params(
        controller, parameters=["label", "time_index", "mode"]
    )

    def profile_fn(**kwargs):
        label = controller.label
        time_index = controller.time_index

        if label is None:
            return hv.Curve([])

        var = label_to_var[label]
        da = ds[var]

        # If there is no time axis, nothing to display.
        if "time" not in da.dims:
            return hv.Curve([])

        # For fields without a vertical dimension, reuse the 1D time series
        # view instead of a vertical profile.
        try:
            ydim = _detect_vertical_dim(da)
        except ValueError:
            return da.hvplot(
                x="time",
                title=label,
                height=300,
                responsive=True,
            )

        return _profile_element(ds, da, label, ydim, time_index)

    profile_dmap = hv.DynamicMap(
        profile_fn,
        streams=[profile_param_stream],
    ).opts(
        framewise=True,
        height=300,
        responsive=True,
    )

    profile_plot = pn.panel(profile_dmap, sizing_mode="stretch_width")

    # Toggle between views by swapping Column contents
    plot_area = pn.Column(heat_plot, sizing_mode="stretch_width")

    def _toggle_view(*events):
        range_stream.event(x_range=None, y_range=None)
        if controller.mode == "heatmap":
            plot_area.objects = [heat_plot]
        else:
            plot_area.objects = [profile_plot]

    controller.param.watch(_toggle_view, ["mode"])

    var_select = pn.widgets.Select.from_param(controller.param.label, name="Variable")
    mode_toggle = pn.widgets.RadioButtonGroup.from_param(
        controller.param.mode, name="View", button_type="default"
    )
    time_slider = pn.widgets.IntSlider.from_param(
        controller.param.time_index, name="Time index"
    )
    auto_checkbox, button_view, button_global = _make_clim_controls(controller)

    controls = pn.Column(
        var_select,
        mode_toggle,
        time_slider,
        auto_checkbox,
        button_view,
        button_global,
        sizing_mode="stretch_width",
    )

    return pn.Row(plot_area, controls, sizing_mode="stretch_width")
