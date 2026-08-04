from _heatmap_element import _heatmap_element
from _profile_element import _profile_element
from helpers import _make_clim_controls, make_plot_with_controls_layout
from controllers import ClimController
from helpers import _detect_vertical_dim, get_label_to_var


import holoviews as hv
import panel as pn
from holoviews import streams
import numpy as np


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

    def _reduce_to_time_and_vertical(da, ydim):
        """Select index 0 on non-(time, ydim) dims so 2D plotting always works."""
        keep = {"time", ydim}
        extra_dims = [dim for dim in da.dims if dim not in keep]
        if not extra_dims:
            return da
        return da.isel({dim: 0 for dim in extra_dims}, drop=True)

    def heat_fn(x_range=None, y_range=None, **_kwargs):
        label = controller.label
        auto = controller.auto
        trigger = controller.trigger

        if label is None:
            raise RuntimeError("No variable selected for heatmap view")

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
                    responsive="width",
                )

            # If there is a recognizable vertical dimension, treat it as
            # a vertical profile (value vs height).
            ydim = _detect_vertical_dim(da)
            z_vals = ds[ydim].values if ydim in ds else da[ydim].values
            x_vals = da.values
            return hv.Curve((x_vals, z_vals), var, ydim).opts(
                xlabel=label,
                ylabel=ydim,
                title=label,
                height=300,
                responsive="width",
                framewise=False,
                shared_axes=False,
                axiswise=True,
            )

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
                responsive="width",
            ).opts(
                backend_opts={
                    "x_range.bounds": (
                        float(ds.time.min(skipna=True).values),
                        float(ds.time.max(skipna=True).values),
                    )
                }
            )

        da_plot = _reduce_to_time_and_vertical(da, ydim)
        if "time" not in da_plot.dims or ydim not in da_plot.dims or da_plot.ndim != 2:
            raise RuntimeError(
                f"Variable '{label}' is not plottable as time-height after reduction: dims={da_plot.dims}"
            )

        full_da = da_plot
        try:
            vmin = float(full_da.min(skipna=True))
            vmax = float(full_da.max(skipna=True))
        except ValueError as exc:
            raise ValueError(
                f"Variable {label} has non-numeric data {full_da.values} {ds}"
            ) from exc

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

        # Keep clim valid for first render (e.g. all-NaN/constant slices),
        # otherwise Bokeh/HoloViews may omit the colorbar until a later update.
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        elif vmin == vmax:
            pad = 1.0 if vmin == 0 else 0.01 * abs(vmin)
            vmin -= pad
            vmax += pad

        view = _heatmap_element(ds, da_plot, label, ydim, vmin, vmax)
        if view is None:
            raise RuntimeError(f"Heatmap renderer returned None for variable '{label}'")
        return view

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

    def profile_fn(**_kwargs):
        label = controller.label
        time_index = controller.time_index

        if label is None:
            raise RuntimeError("No variable selected for profile view")

        var = label_to_var[label]
        da = ds[var]

        # If there is no time axis, nothing to display.
        if "time" not in da.dims:
            raise RuntimeError(f"Variable '{label}' has no time axis for profile view")

        # For fields without a vertical dimension, reuse the 1D time series
        # view instead of a vertical profile.
        try:
            ydim = _detect_vertical_dim(da)
        except ValueError:
            return da.hvplot(
                x="time",
                title=label,
                height=300,
                responsive="width",
            )

        da_profile = _reduce_to_time_and_vertical(da, ydim)
        if "time" not in da_profile.dims or ydim not in da_profile.dims:
            raise RuntimeError(
                f"Variable '{label}' is not plottable as profile after reduction: dims={da_profile.dims}"
            )

        view = _profile_element(ds, da_profile, label, ydim, time_index)
        if view is None:
            raise RuntimeError(f"Profile renderer returned None for variable '{label}'")
        return view

    profile_dmap = hv.DynamicMap(
        profile_fn,
        streams=[profile_param_stream],
    ).opts(
        framewise=True,
        shared_axes=False,
        axiswise=True,
        height=300,
        responsive="width",
    )

    profile_plot = pn.panel(profile_dmap, sizing_mode="stretch_width")

    # Toggle between views by swapping Column contents
    plot_area = pn.Column(heat_plot, sizing_mode="stretch_width")

    def _toggle_view(*_events):
        range_stream.event(x_range=None, y_range=None)
        if controller.mode == "heatmap":
            plot_area.objects = [heat_plot]
        else:
            plot_area.objects = [profile_plot]

    def _reset_heat_range_on_var_change(_event):
        # Avoid carrying incompatible x/y ranges between variables.
        range_stream.event(x_range=None, y_range=None)

    controller.param.watch(_toggle_view, ["mode"])
    controller.param.watch(_reset_heat_range_on_var_change, ["label"])

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

    return make_plot_with_controls_layout(plot_area, controls)
