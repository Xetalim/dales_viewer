from helpers import (
    determine_clim_and_cmap,
)
from controllers import SliceController, FielddumpSliceController, _select_plot_dims
from helpers import (
    _build_backend_bounds,
    _compute_slice_clim,
    _make_clim_controls,
    get_label_to_var,
    catchall,
)


import holoviews as hv
import hvplot
import panel as pn
from holoviews import streams


def make_slice_panel(ds, slice_dim=None):
    """2D slice panel with variable and slice selectors.

    If ``slice_dim`` is provided, slice along that dimension (e.g. 'yt' for
    crossxz) and expose index + optional time controls, including an
    animation saver.

    If ``slice_dim`` is ``None``, expose a "Slice dim" selector so the user
    can choose which dimension to slice along (as in the original
    fielddump slice panel).
    """

    label_to_var = get_label_to_var(ds)
    labels = list(label_to_var.keys())

    if not labels:
        return pn.pane.Markdown("No compatible variables", sizing_mode="stretch_width")

    # --- Dynamic slice-dimension mode (fielddump-style) ---
    if slice_dim is None:
        dims = [d for d in ds.dims if d != "time"]
        if not dims:
            return pn.pane.Markdown(
                "No sliceable dimensions", sizing_mode="stretch_width"
            )

        if "zt" in dims:
            default_dim = "zt"
            dims = [default_dim] + [d for d in dims if d != default_dim]
        else:
            default_dim = dims[0]

        controller = FielddumpSliceController(var=labels[0], dim=default_dim)
        controller.param["var"].objects = labels
        controller.param["dim"].objects = dims

        n = int(ds.sizes[default_dim])
        controller.index = 0
        controller.param["index"].bounds = (0, max(n - 1, 0))

        # Configure optional time index if time dimension is present
        has_time = "time" in ds.dims
        if has_time:
            n_time = int(ds.sizes["time"])
            controller.time_index = 0
            controller.param["time_index"].bounds = (0, max(n_time - 1, 0))

        def _update_index_bounds(event):
            dim = controller.dim
            if dim not in ds.dims:
                return
            n_loc = int(ds.sizes[dim])
            controller.param["index"].bounds = (0, max(n_loc - 1, 0))
            if controller.index > n_loc - 1:
                controller.index = n_loc - 1

        controller.param.watch(_update_index_bounds, "dim")

        range_stream = streams.RangeXY()
        param_stream = streams.Params(
            controller,
            parameters=[
                "var",
                "dim",
                "index",
                "time_index",
                "auto",
                "trigger",
                "symmetric_cmap",
            ],
        )

        @catchall
        def _slice_fn_dynamic(x_range=None, y_range=None, **kwargs):
            var_label = controller.var
            dim = controller.dim
            idx = controller.index
            t_idx = controller.time_index
            auto = controller.auto
            trigger = controller.trigger
            symmetric = controller.symmetric_cmap

            if var_label is None or dim is None:
                return hv.Curve([])

            var_name = label_to_var[var_label]
            da_var = ds[var_name]

            if dim not in da_var.dims:
                return hv.Curve([])

            # Slice along chosen dimension and, if available, time
            slice_sel = {dim: idx}
            if "time" in da_var.dims:
                slice_sel["time"] = t_idx

            other_dims = [d for d in da_var.dims if d not in slice_sel]
            if len(other_dims) != 2:
                return hv.Curve([])

            xdim, ydim = _select_plot_dims(other_dims)
            if xdim is None or ydim is None:
                return hv.Curve([])

            sliced = da_var.isel(slice_sel)

            vmin, vmax = _compute_slice_clim(
                da_var, sliced, xdim, ydim, x_range, y_range, auto, trigger
            )

            if symmetric:
                vmax_abs = max(abs(vmin), abs(vmax))
                vmin, vmax = -vmax_abs, vmax_abs

            clim, cmap = determine_clim_and_cmap(vmin, vmax)

            if dim in ds.coords:
                coord_val = float(ds[dim].isel({dim: idx}).values)
                title = f"{var_label} @ {dim}={coord_val:.2f}"
            else:
                title = f"{var_label} @ {dim} index {idx}"

            if "time" in da_var.dims:
                if "time" in ds.coords:
                    t_val = float(ds["time"].isel(time=t_idx).values)
                    title += f", time={t_val:.2f}"
                else:
                    title += f", time index {t_idx}"

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

            backend_opts = _build_backend_bounds(ds, xdim, ydim)
            return plot.opts(backend_opts=backend_opts)

        dmap = hv.DynamicMap(
            _slice_fn_dynamic, streams=[range_stream, param_stream]
        ).opts(
            framewise=False,
            shared_axes=False,
            axiswise=True,
        )

        range_stream.source = dmap

        plot = pn.panel(dmap, sizing_mode="stretch_width")

        var_select = pn.widgets.Select.from_param(controller.param.var, name="Variable")
        dim_select = pn.widgets.Select.from_param(
            controller.param.dim, name="Slice dim"
        )
        index_slider = pn.widgets.IntSlider.from_param(
            controller.param.index, name="Index"
        )
        auto_checkbox, button_view, button_global = _make_clim_controls(controller)

        sym_toggle = pn.widgets.Toggle.from_param(
            controller.param.symmetric_cmap,
            name="Symmetric clim around 0",
            button_type="primary",
        )

        widgets = [
            var_select,
            dim_select,
            index_slider,
            auto_checkbox,
            button_view,
            button_global,
            sym_toggle,
        ]
        if has_time:
            time_slider = pn.widgets.IntSlider.from_param(
                controller.param.time_index, name="Time index"
            )
            widgets.append(time_slider)

        controls = pn.Column(*widgets, sizing_mode="stretch_width")

        return pn.Row(plot, controls, sizing_mode="stretch_width")

    # --- Fixed slice-dimension mode (cross-section-style, with animation) ---
    if slice_dim not in ds.dims:
        return pn.pane.Markdown("No compatible variables", sizing_mode="stretch_width")

    controller = SliceController(var=labels[0])
    controller.param["var"].objects = labels

    n = int(ds.sizes[slice_dim])
    controller.index = 0
    controller.param["index"].bounds = (0, max(n - 1, 0))

    # Configure optional time index if time dimension is present
    has_time = "time" in ds.dims
    if has_time:
        n_time = int(ds.sizes["time"])
        controller.time_index = 0
        controller.param["time_index"].bounds = (0, max(n_time - 1, 0))

    range_stream = streams.RangeXY()
    param_stream = streams.Params(
        controller,
        parameters=[
            "var",
            "index",
            "time_index",
            "auto",
            "trigger",
            "symmetric_cmap",
        ],
    )

    @catchall
    def slice_fn(x_range=None, y_range=None, **kwargs):
        var_label = controller.var
        idx = controller.index
        t_idx = controller.time_index
        auto = controller.auto
        trigger = controller.trigger
        symmetric = controller.symmetric_cmap

        if var_label is None:
            return hv.Curve([])

        var_name = label_to_var[var_label]
        da_var = ds[var_name]

        if slice_dim not in da_var.dims:
            return hv.Curve([])

        # Slice along the primary slice dimension and, if available, time
        slice_sel = {slice_dim: idx}
        if "time" in da_var.dims:
            slice_sel["time"] = t_idx

        other_dims = [d for d in da_var.dims if d not in slice_sel]
        if len(other_dims) != 2:
            return hv.Curve([])

        xdim, ydim = _select_plot_dims(other_dims)
        if xdim is None or ydim is None:
            return hv.Curve([])
        sliced = da_var.isel(slice_sel)

        vmin, vmax = _compute_slice_clim(
            da_var, sliced, xdim, ydim, x_range, y_range, auto, trigger
        )

        if symmetric:
            vmax_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vmax_abs, vmax_abs

        clim, cmap = determine_clim_and_cmap(vmin, vmax)

        if slice_dim in ds.coords:
            coord_val = float(ds[slice_dim].isel({slice_dim: idx}).values)
            title = f"{var_label} @ {slice_dim}={coord_val:.2f}"
        else:
            title = f"{var_label} @ {slice_dim} index {idx}"

        if "time" in da_var.dims:
            if "time" in ds.coords:
                t_val = float(ds["time"].isel(time=t_idx).values)
                title += f", time={t_val:.2f}"
            else:
                title += f", time index {t_idx}"

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

        backend_opts = _build_backend_bounds(ds, xdim, ydim)
        return plot.opts(backend_opts=backend_opts)

    dmap = hv.DynamicMap(slice_fn, streams=[range_stream, param_stream]).opts(
        framewise=False,
        shared_axes=False,
        axiswise=True,
    )

    range_stream.source = dmap

    plot = pn.panel(dmap, sizing_mode="stretch_width")

    var_select = pn.widgets.Select.from_param(controller.param.var, name="Variable")
    index_slider = pn.widgets.IntSlider.from_param(
        controller.param.index, name=f"{slice_dim} index"
    )
    auto_checkbox, button_view, button_global = _make_clim_controls(controller)

    sym_toggle = pn.widgets.Toggle.from_param(
        controller.param.symmetric_cmap,
        name="Symmetric clim around 0",
        button_type="primary",
    )

    save_button = pn.widgets.Button(name="Save MP4 animation", button_type="success")

    def _on_save_click(event):
        var_label = controller.var
        if var_label is None:
            return

        var_name = label_to_var.get(var_label)
        if var_name is None:
            return

        da_var = ds[var_name]

        if slice_dim not in da_var.dims:
            return

        n_frames = int(ds.sizes.get(slice_dim, 0))
        if n_frames <= 0:
            return

        base_sel = {}
        if "time" in da_var.dims:
            base_sel["time"] = controller.time_index

        other_dims = [d for d in da_var.dims if d not in base_sel and d != slice_dim]
        if len(other_dims) != 2:
            return

        xdim, ydim = _select_plot_dims(other_dims)
        if xdim is None or ydim is None:
            return

        # Use global clim over the variable for the animation
        vmin = float(da_var.min(skipna=True))
        vmax = float(da_var.max(skipna=True))
        clim, cmap = determine_clim_and_cmap(vmin, vmax)
        frames = {}
        for i in range(n_frames):
            sel = dict(base_sel)
            sel[slice_dim] = i
            sliced = da_var.isel(sel)

            if slice_dim in ds.coords:
                coord_val = float(ds[slice_dim].isel({slice_dim: i}).values)
                title = f"{var_label} @ {slice_dim}={coord_val:.2f}"
            else:
                title = f"{var_label} @ {slice_dim} index {i}"

            if "time" in da_var.dims:
                t_idx = base_sel.get("time", 0)
                if "time" in ds.coords:
                    t_val = float(ds["time"].isel(time=t_idx).values)
                    title += f", time={t_val:.2f}"
                else:
                    title += f", time index {t_idx}"

            frame_plot = sliced.hvplot(
                x=xdim,
                y=ydim,
                clim=clim,
                cmap=cmap,
                colorbar=True,
                title=title,
                height=300,
                width=600,
                responsive=True,
            )

            backend_opts = _build_backend_bounds(ds, xdim, ydim)
            frames[i] = frame_plot.opts(backend_opts=backend_opts)

        hmap = hv.HoloMap(frames, kdims=[slice_dim])
        filename = f"{var_name}_{slice_dim}_animation.html"
        try:
            hvplot.save(hmap, filename, fps=24)
        except Exception as e:
            # If saving fails (e.g. missing ffmpeg), just ignore
            with open("err.log", "w") as file:
                file.write(
                    f"Error saving animation: {e}\n{filename=}\n{var_name=}\n{slice_dim=}\n{n_frames=}"
                )
            return

    save_button.on_click(_on_save_click)

    widgets = [
        var_select,
        index_slider,
        auto_checkbox,
        button_view,
        button_global,
        sym_toggle,
        save_button,
    ]
    if has_time:
        time_slider = pn.widgets.IntSlider.from_param(
            controller.param.time_index, name="Time index"
        )
        widgets.append(time_slider)

    controls = pn.Column(*widgets, sizing_mode="stretch_width")

    return pn.Row(plot, controls, sizing_mode="stretch_width")
