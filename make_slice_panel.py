from controllers import SliceController, FielddumpSliceController
from helpers import (
    _make_clim_controls,
    append_indexed_dim_to_title,
    get_label_to_var,
    catchall,
    make_plot_with_controls_layout,
    plot_2d_heatmap,
    slice_to_2d,
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

    def _resolve_slice_dim_for_var(da_var, requested_dim):
        """Resolve requested slice dim to an available dim on the variable.

        This treats zt/zm as interchangeable index grids for cross-sections:
        when a variable is only available on one of them, keep the selected
        index and transparently use the available sister dimension.
        """

        if requested_dim in da_var.dims:
            return requested_dim

        sister_dim = {"zt": "zm", "zm": "zt"}.get(requested_dim)
        if sister_dim in da_var.dims:
            return sister_dim

        return None

    def _paired_slice_dims(dim_name):
        return {
            "zt": "zm",
            "zm": "zt",
            "xt": "xm",
            "xm": "xt",
            "yt": "ym",
            "ym": "yt",
        }.get(dim_name)

    def _build_slice_indexers(da_var, requested_dim, idx):
        """Build indexers, slicing staggered sister dims at the same index.

        Some DALES fields carry both staggered dimensions (e.g. zt and zm).
        To keep a single logical slice coordinate for users, apply the same
        index to both when both are present.
        """

        dim_eff = _resolve_slice_dim_for_var(da_var, requested_dim)
        if dim_eff is None:
            return None, None

        indexers = {dim_eff: idx}
        sister_dim = _paired_slice_dims(dim_eff)
        if sister_dim in da_var.dims:
            indexers[sister_dim] = idx

        return indexers, dim_eff

    def _shared_index_size(da_var, requested_dim):
        """Return valid index length, constrained across staggered sisters."""

        dim_eff = _resolve_slice_dim_for_var(da_var, requested_dim)
        if dim_eff is None:
            return None, None

        sizes = [int(da_var.sizes[dim_eff])]
        sister_dim = _paired_slice_dims(dim_eff)
        if sister_dim in da_var.dims:
            sizes.append(int(da_var.sizes[sister_dim]))

        return min(sizes), dim_eff

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

        def _update_index_bounds(_event):
            var_label = controller.var
            dim = controller.dim
            if var_label is None or dim is None:
                return

            var_name = label_to_var.get(var_label)
            if var_name is None:
                return

            da_var = ds[var_name]
            n_loc, _dim_eff = _shared_index_size(da_var, dim)
            if n_loc is None:
                return

            controller.param["index"].bounds = (0, max(n_loc - 1, 0))
            if controller.index > n_loc - 1:
                controller.index = n_loc - 1

        controller.param.watch(_update_index_bounds, ["dim", "var"])

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

        # Track the plotted axis context so we only hard-reset start/end when
        # switching to a different coordinate system (e.g. zt->yt slice mode).
        axis_context = {"key": None}

        @catchall
        def _slice_fn_dynamic(x_range=None, y_range=None, **_kwargs):
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

            indexers, dim_eff = _build_slice_indexers(da_var, dim, idx)
            if indexers is None:
                return hv.Curve([])

            # Slice along chosen dimension and, if available, time
            sliced, xdim, ydim, _slice_sel = slice_to_2d(
                da_var, **{**indexers, "time": t_idx}
            )
            if xdim is None or ydim is None:
                return hv.Curve([])

            title = append_indexed_dim_to_title(
                var_label,
                ds,
                dim_eff,
                idx,
                separator=" @ ",
                formatter=lambda value: f"{float(value):.2f}",
            )
            if dim_eff != dim:
                title = f"{title} (using {dim_eff})"
            if "time" in da_var.dims:
                title = append_indexed_dim_to_title(
                    title,
                    ds,
                    "time",
                    t_idx,
                    separator=", ",
                    formatter=lambda value: f"{float(value):.2f}",
                )

            current_key = (var_name, dim_eff, xdim, ydim)
            extra_backend_opts = {}
            if axis_context["key"] != current_key:
                if xdim in ds.coords:
                    x0 = float(ds[xdim].min(skipna=True).values)
                    x1 = float(ds[xdim].max(skipna=True).values)
                    extra_backend_opts["x_range.start"] = x0
                    extra_backend_opts["x_range.end"] = x1
                    extra_backend_opts["x_range.reset_start"] = x0
                    extra_backend_opts["x_range.reset_end"] = x1
                if ydim in ds.coords:
                    y0 = float(ds[ydim].min(skipna=True).values)
                    y1 = float(ds[ydim].max(skipna=True).values)
                    extra_backend_opts["y_range.start"] = y0
                    extra_backend_opts["y_range.end"] = y1
                    extra_backend_opts["y_range.reset_start"] = y0
                    extra_backend_opts["y_range.reset_end"] = y1
                axis_context["key"] = current_key

            return plot_2d_heatmap(
                sliced,
                xdim=xdim,
                ydim=ydim,
                title=title,
                full_da=da_var,
                x_range=x_range,
                y_range=y_range,
                auto=auto,
                trigger=trigger,
                symmetric_cmap=symmetric,
                bounds_source=ds,
                extra_backend_opts=extra_backend_opts,
            )

        dmap = hv.DynamicMap(
            _slice_fn_dynamic, streams=[range_stream, param_stream]
        ).opts(
            framewise=False,
            shared_axes=False,
            axiswise=True,
        )

        range_stream.source = dmap

        def _reset_view_range(_event):
            # When switching variable/slice dimension, old backend ranges can be
            # incompatible with the new axes and yield empty view selections.
            range_stream.event(x_range=None, y_range=None)

        controller.param.watch(_reset_view_range, ["dim", "var"])

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

        return make_plot_with_controls_layout(plot, controls)

    # --- Fixed slice-dimension mode (cross-section-style, with animation) ---
    if slice_dim not in ds.dims:
        return pn.pane.Markdown("No compatible variables", sizing_mode="stretch_width")

    controller = SliceController(var=labels[0])
    controller.param["var"].objects = labels

    first_var_name = label_to_var[labels[0]]
    first_dim_eff = _resolve_slice_dim_for_var(ds[first_var_name], slice_dim)
    if first_dim_eff is None:
        return pn.pane.Markdown("No compatible variables", sizing_mode="stretch_width")

    n, _ = _shared_index_size(ds[first_var_name], slice_dim)
    if n is None:
        return pn.pane.Markdown("No compatible variables", sizing_mode="stretch_width")

    controller.index = 0
    controller.param["index"].bounds = (0, max(n - 1, 0))

    def _update_fixed_index_bounds(_event):
        var_label = controller.var
        if var_label is None:
            return

        var_name = label_to_var.get(var_label)
        if var_name is None:
            return

        n_loc, _dim_eff = _shared_index_size(ds[var_name], slice_dim)
        if n_loc is None:
            return

        controller.param["index"].bounds = (0, max(n_loc - 1, 0))
        if controller.index > n_loc - 1:
            controller.index = n_loc - 1

    controller.param.watch(_update_fixed_index_bounds, "var")

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
    def slice_fn(x_range=None, y_range=None, **_kwargs):
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

        indexers, dim_eff = _build_slice_indexers(da_var, slice_dim, idx)
        if indexers is None:
            return hv.Curve([])

        # Slice along the primary slice dimension and, if available, time
        sliced, xdim, ydim, _slice_sel = slice_to_2d(
            da_var, **{**indexers, "time": t_idx}
        )
        if xdim is None or ydim is None:
            return hv.Curve([])
        title = append_indexed_dim_to_title(
            var_label,
            ds,
            dim_eff,
            idx,
            separator=" @ ",
            formatter=lambda value: f"{float(value):.2f}",
        )
        if dim_eff != slice_dim:
            title = f"{title} (using {dim_eff})"

        if "time" in da_var.dims:
            title = append_indexed_dim_to_title(
                title,
                ds,
                "time",
                t_idx,
                separator=", ",
                formatter=lambda value: f"{float(value):.2f}",
            )
        return plot_2d_heatmap(
            sliced,
            xdim=xdim,
            ydim=ydim,
            title=title,
            full_da=da_var,
            x_range=x_range,
            y_range=y_range,
            auto=auto,
            trigger=trigger,
            symmetric_cmap=symmetric,
            bounds_source=ds,
        )

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

    def _on_save_click(_event):
        var_label = controller.var
        if var_label is None:
            return

        var_name = label_to_var.get(var_label)
        if var_name is None:
            return

        da_var = ds[var_name]

        dim_eff = _resolve_slice_dim_for_var(da_var, slice_dim)
        if dim_eff is None:
            return

        n_frames, _dim_eff = _shared_index_size(da_var, slice_dim)
        if n_frames is None:
            return
        if n_frames <= 0:
            return

        base_sel = {}
        if "time" in da_var.dims:
            base_sel["time"] = controller.time_index

        first_indexers, _first_dim = _build_slice_indexers(da_var, slice_dim, 0)
        if first_indexers is None:
            return

        _first_slice, xdim, ydim, _first_sel = slice_to_2d(da_var, **{**first_indexers, **base_sel})
        if xdim is None or ydim is None:
            return

        # Use global clim over the variable for the animation
        vmin = float(da_var.min(skipna=True))
        vmax = float(da_var.max(skipna=True))
        frames = {}
        for i in range(n_frames):
            frame_indexers, frame_dim_eff = _build_slice_indexers(da_var, slice_dim, i)
            if frame_indexers is None:
                continue

            sliced, _xdim, _ydim, _sel = slice_to_2d(
                da_var, **{**frame_indexers, **base_sel}
            )
            if sliced is None:
                continue

            title = append_indexed_dim_to_title(
                var_label,
                ds,
                frame_dim_eff,
                i,
                separator=" @ ",
                formatter=lambda value: f"{float(value):.2f}",
            )
            if frame_dim_eff != slice_dim:
                title = f"{title} (using {frame_dim_eff})"

            if "time" in da_var.dims:
                t_idx = base_sel.get("time", 0)
                title = append_indexed_dim_to_title(
                    title,
                    ds,
                    "time",
                    t_idx,
                    separator=", ",
                    formatter=lambda value: f"{float(value):.2f}",
                )

            frames[i] = plot_2d_heatmap(
                sliced,
                xdim=xdim,
                ydim=ydim,
                title=title,
                bounds_source=ds,
                vmin=vmin,
                vmax=vmax,
                height=300,
            )

        hmap = hv.HoloMap(frames, kdims=[dim_eff])
        filename = f"{var_name}_{dim_eff}_animation.html"
        try:
            hvplot.save(hmap, filename, fps=24)
        except Exception as e:
            # If saving fails (e.g. missing ffmpeg), just ignore
            with open("err.log", "w", encoding="utf-8") as file:
                file.write(
                    f"Error saving animation: {e}\n{filename=}\n{var_name=}\n{slice_dim=}\n{dim_eff=}\n{n_frames=}"
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

    return make_plot_with_controls_layout(plot, controls)
