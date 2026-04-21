import holoviews as hv
import panel as pn
import numpy as np
import xarray as xr


def build_cover_mask(ds_target, ds_lsm, keep_zero=False):
    """Return a horizontal mask aligned to *ds_target* from lsm cover_slb.

    By default the mask keeps cells where ``cover_slb > 0`` and masks zeros.
    When ``keep_zero=True``, it keeps cells where ``cover_slb == 0`` and masks
    positive-cover cells instead.
    """

    if ds_lsm is None or "cover_slb" not in ds_lsm:
        return None

    cover = ds_lsm["cover_slb"]

    horiz_candidates = {d for d in cover.dims if d in ("x", "y", "xt", "yt")}
    non_horiz = [d for d in cover.dims if d not in horiz_candidates]
    if non_horiz:
        cover = cover.max(dim=non_horiz)

    dim_map = {}
    if "x" in cover.dims and "xt" in ds_target.dims:
        dim_map["x"] = "xt"
    if "y" in cover.dims and "yt" in ds_target.dims:
        dim_map["y"] = "yt"

    if dim_map:
        cover = cover.rename(dim_map)

    if not set(cover.dims).issubset(set(ds_target.dims)):
        return None

    if keep_zero:
        mask = xr.where(cover == 0, 1.0, np.nan)
    else:
        mask = xr.where(cover > 0, 1.0, np.nan)

    return mask.reset_coords(drop=True)


def apply_horizontal_mask(ds, mask):
    """Apply a horizontal mask to 2D and 3D horizontal fields.

    This mirrors the original slrbcross behavior:
    - 2D horizontal fields, typically (time, xt, yt)
    - 3D horizontal-plus-layer fields, typically (time, xt, yt, zts)
    """

    if mask is None:
        return None

    data_vars = {}
    mask_ndim = len(mask.dims)

    for name, da in ds.data_vars.items():
        if not np.issubdtype(da.dtype, np.number):
            data_vars[name] = da
            continue

        has_horizontal_dims = set(mask.dims).issubset(da.dims)
        is_2d_horizontal = da.ndim == mask_ndim + 1
        is_3d_horizontal = da.ndim == mask_ndim + 2

        if has_horizontal_dims and (is_2d_horizontal or is_3d_horizontal):
            if is_3d_horizontal:
                masked = da * mask.values[:, :, np.newaxis]
            else:
                masked = da * mask.values
            masked.attrs = da.attrs.copy()
            data_vars[name] = masked
        else:
            data_vars[name] = da

    return xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)


def _compute_slice_clim(
    full_da, sliced_da, xdim, ydim, x_range, y_range, auto, trigger
):
    """Compute color limits from either current view or sliced range.

    Performance notes
    -----------------
    For large 3D/4D datasets (e.g. fielddump), repeatedly computing a global
    min/max over ``full_da`` on every pan/zoom event is extremely expensive.
    Instead we:

    - Start from the cheaply computed range of the *current slice*;
    - Optionally refine using the view (x/y) window; and
    - Only fall back to a global reduction when the slice/view range is
      non-finite (all-NaN), caching that global range on ``full_da.attrs`` so
      it is computed at most once per variable.
    """

    def _finite_minmax(da):
        """Return finite min/max for a DataArray, or (nan, nan) if unavailable."""
        try:
            finite_da = da.where(np.isfinite(da))
            v0 = float(finite_da.min(skipna=True))
            v1 = float(finite_da.max(skipna=True))
        except (TypeError, ValueError):
            return np.nan, np.nan

        if np.isfinite(v0) and np.isfinite(v1):
            return v0, v1
        return np.nan, np.nan

    # Start from the sliced data range (cheap compared to full_da)
    vmin, vmax = _finite_minmax(sliced_da)

    use_view = x_range is not None and y_range is not None and (auto or trigger > 0)
    if use_view:
        x0, x1 = x_range
        y0, y1 = y_range

        # Guard against invalid or reversed ranges coming from backend reset/switch events.
        if all(np.isfinite(v) for v in (x0, x1, y0, y1)):
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            view_da = sliced_da
            try:
                if xdim in view_da.coords:
                    view_da = view_da.sel({xdim: slice(x0, x1)})
                if ydim in view_da.coords:
                    view_da = view_da.sel({ydim: slice(y0, y1)})
            except (KeyError, TypeError, ValueError):
                view_da = sliced_da

            if view_da.size > 0:
                v_view, V_view = _finite_minmax(view_da)

                # Only override with the view-based clim when it is finite.
                if np.isfinite(v_view) and np.isfinite(V_view):
                    vmin, vmax = v_view, V_view

    # Final safety check: if something still produced non-finite limits,
    # fall back to a cached global range (computed at most once).
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        gmin = full_da.attrs.get("_global_vmin", None)
        gmax = full_da.attrs.get("_global_vmax", None)

        if gmin is None or gmax is None:
            gmin, gmax = _finite_minmax(full_da)
            full_da.attrs["_global_vmin"] = gmin
            full_da.attrs["_global_vmax"] = gmax

        vmin, vmax = gmin, gmax

    # Last-resort defaults keep the plot and colorbar stable for all-NaN slices.
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0

    if vmin == vmax:
        pad = 1.0 if vmin == 0 else 0.01 * abs(vmin)
        vmin -= pad
        vmax += pad

    return vmin, vmax


def _detect_vertical_dim(da):
    """Return the name of the vertical dimension for a field.

    Supports the common DALES-style vertical dimensions. If none of the
    known vertical dimensions are present, a ValueError is raised.
    """

    for dim in ("zt", "zm", "zh"):
        if dim in da.dims:
            return dim
    raise ValueError(f"{da.name} has no vertical dimension")


def get_label_to_var(ds):
    # Build label mapping: "short: long" -> short
    label_to_var = {}

    for v in ds.data_vars:
        long_name = ds[v].attrs.get("long_name", "")
        if long_name:
            label = f"{v}: {long_name}"
        else:
            label = v
        label_to_var[label] = v
    return label_to_var


def get_variable_dim(ds):
    return hv.Dimension(
        "variable",
        values=list(get_label_to_var(ds).keys()),
        label="Variable",
    )


def _build_backend_bounds(ds, xdim, ydim):
    """Build axis bounds backend options when matching coordinates are available."""
    backend_opts = {}
    if xdim in ds.coords:
        backend_opts["x_range.bounds"] = (
            float(ds[xdim].min(skipna=True).values),
            float(ds[xdim].max(skipna=True).values),
        )
    if ydim in ds.coords:
        backend_opts["y_range.bounds"] = (
            float(ds[ydim].min(skipna=True).values),
            float(ds[ydim].max(skipna=True).values),
        )
    return backend_opts


def _make_clim_controls(controller):
    """Return shared clim controls wired to a controller with auto/trigger params."""
    auto_checkbox = pn.widgets.Checkbox.from_param(
        controller.param.auto, name="Auto clim from view"
    )
    button_view = pn.widgets.Button(name="Reset clim from view", button_type="primary")
    button_global = pn.widgets.Button(
        name="Reset clim from global", button_type="default"
    )

    def _on_click(event):
        controller.trigger += 1

    def _on_global_click(event):
        controller.auto = False
        controller.trigger = 0

    button_view.on_click(_on_click)
    button_global.on_click(_on_global_click)
    return auto_checkbox, button_view, button_global


def determine_clim_and_cmap(vmin, vmax):
    if vmin < 0 and vmax > 0:
        if abs(vmin) > 10 * abs(vmax) or abs(vmax) > 10 * abs(vmin):
            # If one side is much larger, use non-symmetric clim to avoid losing detail
            clim = (vmin, vmax)
            cmap = "viridis"  # sequential colormap
        else:
            vmax_abs = max(abs(vmin), abs(vmax))
            clim = (-vmax_abs, vmax_abs)
            cmap = "RdBu_r"  # diverging colormap
    else:
        clim = (vmin, vmax)
        cmap = "viridis"  # sequential colormap
    return clim, cmap


import functools
import traceback
import sys


def catchall(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return pn.Row(hv.Text(0.5, 0.5, f"Error: {e}"))

    return wrapper
