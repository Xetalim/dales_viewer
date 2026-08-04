import holoviews as hv
import numpy as np


def _profile_element(ds, da, label, ydim, time_index):
    n_time = da.sizes.get("time", len(da["time"]))
    if n_time == 0:
        return hv.Curve([])

    idx = max(0, min(int(time_index), n_time - 1))
    prof = da.isel(time=idx)
    z_vals = ds[ydim].values if ydim in ds.coords else prof[ydim].values
    x_vals = prof.values

    finite_vals = np.asarray(da.values, dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size == 0:
        xmin, xmax = 0.0, 1.0
    else:
        xmin = float(finite_vals.min())
        xmax = float(finite_vals.max())
    if xmin == xmax:
        pad = 1.0 if xmin == 0 else 0.01 * abs(xmin)
        xmin -= pad
        xmax += pad

    curve = hv.Curve((x_vals, z_vals), da.name or "var", ydim).opts(
        xlabel=label,
        ylabel=ydim,
        title=label,
        xlim=(xmin, xmax),
        frame_height=300,
        responsive="width",
        framewise=True,
        shared_axes=False,
        axiswise=True,
    )

    return curve
