import holoviews as hv


def _profile_element(ds, da, label, ydim, time_index):
    n_time = da.sizes.get("time", len(da["time"]))
    if n_time == 0:
        return hv.Curve([])

    idx = max(0, min(int(time_index), n_time - 1))
    prof = da.isel(time=idx)
    z_vals = ds[ydim].values
    x_vals = prof.values

    xmin = float(da.min(skipna=True).values)
    xmax = float(da.max(skipna=True).values)
    curve = hv.Curve((x_vals, z_vals), da.name or "var", ydim).opts(
        xlabel=label,
        ylabel=ydim,
        xlim=(xmin, xmax),
        height=300,
        responsive=True,
    )

    return curve
