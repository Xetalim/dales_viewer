import xarray as xr


def _meanstd_overlay(da_mean, da_std, title=None):
    """Return an hv overlay of mean line + std band for 1D time series."""
    label = title or (da_mean.name or "var")
    da_std_band = xr.Dataset({"y1": da_mean - da_std, "y2": da_mean + da_std})
    band = da_std_band.hvplot.area(
        x="time",
        y="y1",
        y2="y2",
        alpha=0.3,
        color="gray",
        label="std",
        hover=False,
    )
    line = da_mean.hvplot(x="time", title=label)
    return (line * band).opts(frame_height=300, responsive=True)
