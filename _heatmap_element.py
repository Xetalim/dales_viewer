from helpers import plot_2d_heatmap


def _heatmap_element(ds, da, label, ydim, vmin, vmax):
    return plot_2d_heatmap(
        da,
        xdim="time",
        ydim=ydim,
        title=label,
        bounds_source=ds,
        vmin=vmin,
        vmax=vmax,
    )
