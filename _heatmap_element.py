from helpers import determine_clim_and_cmap


def _heatmap_element(ds, da, label, ydim, vmin, vmax):
    clim, cmap = determine_clim_and_cmap(vmin, vmax)

    return da.hvplot(
        x="time",
        y=ydim,
        cmap=cmap,
        clim=clim,
        colorbar=True,
        title=label,
        height=300,
        responsive=True,
    )
