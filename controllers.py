import param


class ClimController(param.Parameterized):
    label = param.ObjectSelector(default=None, objects=[], label="Variable")
    auto = param.Boolean(default=True, doc="Automatically set clim from current view")
    trigger = param.Integer(default=0, doc="Manual trigger to recompute clim from view")
    mode = param.ObjectSelector(
        default="heatmap", objects=["heatmap", "profile"], label="View"
    )
    time_index = param.Integer(default=0, bounds=(0, 0), label="Time index")


class SliceController(param.Parameterized):
    var = param.ObjectSelector(default=None, objects=[], label="Variable")
    index = param.Integer(default=0, bounds=(0, 0), label="Slice index")
    time_index = param.Integer(default=0, bounds=(0, 0), label="Time index")
    auto = param.Boolean(default=True, doc="Automatically set clim from current view")
    trigger = param.Integer(default=0, doc="Manual trigger to recompute clim from view")
    symmetric_cmap = param.Boolean(
        default=False,
        doc="Force symmetric color limits around 0 for this slice",
    )


class FielddumpSliceController(param.Parameterized):
    var = param.ObjectSelector(default=None, objects=[], label="Variable")
    dim = param.ObjectSelector(default=None, objects=[], label="Slice dim")
    index = param.Integer(default=0, bounds=(0, 0), label="Slice index")
    time_index = param.Integer(default=0, bounds=(0, 0), label="Time index")
    auto = param.Boolean(default=True, doc="Automatically set clim from current view")
    trigger = param.Integer(default=0, doc="Manual trigger to recompute clim from view")
    symmetric_cmap = param.Boolean(
        default=False,
        doc="Force symmetric color limits around 0 for this slice",
    )


def _select_plot_dims(other_dims):
    """Select x/y dims for a 2D plot from remaining dimensions."""
    if len(other_dims) != 2:
        return None, None

    if "zt" in other_dims:
        ydim = "zt"
    elif "zm" in other_dims:
        ydim = "zm"
    else:
        ydim = other_dims[1]

    if any(d in other_dims for d in ("xt", "xm")) and any(
        d in other_dims for d in ("yt", "ym")
    ):
        if "yt" in other_dims:
            ydim = "yt"
        elif "ym" in other_dims:
            ydim = "ym"

    xdim = other_dims[0] if other_dims[0] != ydim else other_dims[1]
    return xdim, ydim
