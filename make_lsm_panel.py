import holoviews as hv
import panel as pn
import param
from holoviews import streams

from helpers import catchall, make_plot_with_controls_layout, plot_2d_heatmap


class LSMController(param.Parameterized):
    category = param.ObjectSelector(default=None, objects=[], label="Land use category")
    variable = param.ObjectSelector(default=None, objects=[], label="Variable")
    z_index = param.Integer(default=0, bounds=(0, 0), label="Soil layer index")


def _read_lu_suffixes(ds):
    """Read land-use short names from the lushort char variable."""
    if "lushort" not in ds:
        return []

    raw = ds["lushort"].values
    suffixes = []
    for row in raw:
        if hasattr(row, "tobytes"):
            value = row.tobytes().decode("utf-8", errors="ignore").strip("\x00").strip()
        else:
            value = str(row).strip("\x00").strip()
        if value:
            suffixes.append(value)

    return suffixes


def _parse_lsm_categories(ds):
    """Group LSM variables by land-use suffix (from lushort) and soil."""
    lu_suffixes = _read_lu_suffixes(ds)
    categories = {suffix: {} for suffix in lu_suffixes}
    categories["soil"] = {}

    for var_name in ds.data_vars:
        if ds[var_name].dtype.kind in ("S", "U", "O"):
            continue

        matched = False
        for suffix in lu_suffixes:
            if var_name.endswith(f"_{suffix}"):
                base_name = var_name[: -(len(suffix) + 1)]
                categories[suffix][base_name] = var_name
                matched = True
                break

        if not matched and "z" in ds[var_name].dims:
            categories["soil"][var_name] = var_name

    return categories


def make_lsm_panel(ds):
    categories = _parse_lsm_categories(ds)
    cat_names = [name for name in categories if categories[name]]

    controller = LSMController()
    controller.param["category"].objects = cat_names
    controller.category = cat_names[0] if cat_names else None

    default_vars = list(categories[cat_names[0]].keys()) if cat_names else []
    controller.param["variable"].objects = default_vars
    controller.variable = default_vars[0] if default_vars else None

    if "z" in ds.dims:
        n_z = int(ds.sizes["z"])
        controller.param["z_index"].bounds = (0, max(n_z - 1, 0))

    def _update_vars(_event):
        cat = controller.category
        var_list = list(categories.get(cat, {}).keys())
        controller.param["variable"].objects = var_list
        controller.variable = var_list[0] if var_list else None

    controller.param.watch(_update_vars, "category")

    param_stream = streams.Params(
        controller, parameters=["category", "variable", "z_index"]
    )

    @catchall
    def plot_fn(**_kwargs):
        cat = controller.category
        var_base = controller.variable
        z_idx = controller.z_index

        if var_base is None:
            return hv.Curve([])

        cat_vars = categories.get(cat, {})
        var_name = cat_vars.get(var_base, var_base)
        if var_name not in ds:
            return hv.Curve([])

        da = ds[var_name]
        if "z" in da.dims:
            da = da.isel(z=z_idx)

        long_name = da.attrs.get("long_name", "")
        title = f"{var_name}: {long_name}" if long_name else var_name
        return plot_2d_heatmap(
            da,
            xdim="x",
            ydim="y",
            title=title,
            bounds_source=ds,
        )

    dmap = hv.DynamicMap(plot_fn, streams=[param_stream]).opts(
        framewise=False,
        shared_axes=False,
        axiswise=True,
    )

    plot = pn.panel(dmap, sizing_mode="stretch_width")

    cat_select = pn.widgets.Select.from_param(
        controller.param.category, name="Land use category"
    )
    var_select = pn.widgets.Select.from_param(
        controller.param.variable, name="Variable"
    )
    z_slider = pn.widgets.IntSlider.from_param(
        controller.param.z_index, name="Soil layer (z)"
    )
    z_slider.visible = controller.category == "soil"

    def _toggle_z(_event):
        z_slider.visible = controller.category == "soil"

    controller.param.watch(_toggle_z, "category")

    controls = pn.Column(cat_select, var_select, z_slider, sizing_mode="stretch_width")
    return make_plot_with_controls_layout(plot, controls)
