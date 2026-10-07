import numpy as np
import panel as pn
import param

from make_slice_panel import make_slice_panel


class DalesVolumeViewer(param.Parameterized):
    """Render ql on a regular VTK grid using mean coordinate spacing."""

    time = param.Integer(default=0, bounds=(0, None), label="Time index")
    z_stride = param.Integer(default=1, bounds=(1, None), label="Z stride")
    xy_stride = param.Integer(default=1, bounds=(1, None), label="XY stride")
    vertical_exaggeration = param.Number(
        default=3.0, bounds=(0.1, 20.0), label="Vertical exaggeration"
    )

    def __init__(self, ds, **params):
        self.ds = ds
        super().__init__(**params)
        self.param.time.bounds = (0, max(ds["ql"].sizes.get("time", 1) - 1, 0))
        self.param.z_stride.bounds = (1, ds["ql"].sizes["zt"])
        self.param.xy_stride.bounds = (
            1,
            min(ds["ql"].sizes["xt"], ds["ql"].sizes["yt"]),
        )
        self._volume = None

    def view(self):
        if self._volume is None:
            self._update_volume()
        return self._volume

    @param.depends(
        "time", "z_stride", "xy_stride", "vertical_exaggeration", watch=True
    )
    def _update_volume(self):
        ql = self.ds["ql"]
        if "time" in ql.dims:
            ql = ql.isel(time=self.time)
        ql = (
            ql.isel(
                xt=slice(None, None, self.xy_stride),
                yt=slice(None, None, self.xy_stride),
                zt=slice(None, None, self.z_stride),
            )
            .transpose("xt", "yt", "zt")
            .load()
        )
        spacing = []
        for dim, stride in (
            ("xt", self.xy_stride),
            ("yt", self.xy_stride),
            ("zt", self.z_stride),
        ):
            coord = ql[dim]
            original = self.ds[dim]
            if coord.size > 1:
                spacing.append(float(coord.diff(dim).mean()))
            elif original.size > 1:
                spacing.append(float(original.diff(dim).mean()) * stride)
            else:
                spacing.append(1.0)

        spacing[2] *= self.vertical_exaggeration
        data = np.nan_to_num(
            ql.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        origin = tuple(float(ql[dim][0]) for dim in ("xt", "yt", "zt"))
        if self._volume is None:
            self._volume = pn.pane.VTKVolume(
                data,
                spacing=tuple(spacing),
                origin=origin,
                display_volume=True,
                edge_gradient=0,
                sampling=0.4,
                orientation_widget=True,
                controller_expanded=True,
                sizing_mode="stretch_width",
                height=520,
            )
        else:
            self._volume.param.update(
                object=data, spacing=tuple(spacing), origin=origin
            )


def make_fielddump_slice_panel(ds):
    """Build slices and an optional reactive ql volume view."""
    slices = make_slice_panel(ds, slice_dim=None)
    if "ql" not in ds:
        return slices
    ql = ds["ql"]
    spatial_dims = {"xt", "yt", "zt"}
    if (
        not spatial_dims.issubset(ql.dims)
        or set(ql.dims) - spatial_dims - {"time"}
        or any(size == 0 for size in ql.sizes.values())
        or not spatial_dims.issubset(ds.coords)
    ):
        return slices

    viewer = DalesVolumeViewer(ds)
    controls = pn.Param(
        viewer,
        parameters=["time", "z_stride", "xy_stride", "vertical_exaggeration"],
        show_name=False,
        sizing_mode="stretch_width",
    )
    volume = pn.Column(
        controls,
        pn.panel(viewer.view, defer_load=True),
        sizing_mode="stretch_width",
        min_height=600,
    )
    return pn.Tabs(
        ("Slices", slices),
        ("ql volume", volume),
        dynamic=True,
        sizing_mode="stretch_width",
    )
