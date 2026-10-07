import numpy as np
import panel as pn
import param
from bokeh.models import CustomJS

from make_slice_panel import make_slice_panel


class _DenseVTKVolume(pn.pane.VTKVolume):
    def _get_model(self, doc, root=None, parent=None, comm=None):
        model = super()._get_model(doc, root, parent, comm)
        opacity = CustomJS(args={"volume": model}, code="""
            const applyOpacity = () => {
                const renderer = volume.renderer_el;
                if (!renderer) return;
                const actor = renderer.getRenderer().getVolumes()[0];
                if (!actor) return;
                const grid = actor.getMapper().getInputData();
                const bounds = grid.getBounds();
                const diagonal = Math.hypot(
                    bounds[1] - bounds[0], bounds[3] - bounds[2],
                    bounds[5] - bounds[4]
                );
                const distance = diagonal / Math.max(...grid.getDimensions());
                actor.getProperty().setScalarOpacityUnitDistance(
                    0, Math.max(distance / 3, Number.EPSILON)
                );
                renderer.getRenderWindow().render();
            };
            requestAnimationFrame(() => requestAnimationFrame(applyOpacity));
        """)
        model.js_on_change("data", opacity)
        model.js_on_change("camera", opacity)
        return model


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

    def volume_view(self):
        pane = self.view()
        fullscreen = pn.widgets.Button(
            name="Fullscreen", icon="arrows-maximize", width=140
        )
        fullscreen.js_on_click(args={"volume": pane}, code="""
            const renderer = volume.renderer_el;
            if (!renderer) return;
            if (document.fullscreenElement) {
                document.exitFullscreen();
                return;
            }
            const host = renderer.getContainer().getRootNode().host;
            if (!host || !host.requestFullscreen) return;
            const previous = {
                width: volume.width, height: volume.height,
                sizing_mode: volume.sizing_mode
            };
            const resize = () => {
                if (document.fullscreenElement !== host) return;
                volume.sizing_mode = "fixed";
                volume.width = window.innerWidth;
                volume.height = window.innerHeight;
                requestAnimationFrame(() => volume.renderer_el?.resize());
            };
            const changed = () => {
                if (document.fullscreenElement === host) {
                    resize();
                } else {
                    volume.width = previous.width;
                    volume.height = previous.height;
                    volume.sizing_mode = previous.sizing_mode;
                    document.removeEventListener("fullscreenchange", changed);
                    window.removeEventListener("resize", resize);
                    requestAnimationFrame(() => volume.renderer_el?.resize());
                }
            };
            document.addEventListener("fullscreenchange", changed);
            window.addEventListener("resize", resize);
            host.requestFullscreen().catch((error) => {
                document.removeEventListener("fullscreenchange", changed);
                window.removeEventListener("resize", resize);
                console.warn("Could not enter fullscreen", error);
            });
        """)
        return pn.Column(
            pn.Row(pn.Spacer(sizing_mode="stretch_width"), fullscreen),
            pn.Row(
                pane.controls(jslink=True),
                pane,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    @param.depends("time", "z_stride", "xy_stride", "vertical_exaggeration", watch=True)
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
            self._volume = _DenseVTKVolume(
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
    time_max = viewer.param.time.bounds[1]
    time_slider = pn.widgets.IntSlider(
        name="Time index", start=0, end=max(time_max, 1), value=viewer.time,
        disabled=time_max == 0, sizing_mode="stretch_width",
    )
    time_slider.param.watch(
        lambda event: setattr(viewer, "time", event.new), "value_throttled"
    )
    controls = pn.Column(time_slider, pn.Param(
        viewer,
        parameters=["z_stride", "xy_stride", "vertical_exaggeration"],
        show_name=False,
        sizing_mode="stretch_width",
    ), sizing_mode="stretch_width")
    volume = pn.Column(
        controls,
        pn.panel(viewer.volume_view, defer_load=True),
        sizing_mode="stretch_width",
        min_height=600,
    )
    return pn.Tabs(
        ("Slices", slices),
        ("ql volume", volume),
        dynamic=True,
        sizing_mode="stretch_width",
    )
