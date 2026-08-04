import panel as pn
import pathlib
import logging

import holoviews as hv

from file_catalog_config import (
    available_specs,
    load_file_catalog,
    resolve_runtime_specs,
)
from file_function_registry import make_builder_registry, make_loader_registry

hv.extension("bokeh")
pn.extension(sizing_mode="stretch_width")

# Ensure that panning/zooming in one plot does not affect others by
# disabling axis sharing globally. Individual panels can still override
# this per-element if needed.
hv.opts.defaults(
    hv.opts.Image(shared_axes=False, axiswise=True),
    hv.opts.QuadMesh(shared_axes=False, axiswise=True),
    hv.opts.Curve(shared_axes=False, axiswise=True),
    hv.opts.HeatMap(shared_axes=False, axiswise=True),
)

FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


def make_app(folder):
    output_path = pathlib.Path(folder)
    ds_cache = {}

    catalog_path = pathlib.Path(__file__).with_name("file_catalog.yaml")
    raw_specs = load_file_catalog(catalog_path)
    found_specs = available_specs(raw_specs, output_path)
    found_ids = {spec["id"] for spec in found_specs}
    missing_specs = [spec for spec in raw_specs if spec["id"] not in found_ids]

    loader_registry = make_loader_registry(output_path)
    builder_registry = make_builder_registry(ds_cache)
    runtime_specs = resolve_runtime_specs(
        found_specs, loader_registry, builder_registry
    )

    if not runtime_specs:
        return pn.Column(
            pn.pane.Markdown(f"# {output_path.resolve()}"),
            pn.pane.Markdown("No known DALES output files were found in this folder."),
            sizing_mode="stretch_width",
        )

    title_to_spec = {spec["title"]: spec for spec in runtime_specs}
    loaded_cards = {}  # key -> pn.Card, only populated after successful load
    _busy = [False]  # re-entrancy guard for the selection watcher
    available_titles = [spec["title"] for spec in runtime_specs]

    main_col = pn.Column(sizing_mode="stretch_width", styles={"min-width": "0"})
    status = pn.pane.Markdown("", sizing_mode="stretch_width")
    sidebar_visible = pn.widgets.Toggle(
        name="☰",
        value=True,
        button_type="default",
        width=48,
        height=40,
    )
    select_all_button = pn.widgets.Button(name="Select all", button_type="primary")

    file_selector = pn.widgets.CheckBoxGroup(
        name="Files",
        options=available_titles,
        value=[],
    )

    missing_buttons = [
        pn.widgets.Button(
            name=spec["title"], disabled=True, sizing_mode="stretch_width"
        )
        for spec in missing_specs
    ]

    def _refresh_main():
        panels = [
            loaded_cards[title_to_spec[t]["key"]]
            for t in file_selector.value
            if title_to_spec[t]["key"] in loaded_cards
        ]
        main_col.objects = panels or [
            pn.pane.Markdown("Select files from the sidebar to load them.")
        ]

    def _on_selection_change(event):
        if _busy[0]:
            return
        _busy[0] = True
        try:
            failed = []
            for title in event.new:
                spec = title_to_spec[title]
                key = spec["key"]
                if key in loaded_cards:
                    continue
                data = spec["load_fn"]()
                ds_cache[key] = data
                if data is None:
                    failed.append(title)
                else:
                    loaded_cards[key] = spec["build_fn"](data)
            if failed:
                file_selector.value = [t for t in event.new if t not in failed]
                status.object = f"Could not load: {', '.join(failed)}"
            else:
                status.object = ""
            _refresh_main()
        finally:
            _busy[0] = False

    def _select_all(_event):
        file_selector.value = list(available_titles)

    def _toggle_sidebar(event):
        sidebar.visible = event.new
        sidebar_visible.name = "☰"

    file_selector.param.watch(_on_selection_change, "value")
    select_all_button.on_click(_select_all)
    _refresh_main()

    sidebar = pn.Column(
        pn.pane.Markdown(f"**{output_path.resolve()}**"),
        pn.Spacer(height=8),
        select_all_button,
        pn.Spacer(height=8),
        pn.pane.Markdown("**Available files**"),
        file_selector,
        pn.Spacer(height=12),
        pn.pane.Markdown("**Missing files**"),
        *missing_buttons,
        status,
        width=280,
        styles={"padding": "12px", "background": "#f8f8f8", "overflow-y": "auto"},
        sizing_mode="stretch_height",
    )

    sidebar_visible.param.watch(_toggle_sidebar, "value")

    return pn.Row(
        pn.Column(
            sidebar_visible,
            width=60,
            sizing_mode="stretch_height",
            styles={"padding": "8px 0 0 8px"},
        ),
        sidebar,
        main_col,
        sizing_mode="stretch_both",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DALES viewer")
    parser.add_argument(
        "folder", nargs="?", default=".", help="Path to DALES output folder"
    )
    parser.add_argument("--port", type=int, default=5007)
    parser.add_argument("--dask", type=int, default=38227)
    parser.add_argument("--show", type=bool, default=False)
    args = parser.parse_args()

    pn.config.console_output = "accumulate"

    import traceback, sys

    def _print_exception(ex):
        traceback.print_exception(type(ex), ex, ex.__traceback__, file=sys.stderr)

    pn.extension(exception_handler=_print_exception)
    # client = Client(f"tcp://127.0.0.1:{args.dask}")
    app = make_app(args.folder)
    print("Launching DALES viewer... on port", args.port)
    pn.serve(app, show=args.show, port=args.port, admin=True)
