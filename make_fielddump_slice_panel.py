from make_slice_panel import make_slice_panel


def make_fielddump_slice_panel(ds):
    """Deprecated wrapper; use make_slice_panel(ds) instead.

    This now simply delegates to make_slice_panel(ds, slice_dim=None), which
    exposes a "Slice dim" selector replicating the original behavior while
    sharing the unified slice-panel implementation.
    """

    return make_slice_panel(ds, slice_dim=None)
