import numpy as np


def profiles(ds_profiles):
    ds_sel = ds_profiles  # .isel(time=slice(5, None))
    ln = len(ds_sel.zt)
    return ds_sel.interp(
        zt=np.linspace(ds_sel.zt.min().values, ds_sel.zt.max().values, num=ln),
        zm=np.linspace(ds_sel.zm.min().values, ds_sel.zm.max().values, num=ln),
    )


def cape(ds_cape):
    ds_mean = ds_cape.mean(dim=("xt", "yt"), keep_attrs=True, skipna=True)
    ds_std = ds_cape.std(dim=("xt", "yt"), keep_attrs=True, skipna=True)
    return ds_mean, ds_std


def crosses(ds_crosses):
    ds_mean = ds_crosses.mean(dim=("xt", "yt"), keep_attrs=True, skipna=True)
    ds_std = ds_crosses.std(dim=("xt", "yt"), keep_attrs=True, skipna=True)
    return ds_mean, ds_std
