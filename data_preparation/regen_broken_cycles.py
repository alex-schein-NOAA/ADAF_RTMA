#!/usr/bin/env python3
"""Regenerate the cycles broken by the ceil('h') obs-binning bug.

Those files carry a 4th obs_time_window bin one hour PAST analysis time, which
zeroes obs_mask/obs_source, halves the station count, and (via the dataloader's
`[-obs_time_window:]` slice) shifts every obs channel forward an hour. See
sample_generate.py's `OBS_TIMESTAMP <= analysis_time` filter for the fix.

Regenerates straight to Blosc-ZSTD-L3 + obs-float32, matching convert_blosc.py, so
output can be dropped into data_blosc_combined/ in place. Verifies after write and
refuses to emit a file that still has the bug.

Usage:
  regen_broken_cycles.py --list <file>            # one "<split> <name>.nc" per line
  regen_broken_cycles.py --list <f> --shard i/N   # slurm array shard
"""
import argparse, os, subprocess, sys, tempfile, time

import hdf5plugin  # registers the HDF5 Blosc filter; must precede xarray IO
import numpy as np
import xarray as xr

CLEVEL = 3
DST_ROOT = "/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/data_blosc_combined"
HERE = os.path.dirname(os.path.abspath(__file__))
OBS_TIME_WINDOW = 3


def blosc_enc():
    return dict(hdf5plugin.Blosc(cname="zstd", clevel=CLEVEL,
                                 shuffle=hdf5plugin.Blosc.SHUFFLE))


def regen_one(split, name, workdir, dst_root):
    """Run sample_generate for a single cycle, then re-encode to Blosc-ZSTD-L3."""
    stamp = name[:-3]  # "2021-03-05_00"
    raw_dir = os.path.join(workdir, f"{split}_{stamp}")
    os.makedirs(raw_dir, exist_ok=True)
    raw = os.path.join(raw_dir, name)
    if not os.path.exists(raw):
        cmd = [sys.executable, os.path.join(HERE, "sample_generate.py"),
               "--starting_analysis_time", stamp,
               "--ending_analysis_time", stamp,
               "--save_directory", raw_dir,
               "--obs_source", "combined"]
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(raw):
            raise RuntimeError(f"sample_generate failed for {split}/{name}:\n"
                               f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    dst = os.path.join(dst_root, f"{split}_data", name)
    with xr.open_dataset(raw) as ds:
        ds.load()
        # The bug we are fixing: refuse to ship a file that still has a stray bin.
        n_bins = ds.sizes["obs_time_window"]
        if n_bins != OBS_TIME_WINDOW:
            raise ValueError(f"{split}/{name}: still {n_bins} obs bins after fix")
        if int(ds["obs_mask"].sum()) == 0:
            raise ValueError(f"{split}/{name}: obs_mask still all-zero after fix")

        expected = {}
        for v in ds.data_vars:
            if ds[v].dtype == np.float64:  # obs sta_* -> float32, as convert_blosc.py does
                ds[v] = ds[v].astype(np.float32)
            expected[v] = ds[v].values
        tmp = dst + ".tmp"
        ds.to_netcdf(tmp, engine="h5netcdf", encoding={v: blosc_enc() for v in ds.data_vars})

    with xr.open_dataset(tmp) as r:
        for v, exp in expected.items():
            got = r[v].values
            if got.dtype != exp.dtype or not np.array_equal(got, exp, equal_nan=True):
                os.remove(tmp)
                raise ValueError(f"VERIFY FAILED {split}/{name} var={v}")
    os.replace(tmp, dst)

    for f in os.listdir(raw_dir):
        os.remove(os.path.join(raw_dir, f))
    os.rmdir(raw_dir)
    return os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--shard", default="1/1", help="i/N, 1-based")
    ap.add_argument("--workdir", default=os.environ.get("REGEN_WORKDIR", tempfile.gettempdir()))
    ap.add_argument("--dst-root", default=DST_ROOT, help="overwrite target; use a staging dir to dry-run")
    a = ap.parse_args()

    items = [l.split()[:2] for l in open(a.list) if l.strip()]
    i, n = (int(x) for x in a.shard.split("/"))
    mine = items[i - 1::n]
    print(f"shard {i}/{n}: {len(mine)} of {len(items)} cycles -> {a.dst_root}", flush=True)

    t0 = time.time()
    ok = fail = 0
    for k, (split, name) in enumerate(mine, 1):
        try:
            regen_one(split, name, a.workdir, a.dst_root)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAIL {split}/{name}: {e}", flush=True)
        if k % 10 == 0 or k == len(mine):
            print(f"  {k}/{len(mine)} ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"DONE shard {i}/{n}: ok={ok} fail={fail} in {time.time()-t0:.0f}s", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
