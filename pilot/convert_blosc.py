#!/usr/bin/env python3
"""Re-encode NetCDF files -> Blosc-ZSTD + downcast float64 vars (obs) to float32.

Parallel, resumable (skips completed). Writes via h5netcdf (needs hdf5plugin);
the result is read back transparently by the default netcdf4 engine as long as
`import hdf5plugin` runs in the reader process.

Usage: convert_blosc.py <src_dir> <dst_dir> <glob> <limit> <nprocs> [clevel]
  e.g. convert_blosc.py .../unc/train_data .../blosc/train_data '2021-01-*.nc' 0 24 3
"""
import sys, os, glob, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import hdf5plugin  # registers HDF5 Blosc filter; must import before writing/reading
import numpy as np
import xarray as xr

CLEVEL = 3  # overridden by argv[6]

def blosc_enc(clevel):
    return dict(hdf5plugin.Blosc(cname="zstd", clevel=clevel,
                                 shuffle=hdf5plugin.Blosc.SHUFFLE))

def convert_one(args):
    src, dst, clevel = args
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return (dst, 0.0, True, 0)
    t = time.time()
    with xr.open_dataset(src) as ds:
        ds.load()
        # downcast any float64 data var (the obs sta_* vars) to float32 — the model
        # consumes float32/bf16, so this is bit-faithful and halves obs bytes.
        expected = {}
        for v in ds.data_vars:
            if ds[v].dtype == np.float64:
                ds[v] = ds[v].astype(np.float32)
            expected[v] = ds[v].values  # post-cast values we must reproduce exactly
        enc = {v: blosc_enc(clevel) for v in ds.data_vars}
        tmp = dst + ".tmp"
        ds.to_netcdf(tmp, engine="h5netcdf", encoding=enc)

    # verify-after-write: reopen and assert every var round-trips EXACTLY (lossless
    # codec + intended float32 cast). Fail loud rather than ship silently-wrong data.
    with xr.open_dataset(tmp) as r:
        for v, exp in expected.items():
            got = r[v].values
            if got.dtype != exp.dtype or not np.array_equal(got, exp, equal_nan=True):
                os.remove(tmp)
                raise ValueError(f"VERIFY FAILED {os.path.basename(dst)} var={v} "
                                 f"dtype {got.dtype} vs {exp.dtype}")
    os.replace(tmp, dst)
    return (dst, time.time() - t, False, os.path.getsize(dst))

def main():
    src_dir, dst_dir, pat, limit, nprocs = sys.argv[1:6]
    clevel = int(sys.argv[6]) if len(sys.argv) > 6 else CLEVEL
    limit, nprocs = int(limit), int(nprocs)
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, pat)))
    if limit > 0:
        files = files[:limit]
    jobs = [(f, os.path.join(dst_dir, os.path.basename(f)), clevel) for f in files]
    print(f"blosc-zstd-L{clevel}: {len(jobs)} files {src_dir} -> {dst_dir} "
          f"(nprocs={nprocs})", flush=True)
    t0 = time.time(); done = skipped = 0; total_bytes = 0
    with ProcessPoolExecutor(max_workers=nprocs) as ex:
        futs = [ex.submit(convert_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            _, dt, was_skip, nbytes = fut.result()
            done += 1; skipped += int(was_skip); total_bytes += nbytes
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} (skipped {skipped}) "
                      f"elapsed {time.time()-t0:.0f}s", flush=True)
    gb = total_bytes / 1e9
    print(f"DONE {done} files ({skipped} pre-existing) in {time.time()-t0:.0f}s, "
          f"{gb:.1f} GB written", flush=True)

if __name__ == "__main__":
    main()
