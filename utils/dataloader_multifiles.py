import os
import glob
import torch
import logging
import numpy as np
import pandas as pd
import xarray as xr

try:
    # Registers the HDF5 Blosc filter so Blosc-ZSTD-encoded .nc files decode
    # transparently under the default netcdf4 engine. No-op if data is zlib.
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from torch.utils.data import DataLoader, Dataset, DistributedSampler
# from torch.utils.data.distributed import DistributedSampler
from utils.misc_functions import as_bool

####################
def get_data_loader(params, files_location, distributed, train):
    """Build a loader for one split. `train` selects both the sampler behavior and
    the obs-dropout regime in GetDataset (random+block masks for train, a fixed
    deterministic mask per file for valid). Always returns a sampler slot; it is
    None when not distributed."""
    dataset = GetDataset(params, files_location, train)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler=None

    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        prefetch_factor=params.prefetch_factor if params.num_data_workers > 0 else None,
        shuffle=False,  # (sampler is none),
        sampler=sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, dataset, sampler

####################

class GetDataset(Dataset):
    def __init__(self, params, files_location, train):
        self.params = params
        self.train = train
        self.files_location = files_location
        # self.n_in_channels = params.n_in_channels
        # self.n_out_channels = params.n_out_channels
        # self.add_noise = params.add_noise if train else false
        self.get_file_stats()

    ###

    def get_file_stats(self):
        self.file_paths = glob.glob(self.files_location + "/*.nc")
        self.file_paths.sort()

        # Validation runs every epoch now (checkpoint selection depends on it), and the
        # full valid split is as large as the train split. Score a fixed evenly-strided
        # subset instead: comparability across epochs is what the metric needs, not
        # coverage. 0 = use every file.
        max_files = int(getattr(self.params, "valid_max_files", 0) or 0)
        if (not self.train) and max_files and len(self.file_paths) > max_files:
            stride = len(self.file_paths) / max_files
            self.file_paths = [self.file_paths[int(i * stride)] for i in range(max_files)]

        self.num_samples_total = len(self.file_paths)

        print(f"Getting file stats from {self.file_paths[0]}")
        ds = xr.open_dataset(self.file_paths[0])
    
        # Reversed from original ADAF code - we need x to be lon and y to be lat
        self.org_img_shape_x = ds["hrrr_t"].shape[1]
        self.org_img_shape_y = ds["hrrr_t"].shape[0]
        ds.close()

    ###

    def open_file(self, hour_idx):
        return xr.open_dataset(self.file_paths[hour_idx])

    ###

    def __len__(self):
        return self.num_samples_total

    ###

    def _dropout_rng(self, hour_idx):
        """One fresh Generator per sample -- never np.random.seed(), which poisons the
        worker's global RNG state and (with a fixed seed) hands every sample the same
        draw.

        train: entropy from torch.initial_seed(), which PyTorch varies per worker AND
        per epoch (workers re-fork each epoch), so masks differ across samples and
        across epochs.
        valid: entropy from (obs_mask_seed, hour_idx) only -- the same file gets the
        same held-out stations at every epoch and in every run, so the validation
        metric is comparable across epochs and across runs.
        """
        base = int(getattr(self.params, "obs_mask_seed", 0) or 0)
        if self.train:
            return np.random.default_rng([torch.initial_seed() % (2 ** 63), hour_idx, base])
        return np.random.default_rng([base, hour_idx])

    ###

    def _heldout_mask(self, obs, hour_idx):
        """(y,x) float32 mask: 1 at cells whose station is hidden from the model input.

        Station set = every cell reporting ANY variable at ANALYSIS time (obs[:, -1]).
        The old code used obs[0,0] -- variable 0 in the OLDEST time bin -- which both
        missed stations that only report at analysis time and held out stations with no
        analysis-time truth to score against.

        train: per-sample rate r ~ U(ratio_min, ratio_max) so the model learns
        density-aware spreading rather than tuning to one thinning fraction, plus (with
        probability hold_out_block_prob) a rectangular blackout that forces genuinely
        long-range inference -- with random thinning alone a dropped station almost
        always keeps an intact neighbor 20-30 km away.
        valid: fixed hold_out_obs_ratio, no block.
        """
        rng = self._dropout_rng(hour_idx)
        ny, nx = obs.shape[-2], obs.shape[-1]

        station_cells = np.flatnonzero((obs[:, -1] != 0).any(axis=0).ravel())

        if self.train:
            r_min = float(getattr(self.params, "hold_out_ratio_min", self.params.hold_out_obs_ratio))
            r_max = float(getattr(self.params, "hold_out_ratio_max", self.params.hold_out_obs_ratio))
            ratio = float(rng.uniform(r_min, r_max)) if r_max > r_min else r_min
        else:
            ratio = float(self.params.hold_out_obs_ratio)

        hold_out_num = int(len(station_cells) * ratio)
        held = rng.choice(station_cells, size=hold_out_num, replace=False) if hold_out_num else []

        # dtype matches obs (float32) so obs*(1-obs_mask) stays float32 -- a default
        # float64 zeros here silently upcasts inp_obs, poisoning the concatenate below
        # to float64 (~2x the work + H2D bytes).
        obs_mask = np.zeros(ny * nx, dtype=obs.dtype)
        obs_mask[held] = 1
        obs_mask = obs_mask.reshape(ny, nx)

        block_prob = float(getattr(self.params, "hold_out_block_prob", 0.0) or 0.0)
        if self.train and block_prob > 0 and rng.random() < block_prob:
            b_min = int(getattr(self.params, "hold_out_block_min", 64))
            b_max = int(getattr(self.params, "hold_out_block_max", 192))
            bh = int(rng.integers(b_min, b_max + 1))
            bw = int(rng.integers(b_min, b_max + 1))
            y0 = int(rng.integers(0, max(ny - bh, 0) + 1))
            x0 = int(rng.integers(0, max(nx - bw, 0) + 1))
            # Setting the whole rectangle (not just its station cells) is equivalent:
            # obs*(1-mask) is a no-op at cells with no ob, and every consumer of the
            # mask intersects it with the obs mask.
            obs_mask[y0:y0 + bh, x0:x0 + bw] = 1

        return obs_mask

    ###

    def __getitem__(self, hour_idx):
        with self.open_file(hour_idx) as ds:

            #Load lons, lats, topography
            # .to_numpy() avoids the double materialization of np.array(da.to_array()):
            # to_array() already builds the stacked array; np.array() copied it a 2nd time.
            lon = ds.coords["lon"].to_numpy()[:self.params.img_size_x]
            lat = ds.coords["lat"].to_numpy()[:self.params.img_size_y]
            topo = ds[["z"]].to_array().to_numpy()[:, : self.params.img_size_y, : self.params.img_size_x]

            # Obs-source tag per grid cell (as written by the prep: 1 = mesonet, 2 =
            # METAR, 0 = unset/fill). Loaded early so it can BOTH (a) filter which obs the
            # model trains on (train_obs_source below) and (b) be returned for the
            # source-scoped held-out metric (heldout_metric_source). Zeros if a file
            # predates the tag, which then drops it from metar/mesonet scoring exactly as
            # heldout_eval does. See CLAUDE.md / memory metar-only-eval-kills-wind-ceiling.
            if "obs_source" in ds:
                obs_source = ds["obs_source"].to_numpy()[
                    : self.params.img_size_y, : self.params.img_size_x].astype(np.int8)
            else:
                obs_source = np.zeros((self.params.img_size_y, self.params.img_size_x), dtype=np.int8)

            #Load HRRR fields
            if len(self.params.inp_hrrr_vars) != 0:
                inp_hrrr = ds[self.params.inp_hrrr_vars].to_array().to_numpy()[:, :self.params.img_size_y, :self.params.img_size_x]
                inp_hrrr = np.squeeze(inp_hrrr)

                # Field mask: 1 where valid (non-zero), else 0. Single vectorized pass
                # instead of copy()+boolean fancy-index assignment (which was two passes
                # over the full grid plus an allocation).
                field_mask = (inp_hrrr != 0).astype(inp_hrrr.dtype)

            #Load obs
            if len(self.params.inp_obs_vars) != 0:
                # A prep bug could size the obs time dim > obs_time_window (a stray bin one
                # hour past analysis time). The -obs_time_window: slice below would then
                # silently shift every obs channel forward an hour and leave obs_tar nearly
                # empty. Refuse to train on such a file rather than corrupt the batch.
                n_bins = ds.sizes["obs_time_window"]
                if n_bins != self.params.obs_time_window:
                    raise ValueError(
                        f"{self.file_paths[hour_idx]}: obs_time_window dim is {n_bins}, "
                        f"expected {self.params.obs_time_window}"
                    )
                obs = ds[self.params.inp_obs_vars].to_array().to_numpy()[
                    :, -self.params.obs_time_window:, :self.params.img_size_y, :self.params.img_size_x]

                # train_obs_source: restrict which obs network the MODEL trains on by
                # zeroing every station not of the chosen source, in ALL vars x ALL time
                # slices, BEFORE obs_tar / obs_tar_mask / dropout are derived. So the input
                # the model sees, the obs substituted into the target field, and the held-out
                # target all become single-network. "all" (default) = no change. "metar"
                # feeds the model only the ~2285 clean METAR stations (vs ~22k with mesonet)
                # -> tests whether it can reconstruct the all-obs RTMA analysis from METAR
                # alone. NOTE: field_tar (RTMA, 99.3% of the loss) is unchanged -- RTMA
                # assimilated every ob and cannot be un-mixed. Zeros-obs_source files drop
                # ALL obs under metar/mesonet, which _heldout_mask / the loss handle as "no
                # obs here". See [[heldout-metric-source-key]].
                _tsrc = getattr(self.params, "train_obs_source", "all")
                if _tsrc in ("metar", "mesonet"):
                    _code = 2 if _tsrc == "metar" else 1
                    obs = obs * (obs_source == _code)  # broadcasts (H,W) over (var,time,H,W)

                #Get most recent obs as target
                obs_tar = obs[:, -1]

                ## Quality control - commented out because this should be done in the dataset generation script, so doing it again here is pointless overhead, but keeping in for legacy reasons (may need it later, who knows)
                # obs_tar[(obs_tar <= -1) | (obs_tar >= 1)] = 0

                # Obs mask (used to replace target-field values at observed locations
                # later). Single vectorized pass instead of copy()+fancy-index assignment.
                obs_tar_mask = (obs_tar != 0).astype(obs_tar.dtype)

                #Hold out obs; note the held-out obs are still used to replace target field values
                if as_bool(getattr(self.params, "hold_out_obs", False)):
                    heldout_mask = self._heldout_mask(obs, hour_idx)
                    #Final input obs = obs minus held out obs. The broadcast zeroes the
                    #held-out stations in ALL 4 vars x ALL 3 time slices -- "the station
                    #is offline", which is what the model must learn to cope with.
                    inp_obs = obs*(1-heldout_mask)
                else:
                    heldout_mask = np.zeros(obs.shape[-2:], dtype=obs.dtype)
                    inp_obs = obs
                inp_obs = inp_obs.reshape((-1, self.params.img_size_y, self.params.img_size_x))

        #####
        ## Satellite stuff here, when done
        #####

            #Load target (RTMA) fields
            field_tar = ds[self.params.field_tar_vars].to_array().to_numpy()[:, : self.params.img_size_y, : self.params.img_size_x]

            # GPU-assembly path: hand back the *raw* components (field_tar/obs_tar
            # pre-residual) and let the training process build field_obs_tar, apply
            # the residual, and concatenate on the GPU. Keeps the CPU workers light
            # (I/O + decompress only) so fewer are needed -> less core contention on
            # the training rank. Bit-faithful to the CPU path below.
            if as_bool(getattr(self.params, "gpu_assemble", False)):
                return (inp_hrrr, inp_obs, topo, field_tar, obs_tar,
                        field_mask, obs_tar_mask, heldout_mask, obs_source, lat, lon)

            #Replace target field with obs @ observed locations (all obs locations, including those held out previously)
            field_obs_tar = field_tar.copy()
            field_obs_tar[obs_tar_mask == 1] = 0
            field_obs_tar += obs_tar

            if self.params.learn_residual:
                field_tar = field_tar - inp_hrrr
                obs_tar = obs_tar - inp_hrrr
                field_obs_tar = field_obs_tar - inp_hrrr

            #Make final input tensor
            inp = np.concatenate((inp_hrrr, inp_obs, topo), axis=0)
            #Satellite version here when that's done

        return (inp,
                field_tar,
                obs_tar,
                field_obs_tar,
                inp_hrrr,
                lat,
                lon,
                field_mask,
                obs_tar_mask,
                heldout_mask,
                obs_source)
