import os
import time
import glob
import math
import torch
import logging
import numpy as np
import pandas as pd
import xarray as xr
import hdf5plugin

from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, Sampler

################################

class FractionalDistributedSampler(Sampler):
    def __init__(self, dataset, sample_fraction, num_replicas=None, rank=None, seed=0, drop_last=True):
        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Distributed package is required for FractionalDistributedSampler")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Distributed package is required for FractionalDistributedSampler")
            rank = torch.distributed.get_rank()

        if sample_fraction <= 0 or sample_fraction > 1:
            raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0
        self.indices = []

        dataset_len = len(self.dataset)
        subset_size = int(dataset_len * float(sample_fraction))
        if dataset_len > 0:
            subset_size = max(1, subset_size)
        self.subset_size = min(dataset_len, subset_size)

        if self.drop_last:
            self.num_samples = self.subset_size // self.num_replicas
        else:
            self.num_samples = int(math.ceil(self.subset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def build_indices(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        indices = indices[:self.subset_size]

        if not self.drop_last and len(indices) < self.total_size and len(indices) > 0:
            indices += indices[: self.total_size - len(indices)]
        else:
            indices = indices[:self.total_size]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        return indices

    def __iter__(self):
        self.indices = self.build_indices()
        return iter(self.indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

####################

def get_data_loader(params, files_location, distributed, train):
    dataset = GetDataset(params, files_location, train)

    sample_fraction = float(getattr(params, "train_sample_fraction", 1.0))
    is_training_path = os.path.abspath(files_location) == os.path.abspath(params.train_data_path)
    use_fractional_sampling = train and is_training_path and sample_fraction < 1.0
    train_seed = int(getattr(params, "seed", 0))

    if distributed:
        if train and is_training_path:
            sampler = FractionalDistributedSampler(
                dataset,
                sample_fraction=sample_fraction,
                seed=train_seed,
                drop_last=True,
            )
        else:
            sampler = DistributedSampler(dataset, shuffle=train, seed=train_seed)
    else:
        if is_training_path and train:
            generator = torch.Generator()
            generator.manual_seed(train_seed)
            random_num_samples = len(dataset)
            if use_fractional_sampling:
                random_num_samples = max(1, int(len(dataset) * sample_fraction))
            sampler = RandomSampler(
                dataset,
                replacement=False,
                num_samples=random_num_samples,
                generator=generator,
            )
        else:
            sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        prefetch_factor=params.prefetch_factor if params.num_data_workers > 0 else 1,
        shuffle=False,
        sampler=sampler if train else None,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset

####################

class GetDataset(Dataset):
    def __init__(self, params, files_location, train):
        self.params = params
        self.train = train
        self.files_location = files_location
        self.get_file_stats()

    ###
    
    def get_file_stats(self):
        self.file_paths = glob.glob(self.files_location + "/*.nc")
        self.file_paths.sort()
        max_files = int(getattr(self.params, "valid_max_files", 0) or 0)
        if not self.train and max_files and len(self.file_paths) > max_files:
            stride = len(self.file_paths) / max_files
            self.file_paths = [self.file_paths[int(index * stride)] for index in range(max_files)]
        self.num_samples_total = len(self.file_paths)
    
        print(f"Getting file stats from {self.file_paths[0]}")
        ds = xr.open_dataset(self.file_paths[0])
        
        ds.close()

    ###

    def open_file(self, hour_idx):
        return xr.open_dataset(self.file_paths[hour_idx])

    ###

    def __len__(self):
        return self.num_samples_total

    ###

    def _heldout_mask(self, obs, hour_idx):
        seed = int(getattr(self.params, "obs_mask_seed", 0) or 0)
        if self.train:
            rng = np.random.default_rng([torch.initial_seed() % (2 ** 63), hour_idx, seed])
            minimum = float(getattr(self.params, "hold_out_ratio_min", self.params.hold_out_obs_ratio))
            maximum = float(getattr(self.params, "hold_out_ratio_max", self.params.hold_out_obs_ratio))
            ratio = float(rng.uniform(minimum, maximum)) if maximum > minimum else minimum
        else:
            rng = np.random.default_rng([seed, hour_idx])
            ratio = float(self.params.hold_out_obs_ratio)

        stations = np.flatnonzero((obs[:, -1] != 0).any(axis=0).ravel())
        mask = np.zeros(obs.shape[-2] * obs.shape[-1], dtype=obs.dtype)
        blocking_type = getattr(self.params, "station_blocking_type", "universal")
        if blocking_type not in ("none", "universal", "centered"):
            raise ValueError(f"station_blocking_type must be 'none', 'universal', or 'centered', got {blocking_type!r}")
        if stations.size:
            mask[rng.choice(stations, size=int(stations.size * ratio), replace=False)] = 1
        mask = mask.reshape(obs.shape[-2:])
        block_prob = float(getattr(self.params, "hold_out_block_prob", 0) or 0)
        block_active = self.train and blocking_type != "none" and block_prob > 0 and rng.random() < block_prob
        if block_active:
            min_size = int(getattr(self.params, "hold_out_block_min", 64))
            max_size = int(getattr(self.params, "hold_out_block_max", 192))
            height = int(rng.integers(min_size, max_size + 1))
            width = int(rng.integers(min_size, max_size + 1))
            if blocking_type == "centered":
                rows, cols = np.unravel_index(stations, obs.shape[-2:])
                valid = ((rows >= height // 2) & (rows + height - height // 2 <= obs.shape[-2]) &
                         (cols >= width // 2) & (cols + width - width // 2 <= obs.shape[-1]))
                eligible = stations[valid]
                if eligible.size:
                    center_row, center_col = np.unravel_index(rng.choice(eligible), obs.shape[-2:])
                    row = center_row - height // 2
                    col = center_col - width // 2
                    mask[row:row + height, col:col + width] = 1
                    mask[center_row, center_col] = 0
            else:
                row = int(rng.integers(0, max(obs.shape[-2] - height, 0) + 1))
                col = int(rng.integers(0, max(obs.shape[-1] - width, 0) + 1))
                mask[row:row + height, col:col + width] = 1
        return mask

    ###

    def __getitem__(self, hour_idx):
        with self.open_file(hour_idx) as ds:
            #Load lons, lats, topography
            lon = ds.coords["lon"].to_numpy()[:self.params.img_size_x]
            lat = ds.coords["lat"].to_numpy()[:self.params.img_size_y]
            topo = ds[["z"]].to_array().to_numpy()[:, : self.params.img_size_y, : self.params.img_size_x]
      
            #Load GES fields
            if len(self.params.inp_pred_vars) != 0:
                inp_pred = (ds[self.params.inp_pred_vars].to_array()).to_numpy()[:, :self.params.img_size_y, :self.params.img_size_x]
                inp_pred = np.squeeze(inp_pred)

                # Create field mask: 1 where data is valid (non-zero), 0 where invalid (zero)
                field_mask = (inp_pred != 0).astype(inp_pred.dtype)

            #Load obs
            if len(self.params.inp_obs_vars) != 0:
                obs = (ds[self.params.inp_obs_vars].to_array()).to_numpy()[
                    :, -self.params.obs_time_window:, :self.params.img_size_y, :self.params.img_size_x]

                #Get most recent obs as target
                obs_tar = obs[:, -1]

                # Make a mask of the obs - used to replace values in the target (analysis) field later
                obs_tar_mask = (obs_tar != 0).astype(obs_tar.dtype)

                heldout_mask = np.zeros(obs.shape[-2:], dtype=obs.dtype)
                if self.params.hold_out_obs:
                    heldout_mask = self._heldout_mask(obs, hour_idx)
                    inp_obs = obs * (1 - heldout_mask)
                else:
                    inp_obs = obs
                inp_obs = inp_obs.reshape((-1, self.params.img_size_y, self.params.img_size_x))

            #Load satellite fields
            if len(self.params.inp_sat_vars) != 0:
                sat = (ds[self.params.inp_sat_vars].to_array()).to_numpy()[
                    :, -self.params.obs_time_window:, :self.params.img_size_y, :self.params.img_size_x]
                inp_sat = sat.reshape((-1, self.params.img_size_y, self.params.img_size_x))

            #Load target (analysis) fields
            field_tar = (ds[self.params.field_tar_vars].to_array()).to_numpy()[:, : self.params.img_size_y, : self.params.img_size_x]

            # Return the raw components to the trainer for final assembly, either on GPU (default) or CPU
            return (inp_pred, inp_obs, inp_sat, topo, field_tar, obs_tar, field_mask, obs_tar_mask, heldout_mask, lat, lon)
            
        
                