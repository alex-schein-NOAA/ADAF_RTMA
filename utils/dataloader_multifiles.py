import os
import glob
import torch
import logging
import numpy as np
import pandas as pd
import xarray as xr
import hdf5plugin

from torch.utils.data import DataLoader, Dataset, DistributedSampler
# from torch.utils.data.distributed import DistributedSampler

import time

####################
def get_data_loader(params, files_location, distributed, train):
    dataset = GetDataset(params, files_location, train)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler=None

    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        prefetch_factor=params.prefetch_factor if params.num_data_workers > 0 else 1,
        shuffle=False,  # (sampler is none),
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

    def __getitem__(self, hour_idx):
        with self.open_file(hour_idx) as ds:
            #Load lons, lats, topography
            lon = ds.coords["lon"].to_numpy()[:self.params.img_size_x]
            lat = ds.coords["lat"].to_numpy()[:self.params.img_size_y]
            topo = ds[["z"]].to_array().to_numpy()[:, : self.params.img_size_y, : self.params.img_size_x]
      
            #Load HRRR fields
            if len(self.params.inp_hrrr_vars) != 0:
                inp_hrrr = (ds[self.params.inp_hrrr_vars].to_array()).to_numpy()[:, :self.params.img_size_y, :self.params.img_size_x]
                inp_hrrr = np.squeeze(inp_hrrr)

                # Create field mask: 1 where data is valid (non-zero), 0 where invalid (zero)
                field_mask = (inp_hrrr != 0).astype(inp_hrrr.dtype)

            #Load obs
            if len(self.params.inp_obs_vars) != 0:
                obs = (ds[self.params.inp_obs_vars].to_array()).to_numpy()[
                    :, -self.params.obs_time_window:, :self.params.img_size_y, :self.params.img_size_x]

                #Get most recent obs as target
                obs_tar = obs[:, -1]

                # Make a mask of the obs - used to replace values in the target (RTMA) field later
                obs_tar_mask = (obs_tar != 0).astype(obs_tar.dtype)

                #Hold out obs; note the held-out obs are still used to replace target field values
                if self.params.hold_out_obs:
                    if self.params.obs_mask_seed != 0: #use a set seed; if 0, then use a random seed
                        np.random.seed(self.params.obs_mask_seed)

                    obs_flattened = obs[0,0].flatten()
                    obs_indices = np.where(obs_flattened != 0)[0]

                    hold_out_num = int(len(obs_indices) * self.params.hold_out_obs_ratio)

                    np.random.shuffle(obs_indices)
                    hold_out_obs_indices = obs_indices[:hold_out_num] #pluck out every Nth point

                    #Make the mask without the held out obs
                    obs_mask = np.zeros(np.shape(obs_flattened), dtype=obs.dtype) #ensure no silent upcasting of inp_obs
                    obs_mask[hold_out_obs_indices] = 1
                    obs_mask = obs_mask.reshape(obs[0,0].shape[0], obs[0,0].shape[1])

                    #Final input obs = obs minus held out obs
                    inp_obs = obs*(1-obs_mask)
                    inp_obs = inp_obs.reshape((-1, self.params.img_size_y, self.params.img_size_x)) 
                else:
                    inp_obs = obs.copy()
                    inp_obs = inp_obs.reshape((-1, self.params.img_size_y, self.params.img_size_x)) 
                    obs_mask = np.zeros(np.shape(inp_obs), dtype=inp_obs.dtype) #Maybe not needed here, but ensure no silent upcasting of inp_obs

            #####
            ## Satellite stuff here, when done
            #####

            #Load target (RTMA) fields
            field_tar = (ds[self.params.field_tar_vars].to_array()).to_numpy()[:, : self.params.img_size_y, : self.params.img_size_x]

            # Return the raw components to the trainer for final assembly, either on GPU (default) or CPU (same as commented out section below)
            return (inp_hrrr, inp_obs, topo, field_tar, obs_tar,
                    field_mask, obs_tar_mask, lat, lon)
            
        #     #Replace target field with obs @ observed locations (all obs locations, including those held out previously)
        #     field_obs_tar = field_tar.copy()
        #     field_obs_tar[obs_tar_mask == 1] = 0
        #     field_obs_tar += obs_tar

        #     if self.params.learn_residual:
        #         field_tar = field_tar - inp_hrrr
        #         obs_tar = obs_tar - inp_hrrr
        #         field_obs_tar = field_obs_tar - inp_hrrr

            
        #     #Make final input tensor
        #     inp = np.concatenate((inp_hrrr, inp_obs, topo), axis=0)
        #     #Satellite version here when that's done

        # return (inp,
        #         field_tar,
        #         obs_tar,
        #         field_obs_tar,
        #         inp_hrrr,
        #         lat,
        #         lon,
        #         field_mask,
        #         obs_tar_mask)
                