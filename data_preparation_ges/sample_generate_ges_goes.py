import argparse
import os
import sys
import datetime as dt

# Data Science Core
import numpy as np
import pandas as pd
import xarray as xr
import hdf5plugin

# Custom Packages
from funcs_data_preparation_ges_goes import *

###############################

parser = argparse.ArgumentParser()
parser.add_argument("--starting_analysis_time", type=str, required=True) # Must be formatted as "YYYY-MM-DD_HH"
parser.add_argument("--ending_analysis_time", type=str, required=True)   # Must be formatted as "YYYY-MM-DD_HH"
parser.add_argument("--obs_source", type=str, choices=["metar", "combined"], default="metar")
parser.add_argument("--save_directory", type=str, default=None)
parser.add_argument("--goes_cache_dir", type=str, default="./goes_cache")
parser.add_argument("--goes_channels", nargs="+", type=int, default=[2, 7, 10, 14])
parser.add_argument("--goes_product", type=str, default="ABI-L1b-RadF-Reproc")

args = parser.parse_args()
starting_analysis_time = dt.datetime.strptime(args.starting_analysis_time, "%Y-%m-%d_%H")
ending_analysis_time = dt.datetime.strptime(args.ending_analysis_time, "%Y-%m-%d_%H")
save_directory = args.save_directory

if save_directory is None:
    sys.exit("ERROR: --save_directory must be specified!")
   
analysis_times_list = pd.date_range(start=starting_analysis_time, end=ending_analysis_time, freq='h').to_pydatetime().tolist()

data_prep_dir = f"/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/data_preparation_ges"

### Vars shared between RTMA and sta
# IODA var name -> ADAF station channel
ADAF_CHANNELS = {
    "airTemperature": "t",
    "specificHumidity": "q",
    "windEastward": "u10",
    "windNorthward": "v10"
}

stats_filepath = f"{data_prep_dir}/stats_ges.csv"
stats = pd.read_csv(stats_filepath, index_col=0)

already_exists_count = 0
missing_count = 0
written_count = 0

anl_variables = [f"rtma_anl_{x}" for x in ADAF_CHANNELS.values()] 
ges_variables = [f"rtma_ges_{x}" for x in ADAF_CHANNELS.values()] 

topo_filepath = f"{data_prep_dir}/RTMA_TOPO_2p5km.nc"

### Station & Satellite static configuration
PAST_OBS_ONLY = True
OBS_TIME_WINDOW = 3 # hours
THRESHOLD_MINS = 30
GOOD_QM = {0, 1, 2, 3}  # prepbufr quality markers considered usable
QM_FILLVALUE = 2147483647
FLOAT_FILLVALUE = 1e36
ioda_directory = f"/scratch4/BMC/wrfruc/Micah.Craine/adaf_3yr/ioda/com/rtma/v2.1.4"

blosc2_encoding = dict(hdf5plugin.Blosc2(cname='zstd', clevel=3, filters=hdf5plugin.Blosc2.BITSHUFFLE))

# Obs source files to load per cycle
if args.obs_source == "combined":
    obs_source_files = [("ioda_msonet.nc", "mesonet"), ("ioda_adpsfc.nc", "metar")]
elif args.obs_source == "metar":
    obs_source_files = [("ioda_adpsfc.nc", "metar")]
else:
    raise ValueError(f"obs_source must be 'metar' or 'combined' (got {args.obs_source})")

### State variables assigned in main loop
topo_normed = None
lats_1d = None # Serves as flag variable for domain coordinate setup

# Persistent dictionary cache across outer loop cycles to preserve pre-computed spatial lookups
regrid_cache = {}

for t, analysis_time in enumerate(analysis_times_list):
    output_filename = f"{analysis_time.strftime('%Y-%m-%d_%H')}.nc"
    if os.path.exists(f"{save_directory}/{output_filename}"):
        print(f"{output_filename} already exists in {save_directory}")
        already_exists_count += 1
    else:
        ### Check if all station obs files exist - if not, skip before computation
        skip_analysis = False

        for hour_offset in range(OBS_TIME_WINDOW):
            target_time = analysis_time - dt.timedelta(hours=hour_offset)
            date_str = target_time.strftime("%Y%m%d")
            hour_str = target_time.strftime("%H")
        
            for src_filename, src_label in obs_source_files:
                file_path = f"{ioda_directory}/rtma.{date_str}/{hour_str}/ioda_bufr/det/{src_filename}"

                if not os.path.exists(file_path):
                    print(f"!!! Missing file for target time {date_str}_{hour_str}. Skipping analysis_time: {analysis_time}")
                    skip_analysis = True
                    break
            if skip_analysis:
                break

        if skip_analysis:
            missing_count += 1
            continue

        ### Maintain GOES satellite cache
        cached_goes_files = download_multiple_goes16_files(
            target_time=analysis_time,
            channels=args.goes_channels,
            save_dir=args.goes_cache_dir,
            product=args.goes_product
        )

        if not cached_goes_files:
            print(f"!!! Missing GOES files for target time {analysis_time}. Skipping analysis_time.")
            missing_count += 1
            continue
        
        # Dynamic RTMA directory
        rtma_directory = f"/scratch5/BMC/ai-datadepot/data/models/rtma/2p5km/grib2/{analysis_time.strftime('%Y%m%d')}" 
    
        ges_data = []
        anl_data = []
        
        for i, adaf_var in enumerate(ADAF_CHANNELS.values()):
            ges_filename = f"rtma2p5.t{str(analysis_time.hour).zfill(2)}z.2dvarges_ndfd.grb2_wexp"
            anl_filename = f"rtma2p5.t{str(analysis_time.hour).zfill(2)}z.2dvaranl_ndfd.grb2_wexp"

            xr_ges, xr_anl = fetch_rtma_ges_and_anl(
                adaf_var, 
                rtma_ges_filepath=f"{rtma_directory}/{ges_filename}", 
                rtma_anl_filepath=f"{rtma_directory}/{anl_filename}"
            )
        
            if topo_normed is None:
                topo = xr.open_dataset(topo_filepath)
                topo = topo["orog"]
                topo_normed = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(topo, 'z', stats_filepath)
        
            if lats_1d is None:
                lats_1d = xr_ges['latitude'].data[xr_ges.data != 0]
                lons_1d = xr_ges['longitude'].data[xr_ges.data != 0]
                
                # For reassigning the stations - need full domain
                lats_2d = xr_ges['latitude'].data
                lons_2d = xr_ges['longitude'].data
                
                # Set lat/lon bounds for station use with TOL padding
                TOL = 0.05
                LAT_BOUNDS = (np.min(lats_1d).item() - TOL, np.max(lats_1d).item() + TOL)
                LON_BOUNDS = (np.min(lons_1d).item() - TOL, np.max(lons_1d).item() + TOL)

                df_lats_lons = pd.DataFrame({'lat': lats_1d, 'lon': lons_1d})
            
            if adaf_var == 't':
                xr_ges = xr_ges - 273.15 # convert K to C
                xr_anl = xr_anl - 273.15 
                xr_ges = xr_ges.where(~np.isclose(xr_ges, -273.15), 0)
                xr_anl = xr_anl.where(~np.isclose(xr_anl, -273.15), 0)
        
            xr_ges_normed = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(xr_ges, ges_variables[i], stats_filepath)
            xr_anl_normed = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(xr_anl, anl_variables[i], stats_filepath)
            
            ges_data.append(xr_ges_normed.data)
            anl_data.append(xr_anl_normed.data)
    
        ds_ges_anl = xr.Dataset(
            {
                **{var: (("y", "x"), data) for var, data in zip(anl_variables, anl_data)},
                **{var: (("y", "x"), data) for var, data in zip(ges_variables, ges_data)},
                "z": (("y", "x"), topo_normed.data),
            },
            coords={
                "valid_time": analysis_time,
                "lat": (('y', 'x'), lats_2d),
                "lon": (('y', 'x'), lons_2d)
            },
        )

        ### Regrid GOES satellite data using persistent regrid_cache
        ds_goes = process_and_regrid_goes_cache(
            cached_files=cached_goes_files,
            lats_2d=lats_2d,
            lons_2d=lons_2d,
            analysis_time=analysis_time,
            obs_time_window=OBS_TIME_WINDOW,
            regrid_cache=regrid_cache
        )

        if ds_goes is None:
            print(f"!!! Could not process GOES data for analysis_time: {analysis_time}. Skipping.")
            missing_count += 1
            continue

        ds_goes = normalize_goes_dataset(ds_goes, stats_filepath)
        
        ### Process Station obs
        df_list = []
        
        for hour_offset in range(OBS_TIME_WINDOW):
            target_time = analysis_time - dt.timedelta(hours=hour_offset)
            date_str = target_time.strftime("%Y%m%d")
            hour_str = target_time.strftime("%H")
        
            for src_filename, src_label in obs_source_files:
                file_path = f"{ioda_directory}/rtma.{date_str}/{hour_str}/ioda_bufr/det/{src_filename}"

                hourly_df = load_stations_into_dataframe_and_clean(
                    path=file_path,
                    ADAF_CHANNELS=ADAF_CHANNELS,
                    GOOD_QM=GOOD_QM,
                    QM_FILLVALUE=QM_FILLVALUE,
                    FLOAT_FILLVALUE=FLOAT_FILLVALUE,
                    LAT_BOUNDS=LAT_BOUNDS,
                    LON_BOUNDS=LON_BOUNDS
                )
                if src_label is not None:
                    hourly_df["source"] = src_label

                df_list.append(hourly_df)
        
        df = pd.concat(df_list, ignore_index=True)

        df = assign_closest_with_threshold(
            df, df_lats_lons, 
            lat_min=LAT_BOUNDS[0], lat_max=LAT_BOUNDS[1], 
            lon_min=LON_BOUNDS[0], lon_max=LON_BOUNDS[1], 
            max_dist_km=10
        )
        df = keep_closest_to_hour_per_location_with_time_threshold(df, threshold_mins=THRESHOLD_MINS, past_obs_only=PAST_OBS_ONLY)
        if PAST_OBS_ONLY:
            df['OBS_TIMESTAMP'] = df['OBS_TIMESTAMP'].dt.ceil('h')
        else:
            df['OBS_TIMESTAMP'] = df['OBS_TIMESTAMP'].dt.round('h')
        df = filter_obs_by_temporal_completeness(df, obs_time_window=OBS_TIME_WINDOW, analysis_time=analysis_time)
        if df['sta_t'].iloc[0] > 100: # convert K to C before rejection
            df['sta_t'] = df['sta_t'] - 273.15
        df = reject_out_of_bounds_obs(df)
        df = min_max_norm_ignore_extreme_fill_nan_sta_df(df, stats_path=stats_filepath)

        ds_sta_obs = assemble_station_dataset(df, lats_2d=lats_2d, lons_2d=lons_2d, analysis_time=analysis_time)

        ### Merge RTMA, Station, and GOES satellite datasets
        ds = xr.merge([ds_ges_anl, ds_sta_obs, ds_goes], compat="no_conflicts")

        encoding = {}
        for var in ds.data_vars:
            encoding[var] = blosc2_encoding.copy()
            if ds[var].dtype == np.float64:
                encoding[var]['dtype'] = 'float32'

        ds.to_netcdf(f"{save_directory}/{output_filename}", engine="h5netcdf", encoding=encoding)
        print(f"{output_filename} saved to {save_directory}")
        written_count += 1

print(
    "Run summary: "
    f"requested={len(analysis_times_list)}, "
    f"written={written_count}, "
    f"already_exists={already_exists_count}, "
    f"missing_count={missing_count}"
)
