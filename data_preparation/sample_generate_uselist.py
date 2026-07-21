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
from funcs_data_preparation_uselist import *

###############################

parser = argparse.ArgumentParser()
parser.add_argument("--starting_analysis_time", type=str) #Must be formatted as "YYYY-MM-DD_HH"
parser.add_argument("--ending_analysis_time", type=str) #Must be formatted as "YYYY-MM-DD_HH"
# parser.add_argument("--obs_source", type=str, choices=["mesonet", "combined"], default="mesonet") #'combined' adds METAR/synoptic land (ioda_adpsfc.nc)
parser.add_argument("--save_directory", type=str, default=None)


args = parser.parse_args()
starting_analysis_time = dt.datetime.strptime(args.starting_analysis_time, "%Y-%m-%d_%H")
ending_analysis_time = dt.datetime.strptime(args.ending_analysis_time, "%Y-%m-%d_%H")
save_directory = args.save_directory

if save_directory is None:
   sys.exit("ERROR: --save_directory must be specifed!")
   
analysis_times_list = pd.date_range(start=starting_analysis_time, end=ending_analysis_time, freq='h').to_pydatetime().tolist()

### Vars shared between HRRR/RTMA and sta
# IODA var name -> ADAF station channel
ADAF_CHANNELS = {"airTemperature":  "t",
                 "specificHumidity": "q",
                 "windEastward": "u10",
                 "windNorthward": "v10"}

stats_filepath=f"/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/data_preparation/stats.csv"
stats = pd.read_csv(stats_filepath, index_col=0)

already_exists_count = 0
missing_count = 0
written_count = 0

### HRRR/RTMA specific vars, static
hrrr_forecast_leadtime = 1

rtma_variables = [f"rtma_{x}" for x in ADAF_CHANNELS.values()] 
hrrr_variables = [f"hrrr_{x}" for x in ADAF_CHANNELS.values()] 

hrrr_regridder_filepath = f"/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/data_preparation/regridder_hrrr.nc"
topo_filepath = f"/scratch3/BMC/wrfruc/aschein/Train_Test_Files/terrain_CONUS_URMA_2p5km.grib2"

### Station specific vars, static
PAST_OBS_ONLY = True
OBS_TIME_WINDOW = 3 #hours
THRESHOLD_MINS = 30
GOOD_QM = {0, 1, 2, 3}  # prepbufr quality markers considered usable
QM_FILLVALUE=2147483647
FLOAT_FILLVALUE = 1e36
PROVIDER_USELIST = ["FL-Meso", "SFWMD", "CAIC", "OK-Meso", "WT-Meso", "WYDOT", "NOS-PORT", "NOS-NWLO", "GoMOOS", "METAR"]
ioda_directory = f"/scratch4/BMC/wrfruc/Micah.Craine/adaf_3yr/ioda/com/rtma/v2.1.4"
uselist_directory = f"/home/role.amb-verif/ruc_madis_surface/rap_ops_mesonet_uselists/"

blosc2_encoding = dict(hdf5plugin.Blosc2(cname='zstd', clevel=3, filters=hdf5plugin.Blosc2.BITSHUFFLE))

# Obs source files to load per cycle. Default 'mesonet' is the original, untagged path; 'combined' also loads
# METAR/synoptic land and tags each obs with its source (carried as the obs_source label array in assemble_station_dataset).
## (2026-07-21) Only using combined from now on

# if args.obs_source == "combined":
obs_source_files = [("ioda_msonet.nc", "mesonet"), ("ioda_adpsfc.nc", "metar")]
# else:
    # obs_source_files = [("ioda_msonet.nc", None)]

### Vars assigned in the main loops
dict_bounds = None
topo_normed = None
lats_1d = None #This serves as a flag variable for all the other lat/lon related vars

for t, analysis_time in enumerate(analysis_times_list):
    output_filename = f"{analysis_time.strftime('%Y-%m-%d_%H')}.nc"
    if os.path.exists(f"{save_directory}/{output_filename}"):
        print(f"{output_filename} already exists in {save_directory}")
        already_exists_count+=1
    else:
        ### Check if all mesonet files exist - if not, go to the next time before any computation is done (i.e. wasted)
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
                    break #Break out of the obs_source_files loop
            if skip_analysis: # Break out of the hour_offset loop
                break

        # If the flag was tripped, jump to the next outer analysis_time
        if skip_analysis:
            missing_count+=1
            continue
        
        ### Generate all RTMA, HRRR fields
        hrrr_init_time = analysis_time - dt.timedelta(hours=hrrr_forecast_leadtime) #need to call the proper f01 HRRR file
        
        # Dynamic file directories 
        hrrr_directory = f"/scratch5/BMC/ai-datadepot/data/models/hrrr/conus/grib2/{hrrr_init_time.strftime('%Y%m%d')}"
        rtma_directory = f"/scratch5/BMC/ai-datadepot/data/models/rtma/2p5km/grib2/{analysis_time.strftime('%Y%m%d')}" #2026-05-29 updated to the main depot
    
        hrrr_data = []
        rtma_data = []
        
        for i, adaf_var in enumerate(ADAF_CHANNELS.values()):
        
            hrrr_filename = f"hrrr.t{str(hrrr_init_time.hour).zfill(2)}z.wrfnatf01.grib2"
            rtma_filename = f"rtma2p5.t{str(analysis_time.hour).zfill(2)}z.2dvaranl_ndfd.grb2_wexp"

            if dict_bounds is None:
                xr_hrrr_regridded_cropped, xr_rtma_cropped, dict_bounds = fetch_regrid_crop_hrrr_rtma(adaf_var, 
                                                                                                      hrrr_filepath=f"{hrrr_directory}/{hrrr_filename}", 
                                                                                                      rtma_filepath=f"{rtma_directory}/{rtma_filename}", 
                                                                                                      hrrr_regridder_filepath=hrrr_regridder_filepath)
            else:
                xr_hrrr_regridded_cropped, xr_rtma_cropped, _ = fetch_regrid_crop_hrrr_rtma(adaf_var, 
                                                                                            hrrr_filepath=f"{hrrr_directory}/{hrrr_filename}", 
                                                                                            rtma_filepath=f"{rtma_directory}/{rtma_filename}", 
                                                                                            hrrr_regridder_filepath=hrrr_regridder_filepath, 
                                                                                            dict_bounds=dict_bounds)
            
            if topo_normed is None:
                topo = xr.open_dataset(topo_filepath, engine="cfgrib", backend_kwargs={"indexpath": ""})
                topo = topo["orog"]
                topo = topo.isel({'y': slice(dict_bounds['row_start'], dict_bounds['row_end']), 
                                  'x': slice(dict_bounds['col_start'], dict_bounds['col_end'])})
                topo = topo.where(xr_hrrr_regridded_cropped != 0, 0)
                topo_normed = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(topo, 'z', stats_filepath)
        
            if lats_1d is None:
                lats_1d = xr_rtma_cropped['latitude'].data[xr_rtma_cropped.data != 0]
                lons_1d = xr_rtma_cropped['longitude'].data[xr_rtma_cropped.data != 0]
                
                # For reassigning the stations - need the full domain
                lats_2d = xr_rtma_cropped['latitude'].data
                lons_2d = xr_rtma_cropped['longitude'].data
                
                # Set lat/lon bounds for station use. Put TOL padding on the edges
                TOL = 0.05
                LAT_BOUNDS=(np.min(lats_1d).item()-TOL,np.max(lats_1d).item()+TOL)
                LON_BOUNDS=(np.min(lons_1d).item()-TOL, np.max(lons_1d).item()+TOL)

                df_lats_lons = pd.DataFrame({'lat': lats_1d, 'lon': lons_1d})
            
            
            xr_hrrr_regridded_cropped_fixed = fix_dataset_scaling_shifting(xr_hrrr_regridded_cropped, adaf_var)
            xr_rtma_cropped_fixed = fix_dataset_scaling_shifting(xr_rtma_cropped, adaf_var)
        
            xr_hrrr_regridded_cropped_fixed_normed = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(xr_hrrr_regridded_cropped_fixed,
                                                                                                         hrrr_variables[i],
                                                                                                         stats_filepath)

            xr_rtma_cropped_fixed_normed = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(xr_rtma_cropped_fixed,
                                                                                               rtma_variables[i],
                                                                                               stats_filepath)
            
            hrrr_data.append(xr_hrrr_regridded_cropped_fixed_normed.data)
            rtma_data.append(xr_rtma_cropped_fixed_normed.data)
    
        ds_hrrr_rtma = xr.Dataset(
                {
                # RTMA, normalized
                f"{rtma_variables[0]}": (("y", "x"), rtma_data[0]),
                f"{rtma_variables[1]}": (("y", "x"), rtma_data[1]),
                f"{rtma_variables[2]}": (("y", "x"), rtma_data[2]),
                f"{rtma_variables[3]}": (("y", "x"), rtma_data[3]),
                
                # HRRR, normalized 
                f"{hrrr_variables[0]}": (("y", "x"), hrrr_data[0]),
                f"{hrrr_variables[1]}": (("y", "x"), hrrr_data[1]),
                f"{hrrr_variables[2]}": (("y", "x"), hrrr_data[2]),
                f"{hrrr_variables[3]}": (("y", "x"), hrrr_data[3]),
    
                #Topography, normalized
                f"z" : (("y", "x"), topo_normed.data),
                },
            coords={
                "valid_time": analysis_time,
                "lat": (('y','x'), lats_2d),
                "lon": (('y','x'), lons_2d)
                    },
            )
    
        ### Station obs
        ## Dict of OBS_TIME_WINDOW uselists is made to ensure times straddling different days are handled correctly
        
        # Initialize the target dictionary and a file cache to avoid redundant disk I/O
        uselist_dict = {}
        _file_cache = {}
        df_list = []
        
        for hour_offset in range(OBS_TIME_WINDOW):
            target_time = analysis_time - dt.timedelta(hours=hour_offset)
            date_str = target_time.strftime("%Y%m%d")
            hour_str = target_time.strftime("%H")
            
            # Generate the string for the uselist filename (YYYY-MM-DD)
            uselist_date_str = target_time.strftime("%Y-%m-%d")
            uselist_file_path = f"{uselist_directory}/{uselist_date_str}_meso_uselist.txt"  # Add directory prefix if needed
            
            # Read and parse the uselist file if it hasn't been cached for this day yet
            if uselist_date_str not in _file_cache:
                try:
                    uselist_df = load_mesonet_daily_uselist(uselist_file_path, PROVIDER_USELIST)
                    _file_cache[uselist_date_str] = uselist_df
                except FileNotFoundError:
                    print(f"Warning: Uselist file missing for {uselist_date_str}. Creating empty fallback.")
                    # Fallback to an empty DataFrame with the correct 'name' column to avoid downstream breakdown
                    _file_cache[uselist_date_str] = pd.DataFrame(columns=["name"])
        
            # Map the correct day's dataframe to this specific hourly target_time key
            uselist_dict[target_time] = _file_cache[uselist_date_str]
        
            # Continue with loading the mesonet data files
            for src_filename, src_label in obs_source_files:
                file_path = f"{ioda_directory}/rtma.{date_str}/{hour_str}/ioda_bufr/det/{src_filename}"
        
                hourly_df = load_stations_into_dataframe_and_clean_with_id(path=file_path,
                                                                           ADAF_CHANNELS=ADAF_CHANNELS,
                                                                           GOOD_QM=GOOD_QM,
                                                                           QM_FILLVALUE=QM_FILLVALUE,
                                                                           FLOAT_FILLVALUE=FLOAT_FILLVALUE)
                df_list.append(hourly_df)
        
        # Concatenate all loaded data together and reject future obs (if PAST_OBS_ONLY==True)
        df = pd.concat(df_list, ignore_index=True)
        if PAST_OBS_ONLY:
            df = df[df["OBS_TIMESTAMP"] <= analysis_time]
        
        # Provider filter is necessary even though we later use station uselist, because different networks may have identically-named stations (ex MEDI for OK-Meso and NYSM)
        df = df[df["dataProviderOrigin"].isin(PROVIDER_USELIST)].reset_index(drop=True)

        # Chunk the dataset per hour, fetch the corresponding uselist, filter based on that, and put the complete now-filtered dataframe back together
        filtered_chunks = []
        for hourly_timestamp, df_uselist in uselist_dict.items():
            # Extract the rows matching this specific hour
            # OBS_TIMESTAMP values contain random minutes/seconds, so we need to floor the times to get the appropriate uselist
            hour_mask = df["OBS_TIMESTAMP"].dt.floor("h") == hourly_timestamp
            df_hour = df[hour_mask]
         
            if df_hour.empty:
                continue
        
            valid_stations = df_uselist["name"].unique()
            is_metar = df_hour["dataProviderOrigin"] == "METAR"
            is_approved_mesonet = df_hour["stationIdentification"].isin(valid_stations)
            
            df_hour_filtered = df_hour[is_metar | is_approved_mesonet]

            # Apply quality flagging if flag columns exist in the uselist
            # If a station exists in the uselist, it has at least 1 acceptable variable, so no need to check for that afterwards
            if all(col in df_uselist.columns for col in ["good_T", "good_Td", "good_W"]):
                # Create indexed lookup table for fast station mapping
                uselist_map = df_uselist.drop_duplicates(subset=["name"]).set_index("name")
                
                # Map quality flags to filtered stations
                t_flag = df_hour_filtered["stationIdentification"].map(uselist_map["good_T"])
                td_flag = df_hour_filtered["stationIdentification"].map(uselist_map["good_Td"])
                w_flag = df_hour_filtered["stationIdentification"].map(uselist_map["good_W"])
        
                # Mask station values with NaN where flags are explicitly 0
                df_hour_filtered.loc[t_flag == 0, "sta_t"] = np.nan
                df_hour_filtered.loc[td_flag == 0, "sta_q"] = np.nan
                df_hour_filtered.loc[w_flag == 0, ["sta_u10", "sta_v10"]] = np.nan
            
            filtered_chunks.append(df_hour_filtered)
        
        # Put the complete, culled mesonet dataframe back together
        df = pd.concat(filtered_chunks, ignore_index=True)

        df = assign_closest_with_threshold(df, df_lats_lons, 
                                           lat_min=LAT_BOUNDS[0], lat_max=LAT_BOUNDS[1], 
                                           lon_min=LON_BOUNDS[0], lon_max=LON_BOUNDS[1], 
                                           max_dist_km=10)

        df = keep_closest_to_hour_per_location_with_time_threshold(df, threshold_mins=THRESHOLD_MINS, past_obs_only=PAST_OBS_ONLY)

        if PAST_OBS_ONLY:
            df['OBS_TIMESTAMP'] = df['OBS_TIMESTAMP'].dt.ceil('h')
        else:
            df['OBS_TIMESTAMP'] = df['OBS_TIMESTAMP'].dt.round('h')

        df = filter_obs_by_temporal_completeness(df, OBS_TIME_WINDOW)

        if df['sta_t'].mean() > 150: #convert from K to C. Needs to be done before reject_out_of_bounds_obs
            df['sta_t'] = df['sta_t'] - 273.15

        df = reject_out_of_bounds_obs(df)

        df = min_max_norm_ignore_extreme_keep_nan_sta_df(df, stats_path=stats_filepath)
        
        # Drop stations, but KEEP the METAR/non-METAR column for use in making obs_mask
        df = df.drop(columns=['stationIdentification']).reset_index(drop=True)

        ds_sta_obs = assemble_station_dataset(df, lats_2d=lats_2d, lons_2d=lons_2d, analysis_time=analysis_time)

        ### Merge and compress and save
        ds = xr.merge([ds_hrrr_rtma, ds_sta_obs], compat="no_conflicts")

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
