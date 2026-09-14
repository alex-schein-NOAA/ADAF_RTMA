# Standard Library
import os
import datetime as dt
from datetime import datetime, timedelta

# Data Science Core
import numpy as np
import pandas as pd
import xarray as xr

# AWS & Satellite Processing
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pyproj import Proj
from pyresample import geometry, kd_tree

# Geospatial & Analysis
import xesmf
import h5py
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import KDTree

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib.markers import MarkerStyle


##############################################################################################################

###############################################################
#################### RTMA ges/anl FUNCTIONS ####################
###############################################################

def fetch_rtma_ges_and_anl(adaf_var_str, rtma_ges_filepath, rtma_anl_filepath):
    """ 
    Fetches one variable (as determined by adaf_var_str) for the RTMA first guess field (ges) and final analysis (anl).

    Inputs:
        - adaf_var_str --> valid options = 't', 'q', 'u10', 'v10'
        - rtma_ges_filepath --> path to first-guess .grib2 file to be opened. Assumed to be a complete file, i.e. not already subset to a variable/variables
        - rtma_anl_filepath --> path to RTMA final analysis .grib2 file to be opened. Assumed to be a complete file, i.e. not already subset to a variable/variables
    """

    dict_level_selection = {"t":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                            "q":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}},
                            "u10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}},
                            "v10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}}}
        
    dict_adaf_translate = {'t':'t2m', 
                           'q':'sh2', 
                           'u10':'u10', 
                           'v10':'v10'}
    
    rtma_backend_kwargs = dict(dict_level_selection[adaf_var_str])
    rtma_backend_kwargs["indexpath"] = ""
    
    xr_rtma_ges = xr.open_dataset(rtma_ges_filepath, engine='cfgrib', backend_kwargs=rtma_backend_kwargs)
    xr_rtma_anl = xr.open_dataset(rtma_anl_filepath, engine='cfgrib', backend_kwargs=rtma_backend_kwargs)
    
    # Rename the field to the ADAF name (has to be done on the full dataset before it gets subset to a DataArray)
    xr_rtma_ges = xr_rtma_ges.rename_vars({dict_adaf_translate[adaf_var_str] : adaf_var_str})
    xr_rtma_anl = xr_rtma_anl.rename_vars({dict_adaf_translate[adaf_var_str] : adaf_var_str})

    # Subset down to only the variable we care about
    xr_rtma_ges = xr_rtma_ges[adaf_var_str]
    xr_rtma_anl = xr_rtma_anl[adaf_var_str]

    return xr_rtma_ges, xr_rtma_anl

###################

# def min_max_norm_ignore_extreme_fill_nan_onevar_onetime(xr_data, adaf_var_str, stats_filepath):
#     """
#     Modified version of the min_max_norm_ignore_extreme_fill_nan function. 
#     This one does only a single time (i.e. no time dimension) and a single variable, as the original code doesn't seem to work with all variables at once.

#     Inputs:
#         - xr_data --> RTMA ges or anl data that has been run through fetch_rtma_ges_and_anl and fix_dataset_scaling_shifting. Should be 2D
#         - adaf_var_str --> string to select the right variable's stats. MUST BE MODIFIED BEFOREHAND, e.g. "rtma_t" instead of just "t"
#         - stats_filepath --> str of filepath to the stats.csv file containing variable min/max values
#     """

#     stats = pd.read_csv(stats_filepath, index_col=0)
#     field_tar_stats = stats.loc[stats['variable'] == adaf_var_str]

#     #Loop version of the ndim code
#     vmin = field_tar_stats["min"]
#     vmax = field_tar_stats["max"]
#     vmin = np.array(vmin)
#     vmax = np.array(vmax)
    
#     for j in range(xr_data.ndim-1):
#         vmin = vmin[:, np.newaxis]
#         vmax = vmax[:, np.newaxis]
    
#     for j in range(xr_data.ndim): #no temporal dimension yet - if that gets included as dim 0, then this should only apply to dims 1,2,..,n
#         vmin = np.repeat(vmin, xr_data.shape[j], axis=j)
#         vmax = np.repeat(vmax, xr_data.shape[j], axis=j) 

#     xr_data -= vmin
#     xr_data *= 2.0/(vmax-vmin)
#     xr_data -= 1.0

#     #Mask out-of-range values; xr.where is a little odd for replacing values
#     xr_data = xr_data.where(xr_data <= 1, other=1)
#     xr_data = xr_data.where(xr_data >= -1, other=-1)

#     xr_data = xr_data.fillna(0) 

#     return xr_data

def min_max_norm_ignore_extreme_fill_nan_onevar_onetime(xr_data, adaf_var_str, stats_filepath):
    """
    Normalizes 2D or 3D xarray DataArray using min/max values from stats_filepath.

    Inputs:
        - xr_data --> xarray.DataArray (2D or 3D)
        - adaf_var_str --> string matching 'variable' column in stats.csv (e.g. 'rtma_ges_t', 'goes_channel_02')
        - stats_filepath --> path to stats.csv file
    """
    stats = pd.read_csv(stats_filepath, index_col=0)
    field_tar_stats = stats.loc[stats['variable'] == adaf_var_str]

    if field_tar_stats.empty:
        raise KeyError(f"Variable '{adaf_var_str}' not found in stats file '{stats_filepath}'.")

    vmin = float(field_tar_stats["min"].iloc[0])
    vmax = float(field_tar_stats["max"].iloc[0])

    xr_data = ((xr_data - vmin) * 2.0 / (vmax - vmin)) - 1.0

    xr_data = xr_data.clip(min=-1.0, max=1.0)
    xr_data = xr_data.fillna(0.0)

    return xr_data

    
###############################################################
#################### STATION OBS FUNCTIONS ####################
###############################################################

def load_stations_into_dataframe_and_clean(path, 
                                          ADAF_CHANNELS=None, 
                                          GOOD_QM=None, 
                                          QM_FILLVALUE=None, 
                                          FLOAT_FILLVALUE=None,
                                          LAT_BOUNDS=(20.0, 55,0),
                                          LON_BOUNDS=(225.0, 300.0),
                                          drop_qm_cols=True):
    """
    Function to load METAR or Mesonet data (in IODA standard) from disk and turn it into a cleaned-up pandas Dataframe.
    """
    # Enforce that the user must explicitly pass all configuration values
    if (ADAF_CHANNELS is None or GOOD_QM is None or QM_FILLVALUE is None or FLOAT_FILLVALUE is None):
        raise ValueError("Missing required arguments. You must explicitly provide: ADAF_CHANNELS, GOOD_QM, QM_FILLVALUE, and FLOAT_FILLVALUE.")

    d = {}
    
    with h5py.File(path, "r") as f:
        d["lat"] = f["MetaData/latitude"][:]
        d["lon"] = f["MetaData/longitude"][:]
        
        # Convert timestamps to datetime objects immediately
        d["OBS_TIMESTAMP"] = pd.to_datetime(f["MetaData/dateTime"][:], unit='s', origin='unix', errors='coerce')
        
        for v in ADAF_CHANNELS:
            d[v] = f[f"ObsValue/{v}"][:].astype("float64")
            qmkey = f"QualityMarker/{v}"
            d[v + "_qm"] = f[qmkey][:] if qmkey in f else np.full(d[v].shape, QM_FILLVALUE, dtype="int32") # Fallback array if QM doesn't exist

    # Load into a DataFrame for vectorized row operations
    df = pd.DataFrame(d)
    
    # Geographic Bounding Box Filter (Applied early to optimize groupby performance)
    df = df[(df["lat"] >= LAT_BOUNDS[0]) & (df["lat"] <= LAT_BOUNDS[1])]
    df = df[(df["lon"] >= LON_BOUNDS[0]) & (df["lon"] <= LON_BOUNDS[1])]
    
    # Mask all fill values to NaN so the merge process can identify valid data
    for v in ADAF_CHANNELS:
        df[v] = df[v].mask(df[v] > FLOAT_FILLVALUE, np.nan)
        df[v + "_qm"] = df[v + "_qm"].replace(QM_FILLVALUE, np.nan)
        
    # Merge the split station rows
    # groupby().first() combines rows matching on lat/lon/time, choosing the first non-NaN element
    df_merged = df.groupby(["lat", "lon", "OBS_TIMESTAMP"], as_index=False, dropna=False).first()
    
    #Rename columns to the new schema and define qm_cols early
    rename_dict = {}
    for long_name, short_name in ADAF_CHANNELS.items():
        rename_dict[long_name] = f"sta_{short_name}"
        rename_dict[f"{long_name}_qm"] = f"sta_{short_name}_qm"
        
    df_merged = df_merged.rename(columns=rename_dict)
    
    # Filter out rows with bad Quality Markers
    # Keeps rows if the QM is in GOOD_QM; if the QM is NaN (meaning that variable wasn't recorded) it's dropped
    qm_cols = [x for x in rename_dict.values() if "_qm" in x]
    valid_qm_mask = df_merged[qm_cols].isin(GOOD_QM)
    df_clean = df_merged[valid_qm_mask.all(axis=1)]

    # Handle Quality Marker columns based on user preference
    if drop_qm_cols:
        df_clean = df_clean.drop(columns=qm_cols)
    else:
        df_clean[qm_cols] = df_clean[qm_cols].astype("Int32") #Convert quality marker columns back to ints+NaNs allowed ("Int32")
    
    # Clean up the index and return the final DataFrame
    return df_clean.reset_index(drop=True)

###################

def assign_closest_with_threshold(df_target, df_ref, lat_min, lat_max, lon_min, lon_max, 
                                  max_dist_km=None, lat_col='lat', lon_col='lon'):
    """
    Filters coordinates and assigns the closest reference point, 
    rejecting assignments that exceed max_dist_km.

    Inputs:
        - df_target --> dataframe from concatted Mesonet files, already cut down to the hours of interest
        - df_ref --> dataframe constructed from 1D RTMA lat/lons (only the nonzero data region!)
        - lat/lon min/max --> rough USA bounds (should be read from min/max of 1D lats/lons, +/- tolerance)
        - max_dist_km --> should be 10 
    """

    #Get datasets onto common lat/lon convention, if they differ
    def normalize(df): 
        temp_df = df.copy()
        temp_df[lon_col] = temp_df[lon_col] % 360
        return temp_df

    # Normalize and Filter Dataframes
    df_t_norm = normalize(df_target)
    df_r_norm = normalize(df_ref)

    t_mask = (df_t_norm[lat_col].between(lat_min, lat_max)) & \
             (df_t_norm[lon_col].between(lon_min, lon_max))
    r_mask = (df_r_norm[lat_col].between(lat_min, lat_max)) & \
             (df_r_norm[lon_col].between(lon_min, lon_max))

    df_t_filtered = df_t_norm[t_mask].copy()
    df_r_filtered = df_r_norm[r_mask].drop_duplicates(subset=[lat_col, lon_col])

    if df_t_filtered.empty or df_r_filtered.empty:
        return pd.DataFrame(columns=df_target.columns)

    # Build KDTree and Query
    ref_coords = df_r_filtered[[lat_col, lon_col]].values
    tree = KDTree(ref_coords)
    
    target_coords = df_t_filtered[[lat_col, lon_col]].values
    distances, indices = tree.query(target_coords)

    # Apply Distance Threshold (if provided)
    if max_dist_km is not None:
        # Convert km to approximate decimal degrees
        # 111 km per degree is fine for our purposes
        max_dist_degrees = max_dist_km / 111.0
        
        # Keep only rows where the closest point is within the threshold
        valid_mask = distances <= max_dist_degrees
        df_t_filtered = df_t_filtered[valid_mask]
        indices = indices[valid_mask]
        
        if df_t_filtered.empty:
            return df_t_filtered

    # Overwrite with closest reference coordinates
    closest_matches = ref_coords[indices]
    df_t_filtered[lat_col] = closest_matches[:, 0]
    df_t_filtered[lon_col] = closest_matches[:, 1]

    return df_t_filtered
    
###################

def keep_closest_to_hour_per_location_with_time_threshold(df_input, time_col='OBS_TIMESTAMP', lat_col='lat', lon_col='lon', threshold_mins=30, past_obs_only=True):
    """
    Filters a DataFrame to keep the point closest to each hour, 
    calculated independently for each unique geographic coordinate.

    If past_obs_only=True (default) then only observations from BEFORE each hour will be considered. If False, then observations after the top of the hour will also be considered.
    Returns a dataframe with OBS_TIMESTAMP not yet rounded (or ceilinged) to the nearest hour - that behavior is up to the calling function.
    
    Should be applied to a dataframe that has already undergone regridding (i.e. the output of assign_closest_with_threshold() )
    """
    df = df_input.copy() 
    
    ## Create the rounding target
    if past_obs_only:
        df['target_hour'] = df[time_col].dt.ceil('h') 
    else:
        df['target_hour'] = df[time_col].dt.round('h') 

    ## Calculate absolute difference from that hour
    df['time_diff'] = (df[time_col] - df['target_hour']).abs() #if using ceiling, then absolute difference doesn't matter, but it does if using .round so no harm in keeping it around

    ## Only keep rows where the difference is <= the specified minutes
    max_delta = pd.Timedelta(minutes=threshold_mins)
    df = df[df['time_diff'] <= max_delta]

    ## Sort by coordinates, the target hour, then source preference, then the time difference
    ## This puts the "closest" record at the top of each (Location + Hour) group.
    ## When a cell+hour has both sources, prefer METAR (priority 0) over mesonet (priority 1) before falling back to closest-in-time.
    df['_src_priority'] = (df['source'] != 'metar').astype(int) if 'source' in df.columns else 0
    df_filtered = df.sort_values(by=[lat_col, lon_col, 'target_hour', '_src_priority', 'time_diff']).drop_duplicates(subset=[lat_col, lon_col, 'target_hour'], keep='first')

    #Clean up helper columns
    return df_filtered.drop(columns=['target_hour', 'time_diff', '_src_priority']).sort_values(by=[time_col, lat_col])
    
###################

def filter_obs_by_temporal_completeness(df_input, obs_time_window, analysis_time=None, precision=6):
    """
    Checks if each (lat, lon) point has a row for every hour in the window.
    Points missing any hour are removed entirely.

    If observations spill into an extra hour (e.g., ceil/round pushes 30-59 min
    observations to analysis_time + 1), trim all rows beyond analysis_time before
    enforcing temporal completeness.
    """
    df = df_input.copy()

    if analysis_time is not None:
        unique_hours = df['OBS_TIMESTAMP'].nunique()
        if unique_hours > obs_time_window:
            cutoff_time = pd.Timestamp(analysis_time)
            df = df[df['OBS_TIMESTAMP'] <= cutoff_time]
    
    # Round coordinates to prevent precision-related grouping errors
    df[['lat', 'lon']] = df[['lat', 'lon']].round(precision)
    
    # Count unique timestamps for each coordinate pair
    # transform('nunique') assigns the total count of unique hours to every row in that group
    counts = df.groupby(['lat', 'lon'])['OBS_TIMESTAMP'].transform('nunique')
    
    # Keep only the points where the count matches the required window size
    df_filtered = df[counts == obs_time_window]
    
    return df_filtered
    
###################

def reject_out_of_bounds_obs(df_input, dict_var_ranges=None):
    """
    Performs QC, removing all points where t, q, u10, or v10 is outside the acceptable range.
    """
    df = df_input.copy()

    if dict_var_ranges is None:
        dict_var_ranges = {'sta_t':(-40,50),
                           'sta_q':(0, 0.025), #note this is in kg/kg, not g/kg
                           'sta_u10':(-25,25),
                           'sta_v10':(-25,25)
                          }
    
    for col, (min_val, max_val) in dict_var_ranges.items():
        if col in df.columns:
            df = df[df[col].between(min_val, max_val)] # .between() is inclusive by default (both min and max are kept)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            
    return df
    
###################

def min_max_norm_ignore_extreme_fill_nan_sta_df(df_input, stats_path='stats.csv'):
    """
    Separate version of min_max_norm to apply to dataframes. Redundant, but it's easier than coercing the dataframe into an xarray object and then normalizing with the other version of this function.
    """

    df = df_input.copy()
    
    stats = pd.read_csv(stats_path)
    stats = stats.set_index('variable')

    target_cols = ['sta_t', 'sta_q', 'sta_u10', 'sta_v10']
    
    for col in target_cols:
        # Only process if the column exists in both the dataframe and the stats file
        if col in df.columns and col in stats.index:
            vmin = stats.loc[col, 'min']
            vmax = stats.loc[col, 'max']

            df[col] -= vmin
            df[col] *= 2.0 / (vmax - vmin)
            df[col] -= 1.0
            
            df[col] = np.where(df[col] > 1, 1, df[col])
            df[col] = np.where(df[col] < -1, -1, df[col])
            df[col] = np.nan_to_num(df[col], nan=0.0)
            
    return df
    
###################

def assemble_station_dataset(df_obs, lats_2d, lons_2d, analysis_time, time_col='OBS_TIMESTAMP', lat_col='lat', lon_col='lon', precision=6):
    """
    Takes in the dataframe of OBS_TIME_WINDOW hours of combined station data which has already been run through all the filtering, masking, etc functions.
    Returns an xarray object of the data binned to obs_time_window int coords (0,1,2,...) to merge with the RTMA/topo dataset
    """
    unique_times = sorted(df_obs[time_col].unique())
    time_map = {t: i for i, t in enumerate(unique_times)}
    
    df = df_obs.copy()
    
    # --- PRECISION CHECK: Round observations to [precision] decimal places ---
    df[[lat_col, lon_col]] = df[[lat_col, lon_col]].round(precision)
    
    df['obs_time_window'] = df[time_col].map(time_map)
    
    # Extract coordinates for the KDTree before setting the index 
    df_points = df[[lat_col, lon_col]].values
    t_indices = df['obs_time_window'].values  
    
    # Set MultiIndex for tracking
    df = df.drop(columns=[time_col]).set_index(['obs_time_window', lat_col, lon_col])
    
    # Flatten the original 2D meshgrids into an Nx2 coordinate list
    grid_points = np.column_stack((lats_2d.ravel(), lons_2d.ravel()))
    tree = KDTree(grid_points)
    distances, flat_indices = tree.query(df_points, k=1)
    y_indices, x_indices = np.unravel_index(flat_indices, lats_2d.shape)
    
    # 3D Shape to handle multiple hours (time, y, x) 
    num_times = len(unique_times)
    shape_3d = (num_times, lats_2d.shape[0], lats_2d.shape[1])
    data_vars_3d = {}
    
    for col in df.columns:
        # Since lat/lon are in the index now, they won't appear in df.columns anyway, but keep this safety check anyway
        # 'source' is handled separately below as a per-cell label array, not as a data channel
        if col in [lat_col, lon_col, 'obs_time_window', 'source']:
            continue

        grid_3d = np.full(shape_3d, np.nan)
        grid_3d[t_indices, y_indices, x_indices] = df[col].values
        data_vars_3d[col] = (('obs_time_window', 'y', 'x'), grid_3d)
    
    # Build an xarray Dataset with a time dimension
    ds = xr.Dataset(
        data_vars=data_vars_3d,
        coords={
            'obs_time_window': unique_times,
            'lat': (('y', 'x'), lats_2d),
            'lon': (('y', 'x'), lons_2d)
        }
    )
    
    # Add standard metadata attributes
    ds[lat_col].attrs = {'units': 'degrees_north'}
    ds[lon_col].attrs = {'units': 'degrees_east'}

    # Generate the 2D spatial obs_mask and fill NaNs with 0
    valid_time_ns = np.datetime64(analysis_time, 'ns')
    
    ds_final = (
        ds.assign(
            obs_mask=(
                ds.to_array()
                .notnull()
                .all(dim="variable")  # True if all vars present at this point
                .all(dim="obs_time_window")  # True if present across the whole window 
                .astype(int)   # Final 2D binary mask (y, x)
            )
        )
        .fillna(0)  # Replace all remaining NaNs with 0
        .assign_coords(valid_time=valid_time_ns) # Attach valid_time coordinate
    )

    # Tag each grid cell with its observation source (mesonet vs METAR), parallel to obs_mask.
    # Kept as its own per-cell label array (not a sta_* data channel) so it won't collide with a future data-quality channel.
    if 'source' in df.columns:
        source_code_map = {'mesonet': 1, 'metar': 2}
        source_grid = np.zeros(lats_2d.shape, dtype='int16')
        # np.maximum.at so a cell holding both sources across the window is labeled METAR (2) over mesonet (1), deterministically
        np.maximum.at(source_grid, (y_indices, x_indices), df['source'].map(source_code_map).values.astype('int16'))
        source_grid = source_grid * ds_final['obs_mask'].data  # only label cells that survived into obs_mask
        ds_final = ds_final.assign(obs_source=(('y', 'x'), source_grid))
        ds_final['obs_source'].attrs = {'flag_values': [0, 1, 2], 'flag_meanings': 'none mesonet metar'}

    return ds_final

###################

def reverse_norm_xr(xr_ds, var_name, stats_path=f"/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/data_preparation/stats.csv"):
    """
    Undoes the min-max norm for normalized data.
    Only for xarray datasets at the moment.
    Assumes the zeros at non-station locations have been removed.

    var_name = one of 'sta_t', 'sta_q', 'sta_u10', 'sta_v10'
    """
    xr_ds_tmp = xr_ds.copy()
    
    data = xr_ds_tmp[var_name].data
    stats = pd.read_csv(stats_path)
    stats = stats.set_index('variable')
    
    vmin = stats.loc[var_name, 'min']
    vmax = stats.loc[var_name, 'max']

    data = (data + 1) * (vmax - vmin) / 2 + vmin

    xr_ds_tmp[var_name].data = data
    
    return xr_ds_tmp


###############################################################
#################### GOES SATELLITE FUNCTIONS ####################
###############################################################

def download_multiple_goes16_files(target_time, channels, save_dir='./goes_cache', product='ABI-L1b-RadF-Reproc', verbose=False):
    """
    Maintains a rolling cache of GOES-16 files for target_time, target_time - 1h, and target_time - 2h.
    Gracefully skips missing AWS dates/directories.
    
    :param target_time: datetime object (UTC) representing the anchor time
    :param channels: list of integers representing channels, e.g., [2, 7, 10, 14]
    :param save_dir: target directory for the cache
    :param product: string representing the GOES product (default: ABI-L1b-RadF-Reproc)
    :return: list of local file paths currently active in the cache
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Reusable anonymous S3 client
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'noaa-goes16'

    # 1. Define the 3 hourly target timestamps (T, T-1h, T-2h)
    target_times = [target_time - timedelta(hours=i) for i in range(3)]
    
    desired_files = {}  # Format: {filename: s3_key}

    # 2. Resolve the exact S3 key for each (time, channel) pair
    if verbose:
        print("Resolving required AWS files...")
    for t in target_times:
        year, day_of_year, hour = t.strftime('%Y'), t.strftime('%j'), t.strftime('%H')
        prefix = f"{product}/{year}/{day_of_year}/{hour}/"
        
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        # Check if AWS has data for this specific prefix/hour
        if 'Contents' not in response or not response['Contents']:
            print(f"  Warning: No data found on AWS for prefix {prefix} (Skipping {t.strftime('%Y-%m-%d %H:%M UTC')})")
            continue

        for channel in channels:
            channel_str = f"C{int(channel):02d}"
            candidate_keys = [obj['Key'] for obj in response['Contents'] if channel_str in obj['Key']]
            
            if not candidate_keys:
                print(f"  Warning: No files for C{channel:02d} at {t.strftime('%Y-%m-%d %H:%M UTC')}")
                continue

            # Parse scan start time
            def parse_s3_time(key):
                for part in key.split('_'):
                    if part.startswith('s'):
                        return datetime.strptime(part[1:14], '%Y%j%H%M%S')
                return datetime.min

            closest_key = min(candidate_keys, key=lambda k: abs((parse_s3_time(k) - t).total_seconds()))
            filename = closest_key.split('/')[-1]
            desired_files[filename] = closest_key

    # Safety Guardrail: If all hours/channels were missing on AWS, stop here to avoid clearing local cache
    if not desired_files:
        print("Warning: No matching files were found on AWS for any requested time slots. Cache unchanged.")
        return []

    # 3. Purge unrequested files from local directory
    existing_files = set(os.listdir(save_dir))
    desired_filenames = set(desired_files.keys())

    for file in existing_files:
        filepath = os.path.join(save_dir, file)
        if os.path.isfile(filepath) and file not in desired_filenames:
            if verbose:
                print(f"Deleting stale file from cache: {file}")
            os.remove(filepath)

    # 4. Download missing requested files
    downloaded_paths = []
    for filename, s3_key in desired_files.items():
        local_path = os.path.join(save_dir, filename)
        if os.path.exists(local_path):
            if verbose:
                print(f"Cache hit (skipping download): {filename}")
        else:
            if verbose:
                print(f"Downloading new file: {filename}")
            s3.download_file(bucket_name, s3_key, local_path)
        
        downloaded_paths.append(local_path)

    if verbose:
        print(f"Sync complete. {len(downloaded_paths)} files active in cache.")
    
    return downloaded_paths

###################

def convert_xy_to_latlon(xr_goes):
    """
    Converts GOES-16 fixed grid scan angles (x, y) to latitude and longitude arrays.
    """
    goes_proj_attrs = xr_goes.goes_imager_projection.attrs
    
    # 1. Extract metadata values directly from goes_imager_projection
    h = goes_proj_attrs["perspective_point_height"] 
    lon_0 = goes_proj_attrs["longitude_of_projection_origin"] 
    a = goes_proj_attrs["semi_major_axis"] 
    b = goes_proj_attrs["semi_minor_axis"] 
    
    # 2. Define the geostationary projection
    p = Proj(proj="geos", h=h, lon_0=lon_0, sweep="x", a=a, b=b)
    
    # 3. Read 1D 'x' and 'y' scan angles from NetCDF (in radians)
    x = xr_goes.variables['x'][:]
    y = xr_goes.variables['y'][:]
    
    # 4. Scale radians by satellite height to convert to projection meters
    x_meters = x * h
    y_meters = y * h
    
    # 5. Build 2D grid and perform inverse projection to lat/lon
    X, Y = np.meshgrid(x_meters, y_meters)
    lon, lat = p(X, Y, inverse=True)
    
    # 6. Mask space/off-disk coordinates that return infinity or unprojected points
    invalid_mask = (lon < -180) | (lon > 180) | (lat < -90) | (lat > 90) | np.isinf(lon) | np.isinf(lat)
    lon[invalid_mask] = np.nan
    lat[invalid_mask] = np.nan

    return lon, lat # note order matches x, y

###################

def process_and_regrid_goes_cache(cached_files, lats_2d, lons_2d, analysis_time, obs_time_window=3, radius_of_influence=50000, regrid_cache=None):
    """
    Regrids cached GOES-16 'Rad' fields onto (lats_2d, lons_2d) and returns an xarray.Dataset
    indexed along the 'obs_time_window' dimension. Uses regrid_cache dict to reuse precomputed
    target definitions and KDTree neighbor information per channel across multiple time cycles.
    """
    if not cached_files:
        return None

    if regrid_cache is None:
        regrid_cache = {}

    # Convert target grid longitudes to [-180, 180] range to match GOES swath projections
    lons_2d_norm = ((lons_2d + 180) % 360) - 180

    # Target grid geometry definition (cached or created once)
    if 'target_def' not in regrid_cache:
        regrid_cache['target_def'] = geometry.GridDefinition(lons=lons_2d_norm, lats=lats_2d)
    target_def = regrid_cache['target_def']
    
    # Match the chronological ordering of assemble_station_dataset (T-2h, T-1h, T)
    time_list = sorted([pd.to_datetime(analysis_time - dt.timedelta(hours=i)) for i in range(obs_time_window)])
    time_to_idx = {t: i for i, t in enumerate(time_list)}
    
    channel_data = {}  # {channel_num: 3D array (time, y, x)}

    for filepath in cached_files:
        if not os.path.exists(filepath):
            continue
            
        with xr.open_dataset(filepath) as ds:
            # 1. Identify channel number from dataset coordinates or filename
            if 'band_id' in ds:
                chan = int(ds.band_id.data[0])
            elif 'channel_id' in ds:
                chan = int(ds.channel_id.data[0])
            else:
                for part in os.path.basename(filepath).split('_'):
                    if 'C' in part and part.startswith('M'):
                        chan = int(part.split('C')[-1])
                        break

            # 2. Extract and round file start time to match hourly windows
            time_str = ds.attrs.get('time_coverage_start', '')
            if time_str:
                file_time = pd.to_datetime(time_str)
                if file_time.tz is not None:
                    file_time = file_time.tz_localize(None)
                file_time = file_time.round('h')
            else:
                file_time = None
                for part in os.path.basename(filepath).split('_'):
                    if part.startswith('s'):
                        parsed_dt = datetime.strptime(part[1:14], '%Y%j%H%M%S')
                        file_time = pd.to_datetime(parsed_dt).round('h')
                        break

            if file_time is None:
                continue

            # Map file to closest window index
            closest_time = min(time_list, key=lambda t: abs((t - file_time).total_seconds()))
            t_idx = time_to_idx[closest_time]

            # 3. Precompute or retrieve cached KD-Tree neighbor info for channel
            if chan not in regrid_cache:
                lon, lat = convert_xy_to_latlon(ds)
                source_def = geometry.SwathDefinition(lons=lon, lats=lat)
                
                valid_input_index, valid_output_index, index_array, _ = kd_tree.get_neighbour_info(
                    source_def,
                    target_def,
                    radius_of_influence=radius_of_influence,
                    neighbours=1
                )
                regrid_cache[chan] = (valid_input_index, valid_output_index, index_array)

            valid_input_index, valid_output_index, index_array = regrid_cache[chan]

            # 4. Perform fast regridding using cached neighbor lookups
            rad_data = np.asanyarray(ds['Rad'].data)
            regridded_rad = kd_tree.get_sample_from_neighbour_info(
                'nn', #'nearest',
                target_def.shape,
                rad_data,
                valid_input_index,
                valid_output_index,
                index_array,
                fill_value=np.nan
            )

            # Initialize 3D array for channel if first encounter
            if chan not in channel_data:
                channel_data[chan] = np.full(
                    (obs_time_window, lats_2d.shape[0], lats_2d.shape[1]), 
                    np.nan, 
                    dtype=np.float32
                )

            channel_data[chan][t_idx] = regridded_rad.astype(np.float32)

    if not channel_data:
        return None

    # 5. Build output xarray Dataset
    data_vars = {}
    for chan, grid_3d in channel_data.items():
        var_name = f"goes_channel_{int(chan):02d}" 
        data_vars[var_name] = (('obs_time_window', 'y', 'x'), grid_3d)

    ds_goes = xr.Dataset(
        data_vars=data_vars,
        coords={
            'obs_time_window': time_list,
            'lat': (('y', 'x'), lats_2d),
            'lon': (('y', 'x'), lons_2d)
        }
    )

    return ds_goes

###################

def normalize_goes_dataset(ds_goes, stats_filepath):
    """
    Normalizes all GOES channel variables in ds_goes using min_max_norm_ignore_extreme_fill_nan_onevar_onetime.

    Inputs:
        - ds_goes --> xarray.Dataset containing regridded GOES channel variables (goes_channel_02, etc.)
        - stats_filepath --> path to stats.csv containing normalization constants
    """
    if ds_goes is None:
        return None

    ds_norm = ds_goes.copy()
    for var_name in ds_norm.data_vars:
        ds_norm[var_name] = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(ds_norm[var_name], var_name, stats_filepath)

    return ds_norm