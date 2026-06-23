# Standard Library
import os
import datetime as dt

# Data Science Core
import numpy as np
import pandas as pd
import xarray as xr

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
#################### HRRR/RTMA FUNCTIONS ####################
###############################################################

def fetch_regrid_crop_hrrr_rtma(adaf_var_str, hrrr_filepath, rtma_filepath, hrrr_regridder_filepath, dict_bounds=None):
    """
    Reads in and fetches the appropriate HRRR and RTMA field based on the given variable string (ADAF standard naming), then does the following:
        - Regrids HRRR onto the full RTMA grid
        - Restricts regridded HRRR and RTMA to the rectangular region containing data and ovewrites the RTMA data on the edges with 0 to match HRRR 
        
    Returns 2 full regridded xarray objects, one each for HRRR and RTMA of the variable requested. Further processing should happen in the calling function. 
    Also returns the dict_bounds object for use later - the intention behind this is to pass in None in the first loop, then use the data in the first loop (which should be temperature data, thus avoiding the potential pitfall with values=0) to initalize these bounds for use in all further vars/loops.
    Renames the field in the new xarray objects from the original name to the ADAF name, for convenience in the main workflow.

    Inputs:
        - adaf_var_str --> valid options = 't', 'q', 'u10', 'v10'
        - hrrr_filepath --> path to HRRR .grib2 file to be opened. Assumed to be a complete file, i.e. not already subset to a variable/variables
        - rtma_filepath --> path to RTMA .grib2 file to be opened. Assumed to be a complete file, i.e. not already subset to a variable/variables
        - hrrr_regridder_filepath --> filepath to regridder weights (.nc) file on disk. If the file DNE, weights are recalculated and stored at the provided filepath
        - dict_bounds (default = None) --> if provided, bounds are read from this dict; otherwise bounds are calculated and stored in this dict for future use
        
    Outputs:
        xr_hrrr_regridded_cropped, xr_rtma_cropped, dict_bounds
    """

    dict_level_selection = {"t":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                            "q":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}},
                            "u10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}},
                            "v10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}}}
    
    dict_adaf_translate = {'t':'t2m', 
                           'q':'sh2', 
                           'u10':'u10', 
                           'v10':'v10'}
    
    hrrr_backend_kwargs = dict(dict_level_selection[adaf_var_str])
    hrrr_backend_kwargs["indexpath"] = ""
    rtma_backend_kwargs = dict(dict_level_selection[adaf_var_str])
    rtma_backend_kwargs["indexpath"] = ""
    
    xr_hrrr = xr.open_dataset(hrrr_filepath, engine='cfgrib', backend_kwargs=hrrr_backend_kwargs)
    xr_rtma = xr.open_dataset(rtma_filepath, engine='cfgrib', backend_kwargs=rtma_backend_kwargs)

    if hrrr_regridder_filepath is not None and os.path.exists(hrrr_regridder_filepath):
        hrrr_regridder = xesmf.Regridder(xr_hrrr, xr_rtma, 'bilinear', weights=hrrr_regridder_filepath)
    else: #make regridder and save
        hrrr_regridder = xesmf.Regridder(xr_hrrr, xr_rtma, 'bilinear')
        hrrr_regridder.to_netcdf(hrrr_regridder_filepath)

    # Rename the field to the ADAF name (has to be done on the full dataset before it gets subset to a DataArray)
    xr_hrrr = xr_hrrr.rename_vars({dict_adaf_translate[adaf_var_str] : adaf_var_str})
    xr_rtma = xr_rtma.rename_vars({dict_adaf_translate[adaf_var_str] : adaf_var_str})

    # Subset down to only the variable we care about
    xr_hrrr = xr_hrrr[adaf_var_str]
    xr_rtma = xr_rtma[adaf_var_str]

    xr_hrrr_regridded = hrrr_regridder(xr_hrrr, keep_attrs=True)

    # Crop both HRRR and RTMA to the minimal region
    # Make new bounds if no dict_bounds provided; else read from provided dict
    # Intention is to create dict_bounds from temperature (which should be the first variable done), then save those bounds for the other vars AND topography
    if dict_bounds is None:
        data_rows = np.where(np.any(xr_hrrr_regridded.data != 0, axis=1))[0]
        data_cols = np.where(np.any(xr_hrrr_regridded.data != 0, axis=0))[0]
        row_start, row_end = data_rows[0], data_rows[-1]+1 # +1 is necessary for proper cropping
        col_start, col_end = data_cols[0], data_cols[-1]+1 # +1 is necessary for proper cropping

        dict_bounds = {'row_start' : row_start,
                       'row_end' : row_end,
                       'col_start' : col_start,
                       'col_end' : col_end}
    else:
        row_start = dict_bounds['row_start']
        row_end = dict_bounds['row_end']
        col_start = dict_bounds['col_start']
        col_end = dict_bounds['col_end']

    xr_hrrr_regridded_cropped = xr_hrrr_regridded.isel({'y': slice(row_start, row_end), 'x': slice(col_start, col_end)})
    xr_rtma_cropped = xr_rtma.isel({'y': slice(row_start, row_end), 'x': slice(col_start, col_end)})

    # Make the RTMA data zero in the same locations as xr_hrrr_regridded_cropped
    # !! This may have implications for non-temperature data, as there may be some points in the interior of the domain that are zero for HRRR but not for RTMA !!
    # If this is a problem in the future, may need to add an affine shift to the RTMA data (say, +100), then fill based on that, then subtract off the shift
    # But for now, not dealing with it
    xr_rtma_cropped = xr_rtma_cropped.where(xr_hrrr_regridded_cropped != 0, 0)

    #Sanity check for future work - ensure the cropped data has the expected dimensions (hardcoded)
    if (np.shape(xr_hrrr_regridded_cropped.data) != (1356, 2294)) or (np.shape(xr_rtma_cropped.data) != (1356, 2294)):
        print(f"!! Cropped data does not have the expected shape of (1356, 2294) !!")
        print(f"HRRR shape: {np.shape(xr_hrrr_regridded_cropped.data)} | RTMA shape: {np.shape(xr_rtma_cropped.data)}")

    return xr_hrrr_regridded_cropped, xr_rtma_cropped, dict_bounds
    

###################
    
def fix_dataset_scaling_shifting(xr_data, adaf_var_str):
    """
    Fixes the scaling/shifting of the data of HRRR/RTMA to match ADAF. Only temperature needs to be modified.
    This function must be run before min/max normalization!

    Inputs:
        - xr_data --> either a HRRR or RTMA xarray dataset, resulting from fetch_and_regrid_hrrr_rtma
        - adaf_var_str --> valid options = 't', 'q', 'u10', 'v10'
    """

    if adaf_var_str == 't':
        xr_data = xr_data -273.15 #convert K to C
        return xr_data.where(~np.isclose(xr_data, -273.15), 0) #set all values close to -273.15 to 0
    elif adaf_var_str == 'q':
        return xr_data #no correction needed
    elif adaf_var_str == 'u10':
        return xr_data #no correction needed
    elif adaf_var_str == 'v10':
        return xr_data #no correction needed
    else:
        print(f"Invalid variable abbreviation - should only be 't', 'q', 'u10', 'v10'")
    
###################

def min_max_norm_ignore_extreme_fill_nan_onevar_onetime(xr_data, adaf_var_str, stats_filepath):
    """
    Modified version of the min_max_norm_ignore_extreme_fill_nan function. 
    This one does only a single time (i.e. no time dimension) and a single variable, as the original code doesn't seem to work with all variables at once.

    Inputs:
        - xr_data --> HRRR or RTMA data that has been run through fetch_and_regrid_hrrr_rtma and fix_dataset_scaling_shifting. Should be 2D
        - adaf_var_str --> string to select the right variable's stats. MUST BE MODIFIED BEFOREHAND, e.g. "rtma_t" instead of just "t"
        - stats_filepath --> str of filepath to the stats.csv file containing variable min/max values
    """

    stats = pd.read_csv(stats_filepath, index_col=0)
    field_tar_stats = stats.loc[stats['variable'] == adaf_var_str]

    #Loop version of the ndim code
    vmin = field_tar_stats["min"]
    vmax = field_tar_stats["max"]
    vmin = np.array(vmin)
    vmax = np.array(vmax)
    
    for j in range(xr_data.ndim-1):
        vmin = vmin[:, np.newaxis]
        vmax = vmax[:, np.newaxis]
    
    for j in range(xr_data.ndim): #no temporal dimension yet - if that gets included as dim 0, then this should only apply to dims 1,2,..,n
        vmin = np.repeat(vmin, xr_data.shape[j], axis=j)
        vmax = np.repeat(vmax, xr_data.shape[j], axis=j) 

    xr_data -= vmin
    xr_data *= 2.0/(vmax-vmin)
    xr_data -= 1.0

    #Mask out-of-range values; xr.where is a little odd for replacing values
    xr_data = xr_data.where(xr_data <= 1, other=1)
    xr_data = xr_data.where(xr_data >= -1, other=-1)

    xr_data = xr_data.fillna(0) 

    return xr_data


###############################################################
#################### STATION OBS FUNCTIONS ####################
###############################################################

def load_mesonet_into_dataframe_and_clean(path, 
                                          ADAF_CHANNELS=None, 
                                          GOOD_QM=None, 
                                          QM_FILLVALUE=None, 
                                          FLOAT_FILLVALUE=None,
                                          LAT_BOUNDS=(20.0,55,0),
                                          LON_BOUNDS=(225.0, 300.0),
                                          drop_qm_cols=True):
    """
    Function to load Mesonet data (in IODA standard) from disk and turn it into a cleaned-up pandas Dataframe.
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

    ## Sort by coordinates, the target hour, and then the time difference
    ## This puts the "closest" record at the top of each (Location + Hour) group
    df_filtered = df.sort_values(by=[lat_col, lon_col, 'target_hour', 'time_diff']).drop_duplicates(subset=[lat_col, lon_col, 'target_hour'], keep='first')

    #Clean up helper columns
    return df_filtered.drop(columns=['target_hour', 'time_diff']).sort_values(by=[time_col, lat_col])
    
###################

def filter_obs_by_temporal_completeness(df_input, obs_time_window, precision=6):
    """
    Checks if each (lat, lon) point has a row for every hour in the window.
    Points missing any hour are removed entirely.
    """
    df = df_input.copy()
    
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
    Returns an xarray object of the data binned to obs_time_window int coords (0,1,2,...) to merge with the HRRR/RTMA/topo dataset
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
        if col in [lat_col, lon_col, 'obs_time_window']:
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
    
    return ds_final

###################

def reverse_norm_xr(xr_ds, var_name, stats_path=f"/scratch3/BMC/wrfruc/aschein/ADAF/data_preparation_new/stats.csv"):
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
    
    
#################################################################
#################### MISCELLANEOUS FUNCTIONS ####################
#################################################################

#(2026-06-23) NEEDS FIXING FOR RTMA
def plot_data(xr_dataset, title_str="TITLE", thinning=2):
    extent = [-130, -63, 22, 54] #USA
    projection = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(1,1, figsize=(14,9), subplot_kw={'projection':projection})
    
    ax.set_extent(extent, crs=projection)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3)

    try:
        X, Y = np.meshgrid(xr_dataset.lon, xr_dataset.lat)
    except:
        (X, Y) = (xr_dataset.longitude, xr_dataset.latitude)
        
    im = ax.scatter(X[::thinning], Y[::thinning], c=xr_dataset.data[::thinning], s=0.2, cmap='coolwarm') 
    plt.colorbar(im, shrink=0.4, pad=0.01)

    plt.title(title_str)