### temp_util.py: Functions relating to Keck-II temperature data
### Main reads in Keck II temperature files (AO, env, and L4) and saves each to a FITS file
### Author: Emily Ramey
### Date: 01/04/2021

import os
import numpy as np
import pandas as pd
import glob
from astropy.io import fits
from astropy.table import Table

from . import times

verbose = True

# data_dir = "/u/emily_ramey/work/Keck_Performance/data/"
# temp_dir = data_dir+"temp_data_2/"

col_dict = {
    'AO_benchT1': 'k1:ao:env:benchtemp1_Raw', # k2ENV
    'AO_benchT2': 'k1:ao:env:benchtemp2_Raw',
    'k1:ao:env:elect_vault_t': 'k1:ao:env:elect_vault:T',
    'k0:met:humidityStats.VAL': 'k0:met:humidityStats',
    "k2:ao:env:ETroomtemp_Raw": "El_roomT", # k2AO
    "k2:ao:env:DMracktemp_Raw": "DM_rackT",
    "k2:ao:env:KCAMtemp2_Raw": "KCAM_T2",
    "k2:ao:env:OBrighttemp_Raw": "OB_rightT",
    "k2:ao:env:phototemp_Raw": "photometricT",
    "k2:ao:env:OBlefttemp_Raw": "OB_leftT",
    "k2:ao:env:KCAMtemp1_Raw": "KCAM_T1",
    "k2:ao:env:ACAMtemp_Raw": "AOA_camT",
    "k2:ao:env:LStemp_Raw": "LGS_temp", # k2L4
    "k2:ao:env:LSenchumdity_Raw": "LGS_enclosure_hum",
    "k2:ao:env:LShumidity_Raw": "LGS_humidity",
    "k2:ao:env:LSenctemp_Raw": "LGS_enclosure_temp"
}

# file_pats = { # File name patterns
# #     'k1AOfiles': temp_dir+"k1AOtemps/*/*/*/AO_bench_temps.log",
# #     'k1LTAfiles': temp_dir+"k1LTAtemps/*/*/*/LTA_temps.log",
# #     'k1ENVfiles': temp_dir+"k1envMet/*/*/*/envMet.arT",
#     'k2AO': temp_dir+"k2AOtemps/*/*/*/AO_temps.log",
#     'k2L4': temp_dir+"k2L4temps/*/*/*/L4_env.log",
#     'k2ENV': temp_dir+"k2envMet/*/*/*/envMet.arT",
# }

data_types = {
    'k2AO': {
        'file_pat': "{}{}/AO_temps.log",
        'cols': ['AOA_camT', 'DM_rackT', 'El_roomT', 'KCAM_T1', 
                 'KCAM_T2', 'OB_leftT', 'OB_rightT', 'photometricT', 
                 'k2AO_mjd'],
    },
    'k2L4': {
        'file_pat': "{}/{}/L4_env.log",
        'cols': ['LGS_enclosure_hum', 'LGS_enclosure_temp', 'LGS_humidity',
                 'LGS_temp', 'k2L4_mjd'],
    },
    'k2ENV': {
        'file_pat': "{}/{}/envMet.arT",
        'cols': ['k0:met:dewpointMax', 'k0:met:dewpointMin', 'k0:met:dewpointRaw', 
                 'k0:met:humidityRaw', 'k0:met:humidityStats', 'k0:met:out:windDirection',
                 'k0:met:out:windDirectionMax', 'k0:met:out:windDirectionMin', 
                 'k0:met:out:windSpeedMaxStats', 'k0:met:out:windSpeedMaxmph',
                 'k0:met:out:windSpeedmph', 'k0:met:outTempDwptDiff',
                 'k0:met:pressureRaw', 'k0:met:pressureStats', 'k0:met:tempMax',
                 'k0:met:tempMin', 'k0:met:tempRaw', 'k2:dcs:sec:acsDwptDiff',
                 'k2:dcs:sec:acsTemp', 'k2:dcs:sec:secDwptDiff', 'k2:dcs:sec:secondaryTemp', 
                 'k2:met:humidityRaw','k2:met:humidityStats', 'k2:met:tempRaw', 'k2:met:tempStats',
                 'k2:met:windAzRaw', 'k2:met:windElRaw', 'k2:met:windSpeedMaxmph',
                 'k2:met:windSpeedMinmph', 'k2:met:windSpeedRaw', 'k2:met:windSpeedStats', 
                 'k2:met:windSpeedmph', 'k2ENV_mjd'],
    }
}

# def get_columns():
#     all_data = search_files()
#     for name, data_files in all_data.items():
#         columns = set()
#         for i,file in enumerate(data_files):    
#             try:
#                 df = pd.read_csv(file, header=1, skiprows=[2], quoting=3, skipinitialspace=True,
#                                  na_values=['***'], error_bad_lines=False, warn_bad_lines=False,
#                                 ).replace('"', regex=True)
#             except:
#                 if verbose:
#                     print(f"Warning: read failed for file {file}")
#                 continue

#             if len(df.columns)==1: # no header
#                 if verbose:
#                     print(f"Skipping file {file}, no header")
#                 continue # skip for now
#             df.columns = [col.replace('"', '').strip() for col in df.columns]
#             if "UNIXDate" not in df.columns:
#                 if verbose:
#                     print(f"Skipping file {file}, columns not as expected")
#                 continue # skip for now
#             for col in df.columns:
#                 columns.add(col)
#         all_data[name] = columns
    
#     return all_data


# def search_files(file_pats=file_pats):
#     """ Finds all filenames matching the given file patterns """
#     all_filenames = {}
#     for name, search in file_pats.items():
        
#         filenames = glob.glob(search, recursive=True)
#         all_filenames[name] = filenames
#         if verbose:
#             print(f"Done {name}, length: {len(filenames)}")
    
#     return all_filenames

def collect_data(data_files, col_dict=col_dict):
    """ 
    Takes a list of file names and reads them into a pandas dataframe.
    data_files: list of filenames
    col_dict: dictionary of columns to rename
    returns: dataframe of temperature data from files
    """
    if isinstance(data_files, dict): # Return dict w/dataframes
        new_files = {}
        for name, files in data_files.items():
            if isinstance(files, list):
                # Recurse on each list of files
                new_files[name] = collect_data(files)
        return new_files
    
    all_dfs = [pd.DataFrame()]
    for i,file in enumerate(data_files):   
        if not os.path.isfile(file):
            continue
        try:
            df = pd.read_csv(file, header=1, skiprows=[2], quoting=3, skipinitialspace=True,
                             na_values=['***'], error_bad_lines=False, warn_bad_lines=False,
                            ).replace('"', regex=True)
        except:
            if verbose:
                print(f"Warning: read failed for file {file}")
            continue

        if len(df.columns)==1: # no header
            if verbose:
                print(f"Skipping file {file}, no header")
            continue # skip for now
        df.columns = [col.replace('"', '').strip() for col in df.columns]
        if "UNIXDate" not in df.columns:
            if verbose:
                print(f"Skipping file {file}, columns not as expected")
            continue # skip for now
        
        if col_dict is not None:
            df = df.rename(columns=col_dict)
        
        all_dfs.append(df)
    
    data = pd.concat(all_dfs, ignore_index=True, sort=True)
    return data

def parse_dates(data, date_cols={'HST': ['HSTdate', 'HSTtime'], 'UNIX': ['UNIXDate', 'UNIXTime']}):
    """ 
    Parses specified date and time columns and returns a cleaned data table.
    """
    new_data = data.copy()
    for label,cols in date_cols.items():
        date_col, time_col = cols
        # Parse dates and times, coercing invalid strings to NaN
        datetimes = (pd.to_datetime(data[date_col], exact=False, errors='coerce') + 
                     pd.to_timedelta(data[time_col], errors='coerce'))
        new_data = new_data.drop(columns=cols)
        new_data[label] = datetimes
    
    return new_data

def clean_dates(data, date_cols=['HST', 'UNIX']):
    """ Removes any rows containing invalid dates in date_cols """
    new_data = data.copy()
    for col in date_cols:
        new_data = new_data[~np.isnan(new_data[col])]
    
    return new_data

def clean_data(data, data_cols=None, non_numeric=['HST', 'UNIX']):
    """ Casts columns to a numeric data type """
    if data_cols is None:
        data_cols = [col for col in data.columns if col not in non_numeric]
    
    # Cast data to numeric type, coercing invalid values to NaN
    new_data = data.copy()
    for col in data_cols:
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    
    return new_data

# def to_fits(data, filename, str_cols=['HST', 'UNIX']):
#     """ Writes a FITS file from the given temperature array """
#     fits_data = data.copy()
#     for col in ['k0:met:GEUnitInvalram', 'k0:met:GEunitSvcAlarm']:
#         if col in fits_data.columns:
#             fits_data = fits_data.drop(columns=[col])
#     for col in str_cols:
#         if col in fits_data.columns:
#             fits_data[col] = fits_data[col].astype(str)
#     # Assuming the data columns are already numeric
#     fits_data = Table.from_pandas(fits_data)
#     fits_data.write(filename)
    
#     return

# def from_fits(filename, date_cols=['HST', 'UNIX'], str_cols=['k0:met:GEUnitInvalram', 'k0:met:GEunitSvcAlarm']):
#     """ Reads in a fits file, converts to pandas, and parses date columns (if specified) """
#     data = Table.read(filename).to_pandas()
    
#     # Fix NaNs, because astropy is dumb sometimes
#     data[data==1e+20] = np.nan
    
#     if date_cols is None: return data
    
#     for col in date_cols:
#         if isinstance(data[col][0], bytes): # Cast bytes to utf-8 strings
#             data[col] = data[col].str.decode("utf-8")
#         data[col] = pd.to_datetime(data[col], exact=False, errors='coerce')
    
#     if str_cols is None: return data
    
#     for col in str_cols:
#         if col in data.columns and isinstance(data[col][0], bytes):
#             data[col] = data[col].str.decode("utf-8")
    
#     return data

# def combine_and_save(file_pats=file_pats, location=temp_dir, filename=None):
#     """ 
#     Reads in all data matching file patterns (file_pats), combines them into one table, 
#     cleans them, and saves them to a FITS file
#     """
#     # Find all files matching pattern
#     all_filenames = search_files(file_pats)
#     # Read all data into one table (per dictionary label)
#     all_data = collect_data(all_filenames)
    
#     for name, data in all_data.items():
#         data = parse_dates(data) # Parse date cols into datetimes
#         data = clean_dates(data) # Remove invalid dates
#         data = clean_data(data) # Casts other cols to numeric
#         # Save combined/cleaned data to FITS file
#         filename = location+name+".fits"
#         to_fits(data, filename)
    
#     return

def from_mjds(mjds, dtype, data_dir):
    """ 
    Gets temp data of input type from the specified MJDs.
    mjds: list of Modified Julian Dates to pull data from
    dtype: k2AO, k2L4, or k2ENV file type
    data_dir: path to directory containing temperature data
    returns: dataframe of relevant temperature data
    """
    # Get pd datetimes in HST
    dts = times.mjd_to_dt(mjds, zone='hst')
    # Format list
    datestrings = dts.strftime("%y/%m/%d")
    datestrings = np.unique(datestrings) # one file per date
    # Get relevant filenames
    filenames = [data_types[dtype]['file_pat'].format(data_dir, ds) for ds in datestrings]
    # Get data from filenames
    data = collect_data(filenames)
    
    # Empty dataframe
    if data.empty:
        return pd.DataFrame(columns=data_types[dtype]['cols'])
    
    # Convert dates & times to MJDs
    data["datetime"] = data['HSTdate']+' '+data['HSTtime']
    mjds = times.table_to_mjd(data, columns='datetime', zone='hst')
    # Add to dataframe
    data[dtype+"_mjd"] = mjds
    # Drop other date & time cols
    data.drop(columns=["HSTdate", "HSTtime", "UNIXDate", "UNIXTime", "datetime", 'mjdSec'],
             inplace=True, errors='ignore')
    
    return data