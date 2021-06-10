### mkwc_util.py : Contains utilities for extracting and processing data from the MKWC website
### Author : Emily Ramey
### Date : 6/1/2021

import os
import numpy as np
import pandas as pd
import time_util as times

### NOTE: Seeing data is STORED by UT, with data in HST
### CFHT data is STORED by HST, with data in HST

# cfht_dir = "/u/emily_ramey/work/Keck_Performance/data/weather_data/"
cfht_dir = './'
mkwc_url = 'http://mkwc.ifa.hawaii.edu/'
year_url = mkwc_url+'archive/wx/cfht/cfht-wx.{}.dat'
cfht_cutoff = 55927.41666667 # 01/01/2012 12:00 am HST

# seeing_dir = "/u/emily_ramey/work/Keck_Performance/data/seeing_data"
seeing_dir = './'

# Time columns from MKWC data
time_cols = ['year', 'month', 'day', 'hour', 'minute', 'second']

# Data-specific fields
data_types = {
    'cfht': {
        'web_pat': mkwc_url+'archive/wx/cfht/indiv-days/cfht-wx.{}.dat',
        'file_pat': cfht_dir+"cfht-wx.{}.dat",
        'data_cols': ['wind_speed', 'wind_direction', 'temperature',
                 'relative_humidity', 'pressure'],
    },
    'mass': {
        'web_pat': mkwc_url+'current/seeing/mass/{}.mass.dat',
        'file_pat': seeing_dir+'mass/{}.mass.dat',
        'data_cols': ['mass'],
    },
    'dimm': {
        'web_pat': mkwc_url+'current/seeing/dimm/{}.dimm.dat',
        'file_pat': seeing_dir+'dimm/{}.dimm.dat',
        'data_cols': ['dimm'],
    },
    'masspro': {
        'web_pat': mkwc_url+'current/seeing/masspro/{}.masspro.dat',
        'file_pat': seeing_dir+'masspro/{}.masspro.dat',
        'data_cols': ['masspro_half', 'masspro_1', 'masspro_2', 
                      'masspro_4', 'masspro_8', 'masspro_16', 'masspro'],
    },
}

# Mix and match data & time columns
for dtype in data_types:
    # CFHT files don't have seconds
    tcols = time_cols if dtype != 'cfht' else time_cols[:-1]
    # Format web columns
    data_types[dtype]['web_cols'] = tcols+data_types[dtype]['data_cols']
    # Format file columns
    data_types[dtype]['cols'] = [dtype+'_mjd']+data_types[dtype]['data_cols']
    # Different file storage timezones
    data_types[dtype]['file_zone'] = 'hst' if dtype=='cfht' else 'utc'

#############################
######### Functions #########
#############################

def cfht_from_year(datestrings, year):
    """ 
    Gets pre-2012 MKWC data from the year-long file 
    instead of the by-date files 
    """
    # Get year-long file URL
    url = year_url.format(year)
    
    # Read in data
    try:
        web_cols = data_types['cfht']['web_cols']
        year_data = pd.read_csv(url, delim_whitespace=True, header=None,
                        names=web_cols, usecols=range(len(web_cols)))
    except: # No data, return blank
        return pd.DataFrame(columns=data_types['cfht']['cols'])
    
    # Full dataset
    all_data = [pd.DataFrame(columns=data_types['cfht']['cols'])]
    
    # Slice up dataframe
    for ds in datestrings:
        month, day = int(ds[4:6]), int(ds[6:])
        # Get data by month and day
        df = year_data.loc[(year_data.month==month) & (year_data.day==day)].copy()
        # Format columns
        if not df.empty:
            format_columns(df, 'cfht')
        # Add to full dataset
        all_data.append(df)
    
    return pd.concat(all_data)

def format_columns(df, dtype):
    """ Changes columns (in place) from time_cols to MJD """
    # Get MJDs from HST values
    mjds = times.table_to_mjd(df, columns=time_cols, zone='hst')
    df[dtype+'_mjd'] = mjds
    
    # Drop old times
    df.drop(columns=time_cols, inplace=True, errors='ignore')

def from_url(datestring, dtype):
    """ Pulls cfht file from MKWC website """
    # Format URL
    url = data_types[dtype]['web_pat'].format(datestring)
    
    # Read data
    try: # Check if data is there
        web_cols = data_types[dtype]['web_cols'] # MKWC Weather columns
        df = pd.read_csv(url, delim_whitespace = True, header=None,
                    names=web_cols, usecols=range(len(web_cols)))
    except: # otherwise return blank df
        return pd.DataFrame(columns=data_types[dtype]['cols'])
    
    # Get mjd from time columns
    format_columns(df, dtype)
    
    return df

def from_file(filename, dtype):
    """ Pulls cfht or seeing file from local directory """
    # Read in CSV
    df = pd.read_csv(filename)

    # Check formatting
    if 'mjd' in df.columns:
        df.rename(columns={'mjd':dtype+'_mjd'}, inplace=True)
    elif dtype+'_mjd' not in df.columns: # No MJD
        try: # Change to MJD, if times
            format_columns(df, dtype)
        except: # No time info, return blank
            return pd.DataFrame()
    
    return df

def from_nirc2(mjds, dtype):
    """ 
    Compiles a list of cfht or seeing observations based on MJDs
    note: does not compare MJDs; assumption is inputs are rounded to nearest day
    """
    
    # Get datestrings
    dts = times.mjd_to_dt(mjds, zone=data_types[dtype]['file_zone'])
    datestrings = dts.strftime("%Y%m%d") # e.g. 20170826
    # No duplicates
    datestrings = pd.Series(np.unique(datestrings))
    
    # Blank data structure
    all_data = [pd.DataFrame(columns=data_types[dtype]['cols'])]
    
    # Check for pre-2012 cfht files
    if dtype=='cfht' and any(mjds < cfht_cutoff):
        # Get datetimes
        pre_2012 = datestrings[mjds < cfht_cutoff]
        
        # Compile data by year
        for yr in np.unique(pre_2012.str[:4]):
            ds = pre_2012[pre_2012.str[:4]==yr]
            df = cfht_from_year(ds, yr)
            # Append to full dataset
            all_data.append(df)
    
    # Find data for each file
    for ds in datestrings:
        # Get local filename
        filename = data_types[dtype]['file_pat'].format(ds)
        
        # Check for local files
        if os.path.isfile(filename):
            df = from_file(filename, dtype)
            
        else: # Pull from the web
            df = from_url(ds, dtype)
            
            # Save to local file: TODO
        
        # Add to data
        all_data.append(df)
    
    # Return concatenated dataframe
    return pd.concat(all_data, ignore_index=True)