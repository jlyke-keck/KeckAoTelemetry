### nirc2_util.py : Functions for processing nirc2 data and strehl files
### Author : Emily Ramey
### Date : 6/2/2021

import os
import numpy as np
import pandas as pd
import glob
from astropy.io import fits

from . import times

strehl_cols = ['nirc2_file', 'strehl', 'rms_err', 'fwhm', 'nirc2_mjd']
default_keys = ['AIRMASS', 'ITIME', 'COADDS', 'FWINAME', 'AZ', 'DMGAIN', 'DTGAIN',
              'AOLBFWHM', 'WSFRRT', 'LSAMPPWR', 'LGRMSWF', 'AOAOAMED', 'TUBETEMP']

def from_filename(nirc2_file, data, i, header_kws):
    """ 
    Gets nirc2 header values from a filename as dict 
    or loads values into specified df
    """
    # Check for valid file
    if not os.path.isfile(nirc2_file):
        return
    # Open nirc2 file
    with fits.open(nirc2_file) as file:
        nirc2_hdr = file[0].header
    
    # Get fields from header
    for kw in header_kws:
        # load DataFrame value
        data.loc[i,kw.lower()] = nirc2_hdr.get(kw, np.nan)

def from_strehl(strehl_file, header_kws=default_keys):
    """ Gets NIRC2 header data based on contents of Strehl file """
    # Get directory name
    data_dir = os.path.dirname(strehl_file)
    # Retrieve Strehl data
    strehl_data = pd.read_csv(strehl_file, delim_whitespace = True, 
                              header = None, skiprows = 1, names=strehl_cols)
    # Add true file path
    strehl_data['nirc2_file'] = data_dir + "/" + strehl_data['nirc2_file']
    
    # Add decimal year
    strehl_data['dec_year'] = times.mjd_to_yr(strehl_data.nirc2_mjd)
    
    # Add nirc2 columns
    for col in header_kws:
        strehl_data[col.lower()] = np.nan
    
    # Loop through nirc2 files
    for i,nirc2_file in enumerate(strehl_data.nirc2_file):
        # Load header data into df
        from_filename(nirc2_file, strehl_data, i, header_kws)
    
    # Return data
    return strehl_data
        
        

