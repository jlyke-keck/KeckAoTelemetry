### nirc2_util.py : Functions for processing nirc2 data and strehl files
### Author : Emily Ramey
### Date : 6/2/2021

import os
import numpy as np
import pandas as pd
import glob
from astropy.io import fits

import time_util as times

MAX_LEN = 10 # Max char length of epoch name

nirc2_dir = "/g/lu/data/gc/lgs_data/"
strehl_pat = nirc2_dir+"{}/clean/kp/{}"

strehl_filenames = ['strehl_source.txt', 'irs33N.strehl']
strehl_cols = ['nirc2_file', 'strehl', 'rms_err', 'fwhm', 'nirc2_mjd']
header_kws = ['AIRMASS', 'ITIME', 'COADDS', 'FWINAME', 'AZ', 'DMGAIN', 'DTGAIN',
              'AOLBFWHM', 'WSFRRT', 'LSAMPPWR', 'LGRMSWF', 'AOAOAMED', 'TUBETEMP']

# Note: need to get better at identifying which strehl files we want
# Alternatively, we could have people provide a list
def search_epochs(data_dir=nirc2_dir):
    """ Searches for valid epoch names in NIRC2 data directory """
    # Epochs with valid strehl files
    good_files = {}
    # Loop through all sub-directories
    for epoch in os.listdir(data_dir):
#         # Only valid if {yr}{month}lgs
#         if len(epoch) >= MAX_LEN:
#             continue
        
        # Search epoch for valid strehl file
        for file in strehl_filenames:
            strehl_file = strehl_pat.format(epoch, file)
            # If good, add to dict
            if os.path.isfile(strehl_file):
                good_files[epoch] = strehl_file
    # Returns {epoch:strehl} dict
    return good_files

def from_filename(nirc2_file, data, i):
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

def from_strehl(strehl_file):
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
        from_filename(nirc2_file, strehl_data, i)
    
    # Return data
    return strehl_data
        
        

