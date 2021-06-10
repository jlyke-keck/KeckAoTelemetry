### compiler.py : Compiler for all desired data (nirc2, weather, seeing, telemetry, temperature)
### Author : Emily Ramey
### Date : 6/3/21

import time_util as times
import nirc2_util as nirc2
import telem_util as telem
import temp_util as temp
import mkwc_util as mkwc

import numpy as np
import pandas as pd

accept_labels = ['cfht', 'mass', 'dimm', 'masspro', 'k2AO', 'k2L4', 'k2ENV', 'telem']

expand = {
    'temp': ['k2AO', 'k2L4', 'k2ENV'],
    'seeing': ['mass', 'dimm', 'masspro'],
    'all': accept_labels,
}

# Utility functions
def check_dtypes(data_types):
    """ Returns an expanded / cleaned list of data types from user input """
    new_dtypes = []
    # Check requested data
    for dtype in data_types:
        # Check for nicknames
        if dtype in expand:
            new_dtypes.extend(expand[dtype])
        elif dtype in accept_labels: # can get data
            new_dtypes.append(dtype)
    # Return cleaned list
    return new_dtypes

#############################################
######## Interface with other modules #######
#############################################

def data_func(dtype):
    """ 
    Returns the function to get data for the specified dtype 
    and whether data needs to be matched to nirc2 mjds
    """
    if dtype in ['cfht']+expand['seeing']:
        return lambda files,mjds: mkwc.from_nirc2(mjds,dtype), True
    if dtype in expand['temp']:
        return lambda files,mjds: temp.from_mjds(mjds,dtype), True
    if dtype=='telem':
        return lambda files,mjds: telem.from_nirc2(mjds,files), False


################################
###### Compiler Functions ######
################################

def match_data(mjd1, mjd2):
    """ 
    Matches data1 with its closest points in data2, by mjd values
    Returns an array containing data1 with corresponding rows from data2
    """
    # Edge case
    if mjd1.empty or mjd2.empty:
        return None
    
    # Get mjds from dataframes
    mjd1 = np.array(mjd1).reshape(-1,1) # column vector
    mjd2 = np.array(mjd2).reshape(1,-1) # row vector
    
    # Take difference of mjd1 with mjd2
    diffs = np.abs(mjd1 - mjd2)
    # Find smallest difference for each original mjd
    idxs = np.argmin(diffs, axis=1)
    
    # Return indices of matches in mjd2
    return idxs
    

# data_types can contain: 'chft', 'mass', 'dimm', 'masspro',
# 'telem', 'k2AO', 'k2L4', or 'k2ENV'
# 'temp' or 'seeing' will be expanded to ['k2AO', 'k2L4', 'k2ENV'] and 
# ['mass', 'dimm', 'masspro'], respectively
def combine_strehl(strehl_file, data_types, check=True, test=False):
    """ 
    Combines and matches data from a certain Strehl file with other specified data types.
    NIRC2 files must be in the same directory as the Strehl file.
    """
    
    nirc2_data = nirc2.from_strehl(strehl_file)
    
    if test: # Take first few files
        nirc2_data = nirc2_data.loc[:3]
    
    if check: # Sanitize user input
        data_types = check_dtypes(data_types)
    
    # Full data container
    all_data = [nirc2_data.reset_index(drop=True)]
    
    # Loop through and get data
    for dtype in data_types:
        get_data, match = data_func(dtype) # Data retrieval function
        # Get other data from strehl info
        other_data = get_data(nirc2_data.nirc2_file, nirc2_data.nirc2_mjd)
        
        if match: # Needs to be matched
            if other_data.empty: # No data found
                other_data = pd.DataFrame(columns=other_data.columns, index=range(len(nirc2_data)))
            else: # Get indices of matched data
                idxs = match_data(nirc2_data.nirc2_mjd, other_data[dtype+'_mjd'])
                other_data = other_data.iloc[idxs]
            
#             # Edge case: no data
#             if idxs is not None:
#                 other_data = other_data.iloc[idxs]
        
        # Add to all data
        all_data.append(other_data.reset_index(drop=True))
    
    # Concatenate new data with nirc2
    return pd.concat(all_data, axis=1)

#################################
######### Full Compiler #########
#################################

def compile_all(data_dir, data_types=['all'], test=False, save=False):
    """ Compiles and matches requested data from all strehl files in a given nirc2 directory """
    
    # Expand requested data types
    data_types = check_dtypes(data_types)
    
    # Get all epochs in requested folder
    epochs = nirc2.search_epochs(data_dir=data_dir)
    
    # Full data container
    all_data = []
    
    # Compile data for each epoch
    for epoch, strehl_file in epochs.items():
        data = combine_strehl(strehl_file, data_types, check=False, 
                              test=test)
        # Add epoch name to columns
        data['epoch'] = epoch
        
        # Append to full data
        all_data.append(data)
    
    all_data = pd.concat(all_data, ignore_index=True)
    
    if save:
        all_data.to_csv(save, index=False)
    
    # Concatenate all data
    return all_data