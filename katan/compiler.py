### compiler.py : Compiler for all desired data (nirc2, weather, seeing, telemetry, temperature)
### Author : Emily Ramey
### Date : 6/3/21

from . import times
from . import strehl
from . import mkwc
from . import telemetry as telem
from . import temperature as temp
from . import templates

import numpy as np
import pandas as pd
import os

# Check dependencies
try:
    import yaml
except:
    raise ValueError("PyYAML not installed. Please install from https://anaconda.org/anaconda/pyyaml")

### For importing package files
### Note: may need to edit this later if there are import issues with Py<37
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

# All data types that can be compiled by this package
accept_labels = ['cfht', 'mass', 'dimm', 'masspro', 'k2AO', 'k2L4', 'k2ENV', 'telem']

default_parfile = 'keyword_defaults.yaml'

def load_default_parfile():
    """ Loads default parameter file and returns a dictionary """
    try: # Load from templates module
        file = pkg_resources.open_text(templates, default_parfile)
        default_params = yaml.load(file, Loader=yaml.FullLoader)
    except: # Raise error and request param file
        raise ValueError("Unable to load default parameters. Please specify a parameter file.")
    
    return default_params

def read_params(param_file):
    """ Reads parameters from the specified file or returns default parameters """
    if param_file is None: # load default
        params = load_default_parfile()
    elif isinstance(param_file, str) and os.path.exists(param_file): # Load user-specified
        try:
            params = yaml.load(param_file, Loader=yaml.FullLoader)
        except: # Unable to load
            raise ValueError(f"Failed to load {param_file}. Please check that PyYAML is installed \
            and that the file is formatted correctly.")
    else: # Invalid input
        raise ValueError(f"{param_file} is not a valid parameter file.")
    
    return params

# Shorthand / nicknames for data types
expand = {
    'temp': ['k2AO', 'k2L4', 'k2ENV'],
    'seeing': ['mass', 'dimm', 'masspro'],
    'weather': ['cfht'],
    'telemetry': ['telem'],
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

# Have files with acceptable columns for each data type so you can check inputs
# Accept column inputs to the data function and take optional arguments in combine_strehl

def data_func(dtype):
    """ 
    Returns the function to get data for the specified dtype 
    and whether data needs to be matched to nirc2 mjds
    """
    if dtype in ['cfht']+expand['seeing']: # MKWC
        return lambda files,mjds: mkwc.from_nirc2(mjds,dtype), True
    if dtype in expand['temp']: # Temperature
        return lambda files,mjds: temp.from_mjds(mjds,dtype), True
    if dtype=='telem': # Telemetry
        return lambda files,mjds: telem.from_nirc2(mjds,files), False

def change_cols(data, params):
    """
    Changes / filters columns according to the parameters passed.
    A False entry means the column will be omitted
    A True entry means the column will be included as-is
    A string entry means the column will be re-named
    """
    # Drop bad columns first
    good_cols = [col for col,val in params.items() if (val and col in data.columns)]
    new_data = data[good_cols].copy()
    
    # Re-map column names
    col_mapper = {col: new_col for col,new_col in params.items() if isinstance(new_col, str)}
    new_data.rename(columns=col_mapper, inplace=True)
    
    return new_data

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
def combine_strehl(strehl_file, data_types, file_paths=False, check=True, test=False,
                   param_file=None):
    """ 
    Combines and matches data from a certain Strehl file with other specified data types.
    NIRC2 files must be in the same directory as the Strehl file.
    """
    ### Check file paths parameter dict, load if yaml
    if not isinstance(file_paths, dict) and os.path.isfile(file_paths):
        with open(file_paths) as file:
            file_paths = yaml.load(file, loader=yaml.FullLoader)
    ### Add a catch for YAML not being installed if file specified?
    
    ### Sanitize data types
    if check:
        data_types = check_dtypes(data_types)
    
    ### Read parameter file
    params = read_params(param_file)
    
    ### Read in Strehl file
    nirc2_data = strehl.from_strehl(strehl_file)
    
    if test: # Take only first few files
        nirc2_data = nirc2_data.loc[:3]
    
    # Full data container
    all_data = [nirc2_data.reset_index(drop=True)]
    
    # Loop through and get data
    for dtype in data_types:
        get_data, match = data_func(dtype) # Data retrieval function
        # Get other data from strehl info
        other_data = get_data(nirc2_data.nirc2_file, nirc2_data.nirc2_mjd)
        
        # Change or omit selected columns
        other_data = change_cols(other_data, params[dtype])
        
        if match: # Needs to be matched
            if other_data.empty: # No data found
                other_data = pd.DataFrame(columns=other_data.columns, index=range(len(nirc2_data)))
            else: # Get indices of matched data
                idxs = match_data(nirc2_data.nirc2_mjd, other_data[dtype+'_mjd'])
                other_data = other_data.iloc[idxs]
        
        # Add to all data
        all_data.append(other_data.reset_index(drop=True))
    
    # Concatenate new data with nirc2
    return pd.concat(all_data, axis=1)