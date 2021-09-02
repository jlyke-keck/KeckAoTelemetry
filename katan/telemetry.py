### telem_util.py: For code that interacts with Keck AO Telemetry files
### Author: Emily Ramey
### Date: 11/18/2020

import numpy as np
import pandas as pd
import glob
from scipy.io import readsav

from . import times
from . import templates

### For importing package files
### May need to edit this later if there are import issues with Py<37
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

### Path to telemetry files [EDIT THIS]
# telem_dir = '/g/lu/data/keck_telemetry/'

# Sub-aperture maps
wfs_file = "sub_ap_map.txt"
act_file = "act.txt"

# regex to match telem filenumbers
filenum_match = ".*c(\d+).fits"
filename_match = "{}{}/**/n?{}*LGS*.sav"

TIME_DELTA = 0.001 # About 100 seconds in mjd

RESID_CUTOFF = 349
TT_IDXS = [349,350]
DEFOCUS = 351
LAMBDA = 2.1 # microns

cols = ['telem_file',
        'telem_mjd',
        'TT_mean', 
        'TT_std',
        'DM_mean',
        'DM_std',
        'rmswfe_mean',
        'rmswfe_std',
        'strehl_telem']

def read_map(filename):
    """ Reads a map of actuators or sub-apertures from a file """
    file = pkg_resources.open_text(templates, filename)
    return pd.read_csv(file, delim_whitespace=True, header=None).to_numpy()

def get_times(telem_data, start=None):
    """ Pulls timestamps from a telemetry file in seconds from start """
    if start is None:
        start = telem_data.a.timestamp[0][0]
    
    return (telem_data.a.timestamp[0]-start)/1e7

def resid_mask(ints, wfs_map=read_map(wfs_file), act_map=read_map(act_file), num_aps=236):
    """
    Returns the locations of the valid actuators in the actuator array
    resids: Nx349 residual wavefront array (microns)
    ints: Nx304 intensity array (any units)
    N: Number of timestamps
    """
    # Check inputs
    N = ints.shape[0] # Num timestamps
    
    # Aggregate intensities over all timestamps
    med_ints = np.median(ints, axis=0)
    
    # Fill WFS map with aggregated intensities
    int_map = wfs_map.copy()
    int_map[np.where(int_map==1)] = med_ints
    
    # Find lenslets with greatest intensity
    idxs = np.flip(np.argsort(int_map, axis=None))[:num_aps] # flat idxs of sort
    idxs = np.unravel_index(idxs, wfs_map.shape) # 2D idxs of sort
    
    # Mask for good sub-ap values
    good_aps = np.zeros(wfs_map.shape, dtype=int)
    good_aps[idxs] = 1
    good_aps = good_aps * wfs_map # Just in case
    
    # Mask for good actuator values
    good_acts = np.pad(good_aps, ((1,1),(1,1)))
    good_acts = (good_acts[1:,1:] | good_acts[1:,:-1] 
                 | good_acts[:-1,:-1] | good_acts[:-1,1:]) * act_map
    
    return good_acts

def tt2um(tt_as):
    """ Calculates TT residuals in microns from tt_x and tt_y in arcsec """
    D = 10.5e6 # telescope diameter in microns
    tt = tt_as*4.8e-6 # TT error in radians
    tt_err = np.sqrt(D**2 / 12 * (tt[:,0]**2 + tt[:,1]**2))
    return tt_err

def rms_acts(act_resids, ints):
    """ Clips bad actuators and averages for each timestamp """
    N = act_resids.shape[0] # num timestamps
    
    # Read in actuator map
    act_map = read_map(act_file)
    
    # Get good actuator mask from intensities
    mask = resid_mask(ints)
    flat_map = ~mask[np.where(act_map==1)].astype(bool) # flatten
    flat_map = np.tile(flat_map, (N,1))
    
    # Mask the bad actuators
    good_acts = np.ma.masked_array(act_resids, flat_map)
    # Average resids for each timestep
    act_rms = np.sqrt((good_acts**2).mean(axis=1) - good_acts.mean(axis=1)**2)
    
    return act_rms.compressed()

########################################
######### Processing Files #############
########################################

def get_mjd(telem):
    """ 
    Validates a telemetry file against an MJD value.
    telem: structure returned from readsav
    returns: MJD of telemetry file start
    """
    # Get timestamp
    tstamp = telem.tstamp_str_start.decode('utf-8')
    # Convert to MJD
    telem_mjd = times.str_to_mjd(tstamp, fmt='isot')
    
    # Returns telem mjd
    return telem_mjd

def extract_telem(file, data, idx, check_mjd=None):
    """ 
    Extracts telemetry values from a file to a dataframe.
    file: file path to telemetry, as string
    data: dataframe to load new telemetry into
    idx: index in dataframe receive new telemetry data
    check_mjd: MJD to match with telemetry file (no matching if none)
    """
    # Read IDL file
    telem = readsav(file)
    
    # Make sure MJD matches
    telem_mjd = get_mjd(telem)
    if check_mjd is not None:
        delta = np.abs(telem_mjd-check_mjd)
        if delta > TIME_DELTA:
            return False
    
    # Get residuals and intensities
    act_resids = telem.a.residualwavefront[0][:, :RESID_CUTOFF]
    tt_resids = telem.a.residualwavefront[0][:,TT_IDXS]
    ints = telem.a.subapintensity[0] # Sub-aperture intensities
    
    # Convert TT resids to microns
    tt_microns = tt2um(tt_resids)
    # Get RMS resids from the actuator array
    act_rms = rms_acts(act_resids, ints)
    
    # Total RMS Wavefront Error
    rmswfe = np.sqrt(tt_microns**2 + act_rms**2)
    
    # Strehl calculation
    strehl = np.exp(-(2*np.pi*rmswfe/LAMBDA)**2)
    
    # Assemble aggregate data
    data.loc[idx, cols] = [
        file,
        telem_mjd,
        np.mean(tt_microns),
        np.std(tt_microns),
        np.mean(act_rms),
        np.std(act_rms),
        np.mean(rmswfe),
        np.std(rmswfe),
        np.mean(strehl),
    ]
    
    return True

def from_nirc2(mjds, nirc2_filenames, telem_dir):
    """ 
    Gets a table of telemetry information from a set of mjds and NIRC2 filenames.
    mjds: NIRC2 mjds to match to telemetry files
    nirc2_filenames: NIRC2 filenames to match to telemetry files
    telem_dir: path to directory containing telemetry files
    returns: dataframe of telemetry, in same order as NIRC2 MJDs passed
    """
    N = len(mjds) # number of data points
    # Get file numbers
    filenums = nirc2_filenames.str.extract(filenum_match, expand=False)
    filenums = filenums.str[1:] # First digit doesn't always match
    
    # Get datestrings
    dts = times.mjd_to_dt(mjds) # HST or UTC???
    datestrings = dts.strftime("%Y%m%d") # e.g. 20170826
    
    # Set up dataframe
    data = pd.DataFrame(columns=cols, index=range(N))
    
    # Find telemetry for each file
    for i in range(N):
        # Get filename and number
        fn, ds, mjd = filenums[i], datestrings[i], mjds[i]
        # Search for correct file
        file_pat = filename_match.format(telem_dir, ds, fn)
        all_files = glob.glob(file_pat, recursive=True)
        
        # Extract the first file that matches the MJD to data
        for file in all_files:
            success = extract_telem(file, data, i, check_mjd=mjd)
            if success: break
        
    return data
    