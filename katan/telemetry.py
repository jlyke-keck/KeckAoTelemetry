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

filenum_match = ".*c(\d+).fits" # regex to match telem filenumbers
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
    Return the locations of the valid actuators in the actuator array
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

# def wfs_error2(resids, ints, wfs_map=read_map(wfs_file), act_map=read_map(act_file)):
#     """ Calculates the variance in the wavefront (microns^2) produced by the WFS """
#     # Get residual mask for brightest sub-aps
#     rmask =  resid_mask(resids, ints)
    
#     # Square residuals
#     sig2_resid = np.mean(resids**2, axis=0)
#     # Load into actuator grid
#     resid_grid = act_map.astype(float)
#     resid_grid[np.where(resid_grid==1)] = sig2_resid
    
#     # Mask out actuators next to dimmer apertures
#     sig2_masked = resid_grid * rmask
#     # Sum values and return
#     return np.sum(sig2_masked)

# def tt_error2(tt_resids): ### TODO: fix this so it's not unitless
#     """ Calculate mean tip-tilt residual variance in microns """
#     wvln = 0.658 # wavelength (microns)
#     D = 10.5 * 1e6 # telescope diameter (microns)
    
#     # Magnitude of residuals
#     sig_alpha2 = (tt_resids[:,0] + tt_resids[:,1])**2 # x + y TT residual variance
#     sig_w2 = (np.pi*D/wvln/2.0)**2 * sig_alpha2 # TT resids in microns^2
    
#     return np.mean(sig_w2)

# def total_WFE(telem_data):
#     resids = telem_data.a.residualwavefront[0][:, :RESID_CUTOFF] * 0.6 # WFS resids, microns, Nx349
#     ints = telem_data.a.subapintensity[0] # Sub-ap intensities, Nx304
#     tt_resids = telem_data.a.residualwavefront[0][:, TT_IDXS] * np.pi / (180*3600) # TT resids, radians, Nx2
#     defocus = [0] #telem_data.a.residualwavefront[0][:, DEFOCUS] # Defocus resids, microns, Nx1
    
#     return wfs_error2(resids, ints) + tt_error2(tt_resids) + np.mean(defocus**2)

# def mask_residuals_old(telem_data, num_aps=236, wfs_file=wfs_file, act_file=act_file):
#     """ Really freakin complicated array logic to mask out all invalid actuators """
#     # Get data
#     resids = telem_data.a.residualwavefront[0][:, :349]*0.6 # microns
#     intensities = telem_data.a.subapintensity[0] # ADU
#     N = intensities.shape[0]
#     # Get hardware maps
#     wfs_map = read_map(wfs_file)
#     act_map = read_map(act_file)
    
#     # Get X and Y values of sub-aps and replicate them for all timestamps
#     wfs_x, wfs_y = np.where(wfs_map==1)
#     wfs_x, wfs_y = np.tile(wfs_x, (N,1)), np.tile(wfs_y, (N,1))
    
#     # Get valid indices for each timestep
#     idxs = np.flip(np.argsort(intensities, axis=1), axis=1)[:,:num_aps]
#     valid_x = np.take_along_axis(wfs_x, idxs, axis=1)
#     valid_y = np.take_along_axis(wfs_y, idxs, axis=1)
#     valid_z = np.tile(np.arange(N), (num_aps,1)).T
    
#     # Put 1s at each valid index
#     valid_saps = np.zeros((N, 20, 20), int)
#     valid_saps[valid_z, valid_x, valid_y] = 1 # TODO: flip this back
#     # Pad each sheet (timestamp) with zeros at the edges
#     check = valid_saps.reshape(N, 20*20).sum(axis=1)
#     if any(check!=236):
#         print("Shape mismatch in valid sub-ap array")
#     valid_saps = np.pad(valid_saps, ((0,0),(1,1),(1,1)))
    
#     # Get (potentially)valid actuators for sub-aps
#     valid_acts = (valid_saps[:,1:,1:]|valid_saps[:,1:,:-1]|
#                   valid_saps[:,:-1,:-1]|valid_saps[:,:-1,1:]) # 4 corners of sub-aps
#     # Multiply by actuator map to remove any non-actuator positions
#     valid_acts = valid_acts * np.tile(act_map, (N,1,1))
    
#     # Get number of valid actuators in each frame (can vary due to edge cases)
#     valid_act_nums = valid_acts.reshape(N,21*21).sum(axis=1)
    
#     # Map residuals to actuator positions
#     resid_vals = np.tile(act_map, (N,1,1)).astype(float)
#     resid_vals[np.where(resid_vals==1)] = resids.flatten()
    
#     # Mask out invalid actuators
#     valid_acts = valid_acts * resid_vals
#     rms_resids = valid_acts.reshape(N, 21*21)
    
#     # Take the RMS residual for each frame
#     rms_resids = np.sqrt((rms_resids**2).sum(axis=1)/valid_act_nums)
    
#     return rms_resids

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

# Do some telemetry files need to be cropped around the observation?

def get_mjd(telem):
    """ Validates a telemetry file against an MJD value """
    # Get timestamp
    tstamp = telem.tstamp_str_start.decode('utf-8')
    # Convert to MJD
    telem_mjd = times.str_to_mjd(tstamp, fmt='isot')
    
    # Returns telem mjd
    return telem_mjd

def extract_telem(file, data, idx, check_mjd=None):
    """ Extracts telemetry values from a file to a dataframe """
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
    """ Gets a table of telemetry information from a set of mjds and file numbers """
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
    