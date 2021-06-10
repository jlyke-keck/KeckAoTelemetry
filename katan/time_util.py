### time_util.py : to handle time formatting changes between MJD, HST, and UTC
### Author : Emily Ramey
### Date: 5/26/21

# Preamble
import pandas as pd
import numpy as np
import time
import pytz as tz
from datetime import datetime, timezone
from astropy.time import Time, TimezoneInfo
from astropy import units as u, constants as c

hst = tz.timezone('US/Hawaii')

### Conversion / utility functions
def mjd_to_dt(mjds, zone='utc'):
    """ Converts Modified Julian Date(s) to HST or UTC date(s) """
    # Convert mjds -> astropy times -> datetimes
    dts = Time(mjds, format='mjd', scale='utc').to_datetime()
    # Convert to pandas
    dts = pd.to_datetime(dts).tz_localize(tz.utc)
    if zone=='hst':
        dts = dts.tz_convert("US/Hawaii")
    return dts

def table_to_mjd(table, columns, zone='utc'):
    """ Converts date and time columns to mjds """
    # Safety check for list of columns
    if not isinstance(columns, str):
        columns = [col for col in columns if col in table.columns]
    # Convert to datetimes
    dts = pd.to_datetime(table[columns], errors='coerce')
    
    if zone=='hst':# Convert to UTC
        dts = dts.dt.tz_localize(hst)
        dts = dts.dt.tz_convert(tz.utc)
    
    # Masked invalid values
    dts = np.ma.masked_array(dts, mask=dts.isnull())
    
    # Convert to astropy
    times = Time(dts, format='datetime', scale='utc')
    # return MJDs
    return np.ma.getdata(times.mjd)

def str_to_mjd(datestrings, fmt):
    """ Converts astropy-formatted date/time strings (in UTC) to MJD values """
    # Get astropy times
    times = Time(datestrings, format=fmt, scale='utc')
    # Convert to mjd
    return times.mjd

def mjd_to_yr(mjds):
    """ Converts MJD to Decimal Year """
    return Time(mjds, format='mjd', scale='utc').decimalyear 