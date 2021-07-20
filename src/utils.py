""" General helper functions used throughout the project. """

# Global
import iceutils as ice
import pickle
import numpy as np

# Relative
from pathlib import Path
from scipy.signal import savgol_filter

# Local
from constants import TRANSECT_ROOT


def get_transect_signal(stack, transect):
    signal = []
    for data in stack._datasets['data']:
        signal.append(load_timeseries(None, transect, data=data, hdr=stack.hdr))
    return np.array(signal).T

def resample_landsat(landsat_root, infile, outfile, target_epsg=3413):
    """ Resample Landsat to a different projection """
    landsat = ice.Raster(rasterfile=landsat_root + infile)

    # Resample raster to new projection
    landsat_out = ice.warp(landsat, target_epsg=target_epsg, n_proc=5, cval=np.nan)

    # Save raster with new projection
    landsat_out.write_gdal(landsat_root + outfile, epsg=target_epsg)

def load_timeseries(raster, transect, data=None, hdr=None):
    # TODO : not timeseries!
    # Load data into memory
    if data is None:
        data = np.array(raster.data)

    timeseries = []
    for x, y in zip(transect['x'], transect['y']):
        if hdr is None:
            i, j = raster.hdr.xy_to_imagecoord(x, y)
        else:
            i, j = hdr.xy_to_imagecoord(x, y)
        timeseries.append(data[i, j])

    return np.array(timeseries)

def snake_to_title(text):
    """ Convert from snake_case to Title case. """
    return text.replace('_', ' ').title()

def smooth_timeseries(data, win=11, poly=2):
    """ Apply Savitzky Golay smoothing filter to a 1D timeseries. """
    return savgol_filter(data, win, poly)

def extent_to_proj_win(extent):
    return [extent[0], extent[3], extent[1], extent[2]]

def save_transects(transects):
    Path('..' + TRANSECT_ROOT).mkdir(parents=True, exist_ok=True)

    for transect in transects:
        save = '..' + TRANSECT_ROOT + '/' + transect['label'].lower().replace(" ", "_") + '.pickle'
        with open(save, 'wb') as f:
            pickle.dump(transect, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_transects():
    labels = ['columbia_main', 'columbia_east', 'post']
    transects = {}

    for label in labels:
        fname = '..' + TRANSECT_ROOT + '/' + label + '.pickle'
        with open(fname, 'rb') as f:
            transects[label] = pickle.load(f)

    return transects

def remove_outliers(t, d):
    # Compute statistics
    mean = np.mean(d)
    stdev = np.std(d)

    mask = np.abs(d - mean) < 3*stdev

    return t[mask], d[mask]

def get_timeseries_fit(t, st, build_design_matrix, d):
    dmask = np.isfinite(d)
    init_t = t[dmask]
    init_d = d[dmask]

    tmask = init_t > 2010.5
    init_t = init_t[tmask]
    init_d = init_d[tmask]

    t, d = remove_outliers(init_t, init_d)
    G = build_design_matrix(t)

    # Coeffs
    z = np.linalg.lstsq(G, d, rcond=1.0e-14)[0]

    # Get fit relative to start
    sG = build_design_matrix(st)
    return np.dot(sG, z)

def interpolate_stack(stack_data, stack_hdr, ref_hdr):
    data = []
    for im in stack_data:
        data.append(ice.interpolate_array(im, stack_hdr, None, None, ref_hdr=ref_hdr))
    return np.array(data)

def create_relative_stack(stackfile, ref_stackfile):
    """ Generate a new Stack where each raster in the data is relative to the first.

    Parameters
    ----------
    stackfile: str
        Name of new stackfile to create
    ref_stackfile: str
        Name of reference stackfile

    Post
    ----
    stackfile is a new stack where the first slice is a 0 array and all other data
    in stack is relative to this.
    """
    ref = ice.Stack(ref_stackfile)
    ref_data = np.array(ref._datasets['data'])

    rel_data = ref_data - ref_data[0]

    stack = ice.Stack(stackfile, mode='w', init_tdec=ref.tdec, init_rasterinfo=ref.hdr)
    stack.fid.create_dataset('data', data=rel_data)

    ref.fid.close()
    stack.fid.close()