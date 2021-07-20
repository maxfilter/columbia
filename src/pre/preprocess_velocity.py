""" Create velocity and error stacks from Dr. Ian Joughin's binary files.

Results in ex, ey, vx, vy, and v Stacks.
"""

# ~ Imports ...................................................................

# Absolute
import glob
import iceutils as ice
import numpy as np
import os
# https://github.com/fastice/utilities
import utilities as u

# Relative
from pathlib import Path

# Local
import sys
sys.path.append('..')
from constants import (BIN_ROOT, EXTENT_EPSG_3413, TIF_ROOT,
    VELOCITY_ROOT)
from utils import extent_to_proj_win

# ~ Functions .................................................................
def binary_to_tif(binary_root, tif_root, verbose=False):
    """ Convert Dr. Ian Joughin's binary files to tif files.
    
    Parameters
    ----------
    binary_root: str
        relative path to directory containing the binary trackfiles
    tif_root: str
        relative path to directory for storing generated Geotiff files
    verbose: bool
        write relevant output to stdout
    """

    if verbose:
        print('Converting byte-swapped binary files to GeoTiff.')
    
    Path(tif_root).mkdir(parents=True, exist_ok=True)

    # Create geoimage objects
    vel = u.geoimage(geoType='velocity')
    err = u.geoimage(geoType='error')

    for track_path in glob.glob(binary_root + '/*'):
        binary_fname = track_path + '/mosaicOffsets'

        # Read velocities in native format
        vel.readData(binary_fname)

        # Get date
        date = vel.parseMyMeta(binary_fname + '.meta')
        datestr = date.strftime('%Y%m%d')

        # Create sub-directory for storing tif
        Path(tif_root + '/' + datestr).mkdir(parents=True, exist_ok=True)
        tif_fname = tif_root + '/' + datestr + '/columbia_' + datestr # Store by date

        # Output as geotiff
        vel.writeMyTiff(tif_fname)

        # Errors
        err.readData(binary_fname)
        err.writeMyTiff(tif_fname)


def create_velocity_stack(data_root, dtype, stackfile, proj_win, verbose=False):
    """ Create a velocity stack
    
    Parameters
    ----------
    data_root:
        relative path to the directory containing the rasters
    dtype: str
        [v, vx, vy, ex, ey]
    stackfile: str
        file name for the generated Stack
    proj_win: list
        GDAL projection window
    verbose: bool
        write relevant output to stdout
    """

    if verbose:
        print('Generating', dtype, 'Stack.')

    # Stack header
    hdr = None

    # Set a lower bound for valid data
    lb = 0
    if dtype in ['vx', 'vy']:
        lb = -1e6 # m/yr

    # Get dates and data from files
    init_tdec, init_data = [], []
    for root, _, files in os.walk(data_root):
        if len(files) == 0:
            continue

        # Get filename (not elegant, but it works)
        rasterfile = ''
        for f in files:
            if dtype + '.tif' in f:
                rasterfile = f

        # Get date
        date = root.split('/')[-1]
        tdec = ice.datestr2tdec(yy=int(date[0:4]), mm=int(date[4:6]), dd=int(date[6:8]))
        init_tdec.append(tdec)

        # Get data
        raster = ice.Raster(rasterfile=root + '/' + rasterfile, projWin=proj_win)

        # Apply data bounds
        raster_data = np.array(raster.data)
        raster_data[raster_data < lb] = np.nan
        
        init_data.append(raster_data)

        # Store the first header
        if hdr is None:
            hdr = raster.hdr

    # Sort data before adding to Stack
    indsort = np.argsort(init_tdec)
    init_tdec = np.array(init_tdec)[indsort]
    init_data = np.array(init_data)[indsort]

    # Create Stack
    stack = ice.Stack(stackfile, mode='w', init_tdec=np.array(init_tdec), init_rasterinfo=hdr)
    stack.fid.create_dataset("data", data=np.array(init_data))
    stack.fid.close()


def create_velocity_stacks(data_root, stack_root, proj_win, verbose=False):
    """ Create a Stack for each type of data: ex, ey, v, vx, vy. 
    
    Parameters
    ----------
    data_root:
        relative path to the directory containing the rasters
    stack_root: str
        relative path to the directory to place generated Stacks
    proj_win: list
        GDAL projection window
    verbose: bool
        write relevant output to stdout
    """

    Path(stack_root).mkdir(parents=True, exist_ok=True)
    types = ['ex', 'ey', 'v', 'vx', 'vy']
    for dtype in types:
        stackfile = stack_root + '/columbia_' + dtype + '_stack.hdf5'
        create_velocity_stack(data_root, dtype, stackfile, proj_win, verbose=verbose)

def preprocess_velocity(verbose=True):
    """ Create velocity Stacks from binary files. """
    if verbose:
        print('Preprocessing velocity data ....................................................')

    bin_root = '..' + BIN_ROOT
    tif_root = '..' + TIF_ROOT
    stack_root = '..' + VELOCITY_ROOT
    proj_win = extent_to_proj_win(EXTENT_EPSG_3413)

    print(bin_root)

    binary_to_tif(bin_root, tif_root, verbose=verbose)
    create_velocity_stacks(tif_root, stack_root, proj_win, verbose=verbose)


if __name__ == '__main__':
    preprocess_velocity()

