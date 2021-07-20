# Global
import iceutils as ice
import numpy as np
import os

# Relative
from functools import partial
from itertools import product
from multiprocessing import Pool

# Local
import sys
sys.path.append('..')
from constants import ARCTICDEM_STACKFILE, DEM_MODEL_RELATIVE_STACKFILE, DEM_MODEL_STACKFILE, FIG_ROOT, V_STACKFILE
from visualize import view_timeseries_fit, view_transect_signal
from utils import create_relative_stack, load_transects, get_transect_signal, get_timeseries_fit, interpolate_stack

def build_design_matrix(t):
    """
    Function to create matrix for time series model. The model will consist of                           
       - quadratic trend
    """
    return np.column_stack((np.ones_like(t), t, t**2))


def model_dem_tseries():
    dem_stack = ice.Stack('..' + ARCTICDEM_STACKFILE)
    v_stack = ice.Stack('..' + V_STACKFILE)

    t = dem_stack.tdec

    # Reduce array dimension
    data = interpolate_stack(dem_stack._datasets['data'], dem_stack.hdr, v_stack.hdr)

    # Create list of coordinates
    Ny, Nx = data[0].shape
    x = np.arange(0, Nx)
    y = np.arange(0, Ny)
    coords = product(y, x)

    print('Getting time series from Stack...')
    tseries = [data[:, i, j] for i, j in coords]

    # Build smooth time
    st = ice.generateRegularTimeArray(min(t), max(t))
    model = np.full((len(st), Ny, Nx), np.nan)

    print('Building model...')
    get_model_fit = partial(get_timeseries_fit, t, st, build_design_matrix)
    with Pool(processes=None) as p:
        model = list(p.map(get_model_fit, tseries))

    model = np.reshape(model, (Ny, Nx, len(st)))
    model = np.array([model[:, :, i] for i in range(len(st))]) # NHW

    # Save model fit
    stack = ice.Stack('..' + DEM_MODEL_STACKFILE, mode='w', init_tdec=st, init_rasterinfo=v_stack.hdr)
    stack.fid.create_dataset('data', data=np.array(model))
    stack.fid.close()

    dem_stack.fid.close()
    v_stack.fid.close()


def view_dem_model():
    dem_raw = ice.Stack('..' + ARCTICDEM_STACKFILE)
    dem_model= ice.Stack('..' + DEM_MODEL_STACKFILE)
    dem_rel_model= ice.Stack('..' + DEM_MODEL_RELATIVE_STACKFILE)
    view_transect_signal(dem_rel_model, '..' + FIG_ROOT + '/columbia_dem_model_transect_signal.jpg', clabel='Elevation Change', title='Columbia Glacier Elevation Change Along Centerline Transect')

    view_timeseries_fit(dem_model, dem_raw, '..' + FIG_ROOT + '/columbia_dem_timeseries_fit.jpg', idx=50, title='Elevation Model Timeseries Fit', xlabel='Year', ylabel='Elevation')
    dem_rel_model.fid.close()
    dem_model.fid.close()


def analyze_dem():
    if not os.path.exists('..' + DEM_MODEL_STACKFILE):
        model_dem_tseries()

    if not os.path.exists('..' + DEM_MODEL_RELATIVE_STACKFILE):
        create_relative_stack('..' + DEM_MODEL_RELATIVE_STACKFILE, '..' + DEM_MODEL_STACKFILE)

    view_dem_model()

    