# ~ Imports ...................................................................
# Global
import iceutils as ice
import matplotlib.pyplot as plt
import numpy as np
import os

# Relative
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

# Local
import sys
sys.path.append('..')
from constants import (ARCTICDEM_STACKFILE, COLUMBIA_EAST, COLUMBIA_MAIN, 
    FIG_ROOT, ICE_THICKNESS_MODEL_RELATIVE_STACKFILE, ICE_THICKNESS_MODEL_STACKFILE, ICE_THICKNESS_ROOT, ICE_THICKNESS_STACKFILE, POST, V_STACKFILE)
from utils import (create_relative_stack, get_timeseries_fit, interpolate_stack, load_timeseries, load_transects, smooth_timeseries, 
    snake_to_title)
from visualize import view_transect_signal, view_timeseries_fit
sys.path.append('analysis')
from bedrock import load_bedrock_data

# ~ Stack .....................................................................
# TODO: which way to resample? lower resolution or higher?
def resample_bedrock_to_arcticdem(bedrock_data, arcticdem_hdr, bedrock_hdr):
    return ice.interpolate_array(bedrock_data, bedrock_hdr, None, None, 
        ref_hdr=arcticdem_hdr, cval=np.nan)

def resample_arcticdem_to_bedrock(arcticdem_data, arcticdem_hdr, bedrock_hdr):
    return ice.interpolate_array(arcticdem_data, arcticdem_hdr, None, None, 
        ref_hdr=bedrock_hdr, cval=np.nan)

def create_ice_thickness_stack(bedrock, dems):
    v_stack = ice.Stack('..' + V_STACKFILE)
    Path('..' + ICE_THICKNESS_ROOT).mkdir(parents=True, exist_ok=True)

    # Interpolate Array
    warped_dems = interpolate_stack(dems._datasets['data'], dems.hdr, v_stack.hdr)
    warped_bedrock = ice.interpolate_array(bedrock.data, bedrock.hdr, None, None, ref_hdr=v_stack.hdr)

    init_data = []
    for dem in warped_dems:
        thickness = dem - warped_bedrock
        init_data.append(thickness)

    stack = ice.Stack('..' + ICE_THICKNESS_STACKFILE, mode='w', 
        init_tdec=dems.tdec, init_rasterinfo=v_stack.hdr)
    stack.fid.create_dataset('data', data=np.array(init_data))
    stack.fid.close()
    v_stack.fid.close()


# ~ Plots .....................................................................
def view_transect_mean_ice_thickness(bedrock, thickness, transects):
    mean = np.nanmean(thickness._datasets['data'], axis=0)

    fig, axs = plt.subplots(nrows=3, figsize=(9,9))

    plt.suptitle('Columbia Glacier Mean Ice Thickness', fontweight='bold')

    idx = 0
    for label in [COLUMBIA_EAST, COLUMBIA_MAIN, POST]:
        thickness_data = smooth_timeseries(load_timeseries(None, 
            transects[label], data=mean, hdr=thickness.hdr))
        bedrock_data = smooth_timeseries(load_timeseries(bedrock, 
            transects[label]))
        transect_dist = ice.compute_path_length(transects[label]['x'], 
            transects[label]['y'])


        # Plot Ice Thickness
        axs[idx].plot(transect_dist, thickness_data, label='Ice Thickness')
        axs[idx].set_ylabel('Mean thickness (m)')

        # Plot Bedrock Elevation
        vax = axs[idx].twinx()
        vax.plot(transect_dist, bedrock_data, 'r', label='Bedrock Elevation')
        vax.set_ylabel('Bed Elevation (m)')
        vax.legend()

        axs[idx].set_title(snake_to_title(label))
        axs[idx].set_xlabel('Upstream Distance (m)')

        idx += 1

    fig.set_tight_layout(True)

    save = '..' + FIG_ROOT + '/mean_ice_thickness_transect.jpg'
    plt.savefig(save, dpi=300)
    plt.show()


def view_mean_thickness_vs_bedrock(bedrock, thickness, transects, label):
    mean = np.nanmean(thickness._datasets['data'], axis=0)
    thickness_data = smooth_timeseries(load_timeseries(None, transects[label],
        data=mean, hdr=thickness.hdr))
    bedrock_data = smooth_timeseries(load_timeseries(bedrock, transects[label]))

    fig, axs = plt.subplots(figsize=(9,9))
    plt.suptitle(snake_to_title(label) + ' Bedrock vs Mean Ice Thickness',
        fontweight='bold')

    # Plot Ice Thickness
    axs.plot(thickness_data, bedrock_data)
    axs.set_ylabel('Mean thickness (m)')
    axs.set_xlabel('Bedrock Elevation (m)')

    fig.set_tight_layout(True)

    save = '..' + FIG_ROOT + '/mean_ice_thickness_vs_bedrock.jpg'
    plt.savefig(save, dpi=300)
    plt.show()

def build_design_matrix(t):
    """
    Function to create matrix for time series model. The model will consist of                           
       - quadratic trend
    """
    return np.column_stack((np.ones_like(t), t, t**2))

def model_ice_thickness_tseries(raw_stack):
    t = raw_stack.tdec
    v_stack = ice.Stack('..' + V_STACKFILE)

    # Interpolate Array
    data = raw_stack._datasets['data']

    # Create list of coordinates
    Ny, Nx = data[0].shape
    print(Ny, Nx)
    x = np.arange(0, Nx)
    y = np.arange(0, Ny)
    coords = product(y, x)

    print('Getting time series from Stack...')
    tseries = []
    for i, j in tqdm(coords):
        tseries.append(data[:, i, j])

    # Build smooth time
    st = ice.generateRegularTimeArray(min(v_stack.tdec), max(v_stack.tdec))
    model = np.full((len(st), Ny, Nx), np.nan)

    print('Building model...')
    get_model_fit = partial(get_timeseries_fit, t, st, build_design_matrix)
    with Pool(processes=None) as p:
        model = list(p.map(get_model_fit, tseries))

    model = np.reshape(model, (Ny, Nx, len(st)))
    model = np.array([model[:, :, i] for i in range(len(st))]) # NHW

    # Save model fit
    stack = ice.Stack('..' + ICE_THICKNESS_MODEL_STACKFILE, mode='w', init_tdec=st, init_rasterinfo=raw_stack.hdr)
    stack.fid.create_dataset('data', data=np.array(model))
    stack.fid.close()
    

def analyze_ice_thickness_model(ice_thickness_raw):
    print(ice_thickness_raw._datasets['data'].shape)
    ice_thickness_model= ice.Stack('..' + ICE_THICKNESS_MODEL_STACKFILE)
    ice_thickness_rel_model= ice.Stack('..' + ICE_THICKNESS_MODEL_RELATIVE_STACKFILE)
    view_transect_signal(ice_thickness_rel_model, '..' + FIG_ROOT + '/columbia_ice_thickness_model_transect_signal.jpg', clabel='Ice Thickness Change', title='Columbia Glacier Ice Thickness Change Along Centerline Transect')

    view_timeseries_fit(ice_thickness_model, ice_thickness_raw, '..' + FIG_ROOT + '/columbia_ice_thickness_timeseries_fit.jpg', idx=50, title='Ice Thickness Model Timeseries Fit', xlabel='Year', ylabel='Ice Thickness')
    ice_thickness_rel_model.fid.close()
    ice_thickness_model.fid.close()

def analyze_ice_thickness():
    bedrock = load_bedrock_data()
    transects = load_transects()
    dems = ice.Stack('..' + ARCTICDEM_STACKFILE)

    # Create Stack if it does not exist
    if not os.path.exists('..' + ICE_THICKNESS_STACKFILE):
        print('Creating ice thickness stack sampled to Joughin RasterInfo.')
        create_ice_thickness_stack(bedrock, dems)

    thickness = ice.Stack('..' + ICE_THICKNESS_STACKFILE)

    view_transect_mean_ice_thickness(bedrock, thickness, transects)
    view_mean_thickness_vs_bedrock(bedrock, thickness, transects, COLUMBIA_MAIN)

    if not os.path.exists('..' + ICE_THICKNESS_MODEL_STACKFILE):
        model_ice_thickness_tseries(thickness)

    if not os.path.exists('..' + ICE_THICKNESS_MODEL_RELATIVE_STACKFILE):
        create_relative_stack('..' + ICE_THICKNESS_MODEL_RELATIVE_STACKFILE, '..' + ICE_THICKNESS_MODEL_STACKFILE)

    analyze_ice_thickness_model(thickness)


if __name__ == '__main__':
    analyze_ice_thickness()