# TODO
# See Ch 8.2.1 in Physics of Glaciers

import iceutils as ice
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys

from utils import load_timeseries, load_transects, smooth_timeseries, snake_to_title
sys.path.append('..')
from constants import COLUMBIA_EAST, COLUMBIA_MAIN, DEM_MODEL_STACKFILE, FIG_ROOT, GRAVITATIONAL_ACCELERATION, ICE_DENSITY, ICE_THICKNESS_MODEL_STACKFILE, POST

def compute_driving_stress(ice_thickness, elevation):
    """
    Parameters
    ----------
    ice_thickness : array_like
        1d array of ice thickness values along transect
    elevation : array_like
        1d array of elevation values along transect
    """
    win, poly = 11, 2
    ice_thickness = savgol_filter(ice_thickness, win, poly)
    # Negative sign because elevation data is positive in upstream direction, not downstream
    ds_dx = - savgol_filter(elevation, win, poly, deriv=1)
    return - ICE_DENSITY * GRAVITATIONAL_ACCELERATION * np.multiply(ice_thickness, ds_dx)

def get_driving_stress_timeseries(dem_model, thickness_model, transect):
    t_d = []
    for i in range(len(dem_model._datasets['data'])):
        elevation_data = dem_model._datasets['data'][i]
        ice_thickness_data = thickness_model._datasets['data'][i]

        elevation_tseries = load_timeseries(None, transect, data=elevation_data, hdr=dem_model.hdr)
        thickness_tseries = load_timeseries(None, transect, data=ice_thickness_data, hdr=thickness_model.hdr)

        t_d.append(compute_driving_stress(thickness_tseries, elevation_tseries))

    return t_d
        
def animate_driving_stress_timeseries(driving_stress_dict, transects, tdec, save, fps=100):

    fig, axs = plt.subplots(nrows=3, figsize=(9,9))
    plots = {}
    dist = {}

    for idx, label in enumerate([COLUMBIA_MAIN, COLUMBIA_EAST, POST]):
        transect = transects[label]
        dist[label] = ice.compute_path_length(transect['x'], transect['y'])
        plots[label] = axs[idx].plot(dist[label], driving_stress_dict[label][0])[0]

        axs[idx].set_title(snake_to_title(label))
        axs[idx].set_xlabel('Upstream Distance')
        axs[idx].set_ylabel('Driving Stress')

    # Create Title
    title = 'Driving Stress Along Transect '
    datestr = ice.tdec2datestr(tdec[0])
    tx = fig.suptitle(title + ' ' + datestr, fontweight='bold')

    # Update frame
    def animate(i):
        for label in [COLUMBIA_MAIN, COLUMBIA_EAST, POST]:
            plots[label].set_data(dist[label], driving_stress_dict[label][i])
        datestr = ice.tdec2datestr(tdec[i])
        tx.set_text(title + ' ' + datestr)

    fig.set_tight_layout(True)
    interval = 1000/fps
    anim = animation.FuncAnimation(fig, animate, interval=interval, frames=len(tdec), repeat=True)
    anim.save(save, dpi=300)

def view_driving_stress_timeseries(driving_stress_dict, elevation_model, transects, save, idx=500):
    
    fig, axs = plt.subplots(nrows=3, figsize=(9,9))
    elevation = elevation_model._datasets['data'][idx]

    plt.suptitle('Driving Stress Along Transects on ' + ice.tdec2datestr(elevation_model.tdec[idx]))

    for i, label in enumerate([COLUMBIA_MAIN, COLUMBIA_EAST, POST]):
        transect = transects[label]
        dist = ice.compute_path_length(transect['x'], transect['y'])
        axs[i].plot(dist, driving_stress_dict[label][idx], label='Driving Stress')

        # Plot mean velocity
        elevation_transect = smooth_timeseries(load_timeseries(None, transect, data=elevation, hdr=elevation_model.hdr))
        eax = axs[i].twinx()
        eax.plot(dist, elevation_transect, 'r', label='Elevation')
        eax.set_ylabel('Elevation (m)')
        eax.legend()
        
        axs[i].set_title(snake_to_title(label))
        axs[i].set_xlabel('Upstream Distance')
        axs[i].set_ylabel('Driving Stress')

    fig.set_tight_layout(True)
    plt.savefig(save, dpi=300)

def analyze_driving_stress():
    transects = load_transects()
    dem_model = ice.Stack('..' + DEM_MODEL_STACKFILE)
    thickness_model = ice.Stack('..' + ICE_THICKNESS_MODEL_STACKFILE)

    # Compute driving stress for each transect
    driving_stress_dict = {}
    for label in [COLUMBIA_MAIN, COLUMBIA_EAST, POST]:
        transect = transects[label]
        driving_stress_dict[label] = get_driving_stress_timeseries(dem_model, thickness_model, transect)

    # animate_driving_stress_timeseries(driving_stress_dict, transects, dem_model.tdec, '..' + FIG_ROOT + '/driving_stress_timeseries.mp4')
    view_driving_stress_timeseries(driving_stress_dict, dem_model, transects, '..' + FIG_ROOT + '/driving_stress_transect.jpg')

