
# Global
import iceutils as ice
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Relative
from functools import partial
from itertools import product
from multiprocessing import Pool
from scipy.optimize import curve_fit

# Local
import sys
sys.path.append('..')
from constants import (COLUMBIA_EAST, COLUMBIA_MAIN, FIG_ROOT, LANDSAT_RASTERFILE, POST, SEASONAL_VELOCITY_AMPLITUDE_RASTERFILE, SEASONAL_VELOCITY_MODEL_STACKFILE, SEASONAL_VELOCITY_PHASE_RASTERFILE, SEASONAL_VELOCITY_RAW_STACKFILE, SECULAR_VELOCITY_MODEL_RELATIVE_STACKFILE, SECULAR_VELOCITY_MODEL_STACKFILE, SECULAR_VELOCITY_RAW_STACKFILE, V_STACKFILE)
from utils import get_transect_signal, load_timeseries, load_transects, smooth_timeseries, snake_to_title
from visualize import view_data

# ~ Timeseries ................................................................
def _get_tseries(data, coord):
    row, col = coord
    return data[:, row, col]

def _invert_tseries(solver, model, tseries):
    tseries[tseries < 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _, m, _ = solver.invert(model.G, tseries)

    if m is None:
        return np.full(len(model.G[0]), np.nan)
    else:
        return m
# ~ Seasonal ..................................................................
def compute_seasonal_amp_phase(m, model):
    """ From iceutils model.py """
    try:
        m1, m2 = m[model.iseasonal][-2:]
        phs = np.arctan2(m2, m1) * 182.5/np.pi
        amp = np.sqrt(m1**2 + m2**2)
        if phs < 0.0:
            phs += 365.0
    except ValueError:
        phs, amp = None, None
    return amp, phs

def _get_seasonal_tseries(model, G, m):
    seasonal = {
        'tseries' : np.full(len(G), np.nan),
        'phase' : np.nan,
        'amp' : np.nan
    }
    if not np.isnan(m[0]):
        seasonal['tseries'] = np.dot(G[:,model.iseasonal], m[model.iseasonal])
        seasonal['amp'], seasonal['phase'] = compute_seasonal_amp_phase(m, model)

    return seasonal

def _get_seasonal_data(model, m):
    if np.isnan(m[0]):
        return np.full(len(model.G), np.nan)
    else:
        return np.dot(model.G[:,model.iseasonal], m[model.iseasonal])

def model_seasonal_tseries(stack, model, stdec, sG, z, min_amplitude=100):
    """ Run the inversion and store seasonal data """
    Ny, Nx = stack._datasets['data'][0].shape

    print('Getting seasonal data...')
    get_seasonal_tseries = partial(_get_seasonal_tseries, model, sG)
    with Pool(processes=None) as p:
        seasonal = list(p.map(get_seasonal_tseries, z))

    # Amplitude
    print('Saving amplitude as raster...')
    seasonal_amp = [result['amp'] for result in seasonal]
    seasonal_amp = np.reshape(seasonal_amp, (Ny, Nx))
    amp_raster = ice.Raster(data=seasonal_amp, hdr=stack.hdr)
    amp_raster.write_gdal('..' + SEASONAL_VELOCITY_AMPLITUDE_RASTERFILE, epsg=stack.hdr.epsg)
    view_data(seasonal_amp, stack.hdr, 
        '..' + FIG_ROOT + '/seasonal_amplitude_data.jpg',
        ref = '..' + LANDSAT_RASTERFILE,
        alpha = 0.7,
        title='Columbia Glacier Seasonal Amplitude',
        clabel='Amplitude (m)',
        clim=[0, 1500])

    print('Saving phase as raster...')
    seasonal_phase = [result['phase'] for result in seasonal]
    seasonal_phase = np.reshape(seasonal_phase, (Ny, Nx))
    phase_raster = ice.Raster(data=seasonal_phase, hdr=stack.hdr)
    phase_raster.write_gdal('..' + SEASONAL_VELOCITY_PHASE_RASTERFILE, epsg=stack.hdr.epsg)
    seasonal_phase[seasonal_phase > 100] = np.nan
    view_data(seasonal_phase, stack.hdr, 
        '..' + FIG_ROOT + '/seasonal_phase_data.jpg',
        ref = '..' + LANDSAT_RASTERFILE,
        alpha = 0.7,
        title='Columbia Glacier Seasonal Phase',
        clabel='Phase (days)',
        clim=[0, 100])

    # Seasonal variation
    seasonal_tseries = [result['tseries'] for result in seasonal]
    seasonal_tseries = np.reshape(seasonal_tseries, (Ny, Nx, len(sG))) # HWN
    seasonal_tseries = np.array([seasonal_tseries[:, :, i] for i in range(len(sG))]) # NHW

    # Mask out data with small amplitudes
    for t in seasonal_tseries:
        t[seasonal_amp < min_amplitude] = np.nan

    print('Saving seasonal data to Stack...')
    stack = ice.Stack('..' + SEASONAL_VELOCITY_MODEL_STACKFILE, mode='w', init_tdec=stdec,
        init_rasterinfo=stack.hdr)
    stack.fid.create_dataset('data', data=np.array(seasonal_tseries))
    stack.fid.close()

# ~ Secular ...................................................................
def _get_secular_tseries(model, G, m): 
    secular = {
        'tseries' : np.full(len(G), np.nan)
    }
    if not np.isnan(m[0]):
        secular['tseries'] = np.dot(G[:,model.isecular], m[model.isecular])

    return secular

def _get_secular_data(model, m):
    if np.isnan(m[0]):
        return np.full(len(model.G), np.nan)
    else:
        return np.dot(model.G[:,model.isecular], m[model.isecular])


def model_secular_tseries(stack, model, stdec, sG, z):
    Ny, Nx = stack._datasets['data'][0].shape

    print('Getting model secular data...')
    get_secular = partial(_get_secular_tseries, model, sG)
    with Pool(processes=None) as p:
        secular = list(p.map(get_secular, z))

    # Secular variation
    secular_tseries = [result['tseries'] for result in secular]
    secular_tseries = np.reshape(secular_tseries, (Ny, Nx, len(sG))) # HWN
    secular_tseries = np.array([secular_tseries[:, :, i] for i in range(len(sG))]) # NHW

    print('Saving secular data to Stack...')
    stack = ice.Stack('..' + SECULAR_VELOCITY_MODEL_STACKFILE, mode='w', init_tdec=stdec,
        init_rasterinfo=stack.hdr)
    stack.fid.create_dataset('data', data=np.array(secular_tseries))
    stack.fid.close()

    # Save another version that is relative to start
    ref = np.array(secular_tseries[0])
    secular_tseries_relative = secular_tseries - ref

    stack = ice.Stack('..' + SECULAR_VELOCITY_MODEL_RELATIVE_STACKFILE, mode='w', init_tdec=stdec,
        init_rasterinfo=stack.hdr)
    stack.fid.create_dataset('data', data=np.array(secular_tseries_relative))
    stack.fid.close()


# ~ Raw Fit ....................................................................
def save_raw_data_fit(stack, model, z):
    print('Saving data of unsmoothed model fit...')
    Ny, Nx = stack._datasets['data'][0].shape

    get_seasonal_data = partial(_get_seasonal_data, model)
    get_secular_data = partial(_get_secular_data, model)
    with Pool(processes=None) as p:
        seasonal_data = list(p.map(get_seasonal_data, z))
        secular_data = list(p.map(get_secular_data, z))

    seasonal_data = np.reshape(seasonal_data, (Ny, Nx, len(model.G)))
    seasonal_data = np.array([seasonal_data[:, :, i] for i in range(len(model.G))]) # NHW

    secular_data = np.reshape(secular_data, (Ny, Nx, len(model.G)))
    secular_data = np.array([secular_data[:, :, i] for i in range(len(model.G))]) # NHW

    # Subtract long term from initial data to get seasonal data
    seasonal_data_raw = np.subtract(stack._datasets['data'], secular_data)
    # Subtract seasonal from raw data to get secular data
    secular_data_raw = np.subtract(stack._datasets['data'], seasonal_data)

    seasonal_stack = ice.Stack('..' + SEASONAL_VELOCITY_RAW_STACKFILE, mode='w', init_tdec=stack.tdec,
        init_rasterinfo=stack.hdr)
    seasonal_stack.fid.create_dataset('data', data=np.array(seasonal_data_raw))
    seasonal_stack.fid.close()

    secular_stack = ice.Stack('..' + SECULAR_VELOCITY_RAW_STACKFILE, mode='w', init_tdec=stack.tdec,
        init_rasterinfo=stack.hdr)
    secular_stack.fid.create_dataset('data', data=np.array(secular_data_raw))
    secular_stack.fid.close()


# ~ Setup .....................................................................
def model_velocity_tseries():
    # Load Stack
    stack = ice.Stack('..' + V_STACKFILE)
    data = np.array(stack._datasets['data'])

    # Create list of coordinates
    Ny, Nx = data[0].shape
    x = np.arange(0, Nx)
    y = np.arange(0, Ny)
    coords = product(y, x)

    print('Getting time series from Stack...')
    get_tseries = partial(_get_tseries, np.array(data))
    with Pool(processes=None) as p:
        stack_tseries = list(p.map(get_tseries, coords))

    # Get t for sample data
    t = ice.tdec2datestr(stack.tdec, returndate=True)

    # Build model
    model = ice.tseries.build_temporal_model(t, poly=2, isplines=[])

    # Build smooth G
    stdec = ice.generateRegularTimeArray(min(stack.tdec), max(stack.tdec))
    st = ice.tdec2datestr(stdec, returndate=True)
    sG = ice.tseries.build_temporal_model(st, poly=2, isplines=[]).G

    # Set up solver
    solver = ice.tseries.select_solver('ridge', reg_indices=model.itransient,
        penalty=0.25)

    # Time series inversion at a point
    invert_tseries = partial(_invert_tseries, solver, model)

    # Calculate coefficients
    print('Calculating coefficients for fit at each point...')
    with Pool(processes=None) as p:
        z = list(p.map(invert_tseries, stack_tseries))

    # Model seasonal and secular components
    model_seasonal_tseries(stack, model, stdec, sG, z)
    model_secular_tseries(stack, model, stdec, sG, z)

    # Store raw data fit
    save_raw_data_fit(stack, model, z)

# ~ Plots .....................................................................
def view_transect_seasonal_secular_signal(seasonal_model, secular_model_rel, transect, label, dist, cmap='Spectral_r'):
    seasonal_data = get_transect_signal(seasonal_model, transect)
    secular_data = get_transect_signal(secular_model_rel, transect)
    extent = [seasonal_model.tdec[0], seasonal_model.tdec[-1], 0, dist[-1]]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 9))

    plt.suptitle(snake_to_title(label), fontweight='bold')

    # Get Seasonal clims
    seasonal_min = np.nanmin(secular_data)
    seasonal_max = np.nanmax(seasonal_data)
    seasonal_cval = min(abs(seasonal_min), abs(seasonal_max))
    if seasonal_cval < 50: 
        seasonal_cval = max(abs(seasonal_min), abs(seasonal_max))
    seasonal_clim = [-seasonal_cval, seasonal_cval]

    # Plot seasonal signal
    seasonal_im = axs[0].imshow(seasonal_data, origin='lower', aspect='auto',
        cmap=cmap, extent=extent, clim=seasonal_clim)
    axs[0].set_title("Seasonal Signal")
    seasonal_cbar = plt.colorbar(seasonal_im, ax=axs[0])
    seasonal_cbar.set_label('Seasonal variation (m/yr)')

    # Get secular clims
    secular_min = np.nanmin(secular_data)
    secular_max = np.nanmax(secular_data)
    if secular_max <= 0:
        secular_clim = [secular_min, 0]
    elif secular_min >= -50:
        secular_clim = [-50, secular_max]
    else:
        secular_cval = min(abs(secular_min), secular_max)
        secular_clim = [-secular_cval, secular_cval]

    # Plot secular signal
    secular_im = axs[1].imshow(secular_data, origin='lower', aspect='auto',
        cmap=cmap, extent=extent, clim=secular_clim)
    axs[1].set_title("Secular Signal")
    secular_cbar = plt.colorbar(secular_im, ax=axs[1])
    secular_cbar.set_label('Long term variation (m/yr)')

    # Add axes labels
    for ax in axs:
        ax.set_ylabel('Upstream distance (m)')
        ax.set_xlabel('Year')

    fig.set_tight_layout(True)

    plt.savefig('..' + FIG_ROOT + '/' + label + '_transect_signal.jpg', dpi=300)


def view_transect_seasonal_amp_phase(transect, dist, label, show=False):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9,9))

    # Mean
    v_stack = ice.Stack('..' + V_STACKFILE)
    mean_vel = np.nanmean(v_stack._datasets['data'], axis=0)
    amp = ice.Raster(rasterfile='..' + SEASONAL_VELOCITY_AMPLITUDE_RASTERFILE)
    phase = ice.Raster(rasterfile='..' + SEASONAL_VELOCITY_PHASE_RASTERFILE)

    mean_vel_transect = smooth_timeseries(load_timeseries(None, transect, data=mean_vel, hdr=v_stack.hdr))
    amp_transect = smooth_timeseries(load_timeseries(None, transect, data=amp.data, hdr=v_stack.hdr))
    phase_transect = smooth_timeseries(load_timeseries(None, transect, data=phase.data, hdr=v_stack.hdr))

    plt.suptitle(snake_to_title(label) + ' Amplitude and Phase of Seasonal Variations along Centerline Transect', fontweight='bold')

    # Plot amplitude
    axs[0].plot(dist, amp_transect, label='Amplitude')
    axs[0].set_title('Amplitude and Mean Velocity')
    axs[0].set_ylabel('Amplitude (m/yr)')

    # Plot mean velocity
    vax = axs[0].twinx()
    vax.plot(dist[0:len(amp_transect)-1], mean_vel_transect[0:len(amp_transect)-1], 'r', label='Mean Velocity')
    vax.set_ylabel('Mean velocity (m/yr)')
    vax.legend()

    # Plot phase
    axs[1].plot(dist, phase_transect, label='Phase')
    axs[1].set_title('Phase')
    axs[1].set_xlabel('Upstream distance (m)')
    axs[1].set_ylabel('Days')

    fig.set_tight_layout(True)
    plt.savefig('..' + FIG_ROOT + '/' + label + '_transect_amp_phase.jpg', dpi=300)
    if show:
        plt.show()

def view_amplitude_model(transects, save, show=False):
    v_stack = ice.Stack('..' + V_STACKFILE)
    amp = ice.Raster(rasterfile='..' + SEASONAL_VELOCITY_AMPLITUDE_RASTERFILE)

    fig, axs = plt.subplots(nrows=3, figsize=(9,9))

    plt.suptitle('Seasonal Amplitude Models', fontweight='bold')

    for i, label in enumerate([COLUMBIA_MAIN, COLUMBIA_EAST, POST]):
        transect = transects[label]
        dist = ice.compute_path_length(transect['x'], transect['y'])
        amp_transect = smooth_timeseries(load_timeseries(None, transect, data=amp.data, hdr=v_stack.hdr))

        # Plot raw data
        axs[i].plot(dist, amp_transect)

        # Replace nan values with real numbers for optimization
        amp_transect = np.nan_to_num(amp_transect)

        # Ignore noise at glacier termini
        start_idx = np.argmax(amp_transect)
        y = amp_transect[start_idx:]
        x = dist[start_idx:]

        # Model amplitude
        # y = ae^(bx)
        a, b = curve_fit(lambda t, a, b: a*np.exp(b*t), x, y, p0=(1500, -0.00015))[0]
        y = a*np.exp(b*x)
        axs[i].plot(x, y, 'r')

        # Add equation text box
        equation = '$y=%de^{%fx}$' % (a, b)
        axs[i].text(x[int(len(x)*.5)], y[int(len(y)*.25)], equation, color='r', fontsize=12)

        axs[i].set_title(snake_to_title(label))
        axs[i].set_xlabel('Upstream Distance (m)')
        axs[i].set_ylabel('Amplitude (m/yr)')

    fig.set_tight_layout(True)  
    plt.savefig(save, dpi=300)
    if show:
        plt.show()
    v_stack.fid.close()

def view_seasonal_secular_timeseries_fit(seasonal_model, secular_model, seasonal_raw, secular_raw, transect, label, idx=125, show=False):
    v_stack = ice.Stack('..' + V_STACKFILE)

    # Get image coordinate at index
    x, y = transect['x'][idx], transect['y'][idx]
    i, j = v_stack.hdr.xy_to_imagecoord(x, y)

    # Get model and raw data fit
    model_tdec = seasonal_model.tdec
    model_seasonal = seasonal_model._datasets['data'][:, i, j]
    model_secular = secular_model._datasets['data'][:, i, j]
    model_total = model_seasonal + model_secular

    raw_tdec = seasonal_raw.tdec
    raw_seasonal = seasonal_raw._datasets['data'][:, i, j]
    raw_secular = secular_raw._datasets['data'][:, i, j]
    raw_total = v_stack._datasets['data'][:, i, j]

    v_stack.fid.close()

    # Generate figure
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9))

    title = snake_to_title(label) + ' Timeseries Fit at (' + str(int(x)) + ', ' + str(int(y)) + ')'
    plt.suptitle(title, fontweight='bold')

    # Seasonal
    axs[0].scatter(raw_tdec, raw_seasonal)
    axs[0].plot(model_tdec, model_seasonal)
    axs[0].set_title('Seasonal')

    # Secular
    axs[1].scatter(raw_tdec, raw_secular)
    axs[1].plot(model_tdec, model_secular)
    axs[1].set_title('Secular')

    # Total
    axs[2].scatter(raw_tdec, raw_total)
    axs[2].plot(model_tdec, model_total)
    axs[2].set_title('Total')

    fig.set_tight_layout(True)
    plt.savefig('..' + FIG_ROOT + '/' + label + '_timeseries_fit.jpg', dpi=300)


def view_model():
    seasonal_model = ice.Stack('..' + SEASONAL_VELOCITY_MODEL_STACKFILE)
    seasonal_raw = ice.Stack('..' + SEASONAL_VELOCITY_RAW_STACKFILE)
    secular_model_rel = ice.Stack('..' + SECULAR_VELOCITY_MODEL_RELATIVE_STACKFILE)
    secular_model = ice.Stack('..' + SECULAR_VELOCITY_MODEL_STACKFILE)
    secular_raw = ice.Stack('..' + SECULAR_VELOCITY_RAW_STACKFILE)
    transects = load_transects()

    view_amplitude_model(transects, '..' + FIG_ROOT + '/seasonal_amplitude_model.jpg')

    for label in [COLUMBIA_MAIN, COLUMBIA_EAST, POST]:
        transect = transects[label]
        dist = ice.compute_path_length(transect['x'], transect['y'])
        view_transect_seasonal_secular_signal(seasonal_model, secular_model_rel, transect, label, dist)
        view_transect_seasonal_amp_phase(transect, dist, label)
        view_seasonal_secular_timeseries_fit(seasonal_model, secular_model, seasonal_raw, secular_raw, transect, label)

# ~ Main ......................................................................
def analyze_velocity():
    # Create model
    # model_velocity_tseries()

    # Visualize model
    view_model()
