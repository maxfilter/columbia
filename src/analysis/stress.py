
import iceutils as ice
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from scipy.signal import savgol_filter

import sys
sys.path.append('..')
from utils import load_timeseries, load_transects, smooth_timeseries, snake_to_title
from constants import COLUMBIA_EAST, COLUMBIA_MAIN, DEM_MODEL_STACKFILE, E_XX_STACKFILE, E_XY_STACKFILE, E_YY_STACKFILE, FIG_ROOT, GRAVITATIONAL_ACCELERATION, ICE_THICKNESS_MODEL_STACKFILE, LANDSAT_RASTERFILE, POST, RHO_ICE, SEASONAL_VELOCITY_MODEL_STACKFILE, SECULAR_VELOCITY_MODEL_STACKFILE, STRESS_ROOT, T_DX_STACKFILE, T_DY_STACKFILE
sys.path.append('analysis')
from bedrock import load_bedrock_data


def create_strain_stress_stacks():
    Path('..' + STRESS_ROOT).mkdir(parents=True, exist_ok=True)

    vx_seasonal_stack = ice.Stack('..' + SEASONAL_VELOCITY_MODEL_STACKFILE.replace('_v_', '_vx_'))
    vx_secular_stack = ice.Stack('..' + SECULAR_VELOCITY_MODEL_STACKFILE.replace('_v_', '_vx_'))
    vy_seasonal_stack = ice.Stack('..' + SEASONAL_VELOCITY_MODEL_STACKFILE.replace('_v_', '_vy_'))
    vy_secular_stack = ice.Stack('..' + SECULAR_VELOCITY_MODEL_STACKFILE.replace('_v_', '_vy_'))
    ice_thickness_stack = ice.Stack('..' + ICE_THICKNESS_MODEL_STACKFILE)

    bedrock = load_bedrock_data()
    b = ice.interpolate_array(bedrock.data, bedrock.hdr, None, None, ref_hdr = vx_seasonal_stack.hdr)

    vx_data = np.array(vx_seasonal_stack._datasets['data']) + np.array(vx_secular_stack._datasets['data'])
    vy_data = np.array(vy_seasonal_stack._datasets['data']) + np.array(vy_secular_stack._datasets['data'])
    
    dx = vx_seasonal_stack.hdr.dx
    dy = vy_seasonal_stack.hdr.dy

    e_xx, e_yy, e_xy, t_dx, t_dy = [], [], [], [], []
    
    print('Getting data...')
    for vx, vy, h in tqdm(zip(vx_data, vy_data, ice_thickness_stack._datasets['data'])):
        # Replace nans with median value
        vx = np.nan_to_num(vx, nan=np.nanmedian(vx.ravel()))
        vy = np.nan_to_num(vy, nan=np.nanmedian(vy.ravel()))

        robust_opts = {'window_size': 250, 'order': 2}

        # Compute stress strain
        strain_dict, stress_dict = ice.compute_stress_strain(vx, vy, dx=dx, dy=dy, grad_method='sgolay', inpaint=False, rotate=True, h=h, b=b, rho_ice=RHO_ICE, g=GRAVITATIONAL_ACCELERATION, **robust_opts)
        e_xx.append(strain_dict['e_xx'])
        e_yy.append(strain_dict['e_yy'])
        e_xy.append(strain_dict['e_xy'])
        t_dx.append(stress_dict['tdx'])
        t_dy.append(stress_dict['tdy'])


    if not os.path.exists('..' + E_XX_STACKFILE):
        e_xx_stack = ice.Stack('..' + E_XX_STACKFILE, mode='w', init_tdec=vx_seasonal_stack.tdec, init_rasterinfo=vx_seasonal_stack.hdr)
        e_xx_stack.fid.create_dataset('data', data=np.array(e_xx))
        e_xx_stack.fid.close()

    if not os.path.exists('..' + E_YY_STACKFILE):
        e_yy_stack = ice.Stack('..' + E_YY_STACKFILE, mode='w', init_tdec=vx_seasonal_stack.tdec, init_rasterinfo=vx_seasonal_stack.hdr)
        e_yy_stack.fid.create_dataset('data', data=np.array(e_yy))
        e_yy_stack.fid.close()

    if not os.path.exists('..' + E_XY_STACKFILE):
        e_xy_stack = ice.Stack('..' + E_XY_STACKFILE, mode='w', init_tdec=vx_seasonal_stack.tdec, init_rasterinfo=vx_seasonal_stack.hdr)
        e_xy_stack.fid.create_dataset('data', data=np.array(e_xy))
        e_xy_stack.fid.close()

    if not os.path.exists('..' + T_DX_STACKFILE):
        t_dx_stack = ice.Stack('..' + T_DX_STACKFILE, mode='w', init_tdec=vx_seasonal_stack.tdec, init_rasterinfo=vx_seasonal_stack.hdr)
        t_dx_stack.fid.create_dataset('data', data=np.array(t_dx))
        t_dx_stack.fid.close()

    if not os.path.exists('..' + T_DY_STACKFILE):
        t_dy_stack = ice.Stack('..' + T_DY_STACKFILE, mode='w', init_tdec=vx_seasonal_stack.tdec, init_rasterinfo=vx_seasonal_stack.hdr)
        t_dy_stack.fid.create_dataset('data', data=np.array(t_dy))
        t_dy_stack.fid.close()

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
    #TODO: divide by distance between points in x here
    ds_dx = - savgol_filter(elevation, win, poly, deriv=1)
    return - RHO_ICE * GRAVITATIONAL_ACCELERATION * np.multiply(ice_thickness, ds_dx)

def get_driving_stress_timeseries(dem_model, thickness_model, transect):
    t_d = []
    for i in range(len(dem_model._datasets['data'])):
        elevation_data = dem_model._datasets['data'][i]
        ice_thickness_data = thickness_model._datasets['data'][i]

        elevation_tseries = load_timeseries(None, transect, data=elevation_data, hdr=dem_model.hdr)
        thickness_tseries = load_timeseries(None, transect, data=ice_thickness_data, hdr=thickness_model.hdr)

        t_d.append(compute_driving_stress(thickness_tseries, elevation_tseries))

    return t_d

def get_transect_driving_stress_timeseries(dem_model, thickness_model, transects):
    transects = load_transects()
    dem_model = ice.Stack('..' + DEM_MODEL_STACKFILE)
    thickness_model = ice.Stack('..' + ICE_THICKNESS_MODEL_STACKFILE)

    # Compute driving stress for each transect
    driving_stress_dict = {}
    for label in [COLUMBIA_MAIN, COLUMBIA_EAST, POST]:
        transect = transects[label]
        driving_stress_dict[label] = get_driving_stress_timeseries(dem_model, thickness_model, transect)

    return driving_stress_dict

# ~ Figures ...................................................................
def view_strain_rates(ref=None, idx=100):
    e_xx_stack = ice.Stack('..' + E_XX_STACKFILE)
    e_yy_stack = ice.Stack('..' + E_YY_STACKFILE)
    e_xy_stack = ice.Stack('..' + E_XY_STACKFILE)

    hdr = e_xx_stack.hdr

    e_xx = e_xx_stack._datasets['data'][idx]
    e_yy = e_yy_stack._datasets['data'][idx]
    e_xy = e_xy_stack._datasets['data'][idx]

    # Load reference SAR image
    if ref is not None:
        sar = ice.Raster(rasterfile=ref)
        if sar.hdr != hdr:
            sar.resample(hdr)
        db = 10.0 * np.log10(sar.data)
        low = np.percentile(db.ravel(), 5)
        high = np.percentile(db.ravel(), 99.9)
    else:
        db = None

    fig, axs = plt.subplots(nrows=3, figsize=(9,9))
    plt.suptitle('Columbia Glacier Strain Rates', fontweight='bold')

    for ax in axs:
        if db is not None:
            ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high, 
            extent=hdr.extent)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high, 
            extent=hdr.extent)
    e_xx_im = axs[0].imshow(e_xx, cmap='coolwarm', clim=[-2, 2], extent=hdr.extent, alpha=0.7)
    e_yy_im = axs[1].imshow(e_yy, cmap='coolwarm', clim=[-2, 2], extent=hdr.extent, alpha=0.7)
    e_xy_im = axs[2].imshow(e_xy, cmap='Spectral_r', clim=[0, 4], extent=hdr.extent, alpha=0.7)

    for i in range(3):
        ims = [e_xx_im, e_yy_im, e_xy_im]
        labels = ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{xy}$']
        div = make_axes_locatable(axs[i])
        cax = div.append_axes('right', '5%', '5%')
        cb = fig.colorbar(ims[i], cax=cax)
        cb.set_label(labels[i])

    fig.set_tight_layout(True)
    plt.savefig('..' + FIG_ROOT + '/strain_fields.jpg', bbox_inches='tight', dpi=300)
    plt.close()

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

def view_driving_stress(ref=None, idx=1000):
    t_dx_stack = ice.Stack('..' + T_DX_STACKFILE)
    t_dy_stack = ice.Stack('..' + T_DY_STACKFILE)

    hdr = t_dx_stack.hdr

    t_dx = t_dx_stack._datasets['data'][idx]
    t_dy = t_dy_stack._datasets['data'][idx]

    # Load reference SAR image
    if ref is not None:
        sar = ice.Raster(rasterfile=ref)
        if sar.hdr != hdr:
            sar.resample(hdr)
        db = 10.0 * np.log10(sar.data)
        low = np.percentile(db.ravel(), 5)
        high = np.percentile(db.ravel(), 99.9)
    else:
        db = None
    
    fig, axs = plt.subplots(nrows=2, figsize=(9,9))

    # Create title
    datestr = ice.tdec2datestr(t_dx_stack.tdec[idx])
    title = 'Columbia Glacier Driving Stress '
    plt.suptitle(title + datestr, fontweight='bold')

    for ax in axs:
        if db is not None:
            ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high, 
            extent=hdr.extent)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    t_dx_im = axs[0].imshow(t_dx, cmap='coolwarm', clim=[-2e5, 2e5], extent=hdr.extent, alpha=0.7)
    axs[0].set_title('Along Flow Driving Stress')
    t_dy_im = axs[1].imshow(t_dy, cmap='coolwarm', clim=[-1.5e5, 1.5e5], extent=hdr.extent, alpha=0.7)
    axs[1].set_title('Across Flow Driving Stress')

    # Add colorbars
    for i in range(2):
        ims = [t_dx_im, t_dy_im]
        labels = ['$\\tau_{dx}$', '$\\tau_{dy}$']
        div = make_axes_locatable(axs[i])
        cax = div.append_axes('right', '5%', '5%')
        cb = fig.colorbar(ims[i], cax=cax)
        cb.set_label(labels[i])

    fig.set_tight_layout(True)
    plt.show()

# ~ Animation .................................................................
def animate_strain_rates(ref=None, fps=50):
    e_xx_stack = ice.Stack('..' + E_XX_STACKFILE)
    e_yy_stack = ice.Stack('..' + E_YY_STACKFILE)
    e_xy_stack = ice.Stack('..' + E_XY_STACKFILE)

    hdr = e_xx_stack.hdr

    e_xx = e_xx_stack._datasets['data']
    e_yy = e_yy_stack._datasets['data']
    e_xy = e_xy_stack._datasets['data']

    # Load reference SAR image
    if ref is not None:
        sar = ice.Raster(rasterfile=ref)
        if sar.hdr != hdr:
            sar.resample(hdr)
        db = 10.0 * np.log10(sar.data)
        low = np.percentile(db.ravel(), 5)
        high = np.percentile(db.ravel(), 99.9)
    else:
        db = None
    
    fig, axs = plt.subplots(nrows=3, figsize=(9,9))

    # Create title
    datestr = ice.tdec2datestr(e_xx_stack.tdec[0])
    title = 'Columbia Glacier Strain Rates '
    tx = plt.suptitle(title + datestr, fontweight='bold')

    for ax in axs:
        if db is not None:
            ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high, 
            extent=hdr.extent)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    e_xx_im = axs[0].imshow(e_xx[0], cmap='coolwarm', clim=[-2, 2], extent=hdr.extent, alpha=0.7)
    e_yy_im = axs[1].imshow(e_yy[0], cmap='coolwarm', clim=[-2, 2], extent=hdr.extent, alpha=0.7)
    e_xy_im = axs[2].imshow(e_xy[0], cmap='Spectral_r', clim=[0, 4], extent=hdr.extent, alpha=0.7)

    # Add colorbars
    for i in range(3):
        ims = [e_xx_im, e_yy_im, e_xy_im]
        labels = ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{xy}$']
        div = make_axes_locatable(axs[i])
        cax = div.append_axes('right', '5%', '5%')
        cb = fig.colorbar(ims[i], cax=cax)
        cb.set_label(labels[i])

    def animate(i):
        e_xx_im.set_data(e_xx[i])
        e_yy_im.set_data(e_yy[i])
        e_xy_im.set_data(e_xy[i])
        datestr = ice.tdec2datestr(e_xx_stack.tdec[i])
        tx.set_text(title + datestr)

    fig.set_tight_layout(True)
    interval = 1000/fps # Convert fps to interval in milliseconds
    anim = animation.FuncAnimation(fig, animate, interval=interval, frames=len(e_xx), repeat=True)
    anim.save('..' + FIG_ROOT + '/strain_rate_movie.mp4', dpi=300)
    plt.close()

def animate_driving_stress(ref=None, fps=50):
    t_dx_stack = ice.Stack('..' + T_DX_STACKFILE)
    t_dy_stack = ice.Stack('..' + T_DY_STACKFILE)

    hdr = t_dx_stack.hdr

    t_dx = t_dx_stack._datasets['data']
    t_dy = t_dy_stack._datasets['data']

    # Load reference SAR image
    if ref is not None:
        sar = ice.Raster(rasterfile=ref)
        if sar.hdr != hdr:
            sar.resample(hdr)
        db = 10.0 * np.log10(sar.data)
        low = np.percentile(db.ravel(), 5)
        high = np.percentile(db.ravel(), 99.9)
    else:
        db = None
    
    fig, axs = plt.subplots(nrows=2, figsize=(9,9))

    # Create title
    datestr = ice.tdec2datestr(t_dx_stack.tdec[0])
    title = 'Columbia Glacier Driving Stress '
    tx = plt.suptitle(title + datestr, fontweight='bold')

    for ax in axs:
        if db is not None:
            ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high, 
            extent=hdr.extent)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    t_dx_im = axs[0].imshow(t_dx[0], cmap='coolwarm', clim=[-2e5, 2e5], extent=hdr.extent, alpha=0.7)
    axs[0].set_title('Along Flow Driving Stress')
    t_dy_im = axs[1].imshow(t_dy[0], cmap='coolwarm', clim=[-1.5e5, 1.5e5], extent=hdr.extent, alpha=0.7)
    axs[1].set_title('Across Flow Driving Stress')

    # Add colorbars
    for i in range(2):
        ims = [t_dx_im, t_dy_im]
        labels = ['$\\tau_{dx}$', '$\\tau_{dy}$']
        div = make_axes_locatable(axs[i])
        cax = div.append_axes('right', '5%', '5%')
        cb = fig.colorbar(ims[i], cax=cax)
        cb.set_label(labels[i])

    def animate(i):
        t_dx_im.set_data(t_dx[i])
        t_dy_im.set_data(t_dy[i])
        datestr = ice.tdec2datestr(t_dx_stack.tdec[i])
        tx.set_text(title + datestr)

    fig.set_tight_layout(True)
    interval = 1000/fps # Convert fps to interval in milliseconds
    anim = animation.FuncAnimation(fig, animate, interval=interval, frames=len(t_dx), repeat=True)
    anim.save('..' + FIG_ROOT + '/driving_stress_movie.mp4', dpi=300)
    plt.close()

def animate_driving_stress_timeseries(driving_stress_dict, transects, tdec, save, fps=50):
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

# ~ Main ......................................................................
def analyze_strain_stress():
    transects = load_transects()
    dem_model = ice.Stack('..' + DEM_MODEL_STACKFILE)
    thickness_model = ice.Stack('..' + ICE_THICKNESS_MODEL_STACKFILE)
    
    # Create Model
    # create_strain_stress_stacks()
    # driving_stress_dict = get_transect_driving_stress_timeseries(dem_model, thickness_model, transects)

    # View Models
    ref = '..' + LANDSAT_RASTERFILE
    # view_strain_rates(ref=ref)
    # animate_strain_rates(ref=ref)
    animate_driving_stress(ref=ref)
    # view_driving_stress(ref=ref)
    # animate_driving_stress_timeseries(driving_stress_dict, transects, dem_model.tdec, '..' + FIG_ROOT + '/driving_stress_timeseries.mp4')
    # view_driving_stress_timeseries(driving_stress_dict, dem_model, transects, '..' + FIG_ROOT + '/driving_stress_transect.jpg')