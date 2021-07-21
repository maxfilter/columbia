
import iceutils as ice
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('..')
from constants import E_XX_STACKFILE, E_XY_STACKFILE, E_YY_STACKFILE, FIG_ROOT, LANDSAT_RASTERFILE, SEASONAL_VELOCITY_MODEL_STACKFILE, SECULAR_VELOCITY_MODEL_STACKFILE


def create_strain_rate_stacks():
    vx_seasonal_stack = ice.Stack('..' + SEASONAL_VELOCITY_MODEL_STACKFILE.replace('_v_', '_vx_'))
    vx_secular_stack = ice.Stack('..' + SECULAR_VELOCITY_MODEL_STACKFILE.replace('_v_', '_vx_'))
    vy_seasonal_stack = ice.Stack('..' + SEASONAL_VELOCITY_MODEL_STACKFILE.replace('_v_', '_vy_'))
    vy_secular_stack = ice.Stack('..' + SECULAR_VELOCITY_MODEL_STACKFILE.replace('_v_', '_vy_'))

    vx_data = np.array(vx_seasonal_stack._datasets['data']) + np.array(vx_secular_stack._datasets['data'])
    vy_data = np.array(vy_seasonal_stack._datasets['data']) + np.array(vy_secular_stack._datasets['data'])
    
    dx = vx_seasonal_stack.hdr.dx
    dy = vy_seasonal_stack.hdr.dy

    e_xx, e_yy, e_xy = [], [], []
    
    print('Getting data...')
    for vx, vy in tqdm(zip(vx_data, vy_data)):
        # Replace nans with median value
        vx = np.nan_to_num(vx, nan=np.nanmedian(vx.ravel()))
        vy = np.nan_to_num(vy, nan=np.nanmedian(vy.ravel()))

        robust_opts = {'window_size': 250, 'order': 2}

        # Compute stress strain
        strain_dict, _ = ice.compute_stress_strain(vx, vy, dx=dx, dy=dy, grad_method='sgolay', inpaint=False, **robust_opts)
        e_xx.append(strain_dict['e_xx'])
        e_yy.append(strain_dict['e_yy'])
        e_xy.append(strain_dict['e_xy'])

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


def analyze_strain_stress():
    # Create Model
    # create_strain_rate_stacks()

    # View Models
    view_strain_rates(ref='..' + LANDSAT_RASTERFILE)
    animate_strain_rates(ref='..' + LANDSAT_RASTERFILE)