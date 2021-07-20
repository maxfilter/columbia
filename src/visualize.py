import iceutils as ice
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from constants import COLUMBIA_EAST, COLUMBIA_MAIN, POST

from utils import get_transect_signal, load_timeseries, load_transects, snake_to_title

def view_data(data, hdr, save, alpha=1.0, clabel=None, clim=None,
    cmap='Spectral_r', figsize=(9,9), ref=None, show=True, 
    title='Title', transects={}, xlabel='X (m)', ylabel='Y (m)'):
    """ Visualize SAR data

    Parameters
    ----------
    data : array_like
        2D array of image data to show
    hdr : ice.RasterInfo
        header object with projection details
    save : str
        filename to save the figure as
    alpha : float
        alpha value for plotting data. Useful if plotting over SAR reference.
        Default: 1.0.
    clabel : str
        Label for the color axis in the plot
    clim : tuple
        Color limit for plot
    cmap : str
        Color map for plot. Default: Spectral_r.
    figsize : tuple
        Figure size for plot. Default: (9, 9).
    ref : str
        Name of file with SAR image for background
    show : bool
        Show the generated plot in a new window when generated. Default: True.
    title : str
        Title for the plot
    transects: dict
        Display the transects on the data
    xlabel : str
        Label for the x axis
    ylabel : str
        Label for the y axis
    """
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

    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)
    cmap = ice.get_cmap(cmap)

    # Add data
    if db is not None:
        ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high,
                        extent=hdr.extent)
    im = ax.imshow(data, extent=hdr.extent, cmap=cmap, clim=clim,
        alpha=alpha)

    # Add transects
    for transect in transects.values():
        ax.plot(transect['x'], transect['y'])

    # Add labels
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(clabel)

    # Adjust layout
    fig.set_tight_layout(True)
    ax.set_aspect('equal')

    plt.savefig(save, dpi=300)

    if show:
        plt.show()

def view_timeseries(timeseries, save, xlabel='Upstream Distance (m)', ylabel='X (m)'):
    """ Plot timeseries for Columbia East, Columbia Main, and Post Glacier. """
    idx = 0

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9,9))
    for label, data in timeseries.items():
        axs[idx].plot(data['dist'], data['values'])
        axs[idx].set_title(snake_to_title(label))
        idx += 1
    
    fig.set_tight_layout(True)
    plt.savefig(save, dpi=300)
    plt.show()


def plot_streamlines(streamlines, hdr, save, alpha=1.0, clabel=None, 
    cmap='plasma', figsize=(9,9), ref=None, show=True, title='Title', 
    xlabel='X (m)', ylabel='Y (m)'):
    """ Visualize set of streamlines.

    Parameters
    ----------
    streamlines : array_like
        List of streamline dicts
    hdr : ice.RasterInfo
        header object with projection details
    save : str
        filename to save the figure as
    alpha : float
        alpha value for plotting data. Useful if plotting over SAR reference.
        Default: 1.0.
    clabel : str
        Label for the color axis in the plot. Default, None
    cmap : str
        Color map for plot. Default: plasma.
    figsize : tuple
        Figure size for plot. Default: (9, 9).
    ref : str
        Name of file with SAR image for background. Default: None.
    show : bool
        Show the generated plot in a new window when generated. Default: True.
    title : str
        Title for the plot. Default: Title.
    xlabel : str
        Label for the x axis. Default: X (m).
    ylabel : str
        Label for the y axis. Default: Y (m).
    """
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

    # Determine upper/lower limits across all streamlines
    vmin = float('inf')
    vmax = 0
    for streamline in streamlines:
        for speed in streamline['s']:
            vmin = min(vmin, speed)
            vmax = max(vmax, speed)
    
    fig, ax = plt.subplots(figsize=(9,9))

    # Add ref image if provided
    if db is not None:
        ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high,
                        extent=hdr.extent)

    # Plot streamlines and corresponding labels
    for streamline in streamlines:
        scat = ax.scatter(streamline['x'], streamline['y'], alpha=alpha, 
            marker='s', s=1, c=streamline['s'], cmap=cmap, vmin=vmin,
            vmax=vmax)

        if streamline['label'] is not None:
            ax.text(streamline['x'][-1], streamline['y'][-1] - 200,
                streamline['label'], fontweight='bold', c='white', ha='center',
                va='top')

    # Add labels
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    cb = fig.colorbar(scat, cax=cax)
    cb.set_label(clabel)

    # Adjust layout
    fig.set_tight_layout(True)
    ax.set_aspect('equal')

    plt.savefig(save, dpi=500)

    if show:
        plt.show()


def view_transect_signal(model, save, clabel='', cmap='Spectral_r', figsize=(9,9), title=None, show=False):
    transects = load_transects()

    fig, axs = plt.subplots(nrows=3, figsize=figsize)

    if title is not None:
        plt.suptitle(title, fontweight='bold')

    for i, label in enumerate([COLUMBIA_MAIN, COLUMBIA_EAST, POST]):
        transect = transects[label]
        dist = ice.compute_path_length(transect['x'], transect['y'])
        extent = [model.tdec[0], model.tdec[-1], 0, dist[-1]]
        signal = get_transect_signal(model, transect)

        axs[i].set_title(snake_to_title(label))

        im = axs[i].imshow(signal, origin='lower', aspect='auto', cmap=cmap, extent=extent)
        cbar = plt.colorbar(im, ax=axs[i])
        cbar.set_label(clabel)

    for ax in axs:
        ax.set_ylabel('Upstream distance (m)')
        ax.set_xlabel('Year')

    fig.set_tight_layout(True)
    plt.savefig(save, dpi=300)

    if show:
        plt.show()

def view_timeseries_fit(model, raw, save, idx=100, figsize=(9,9), title=None, show=False, ylabel='', xlabel='Year'):
    transects = load_transects()
    fig, axs = plt.subplots(nrows=3, figsize=figsize)

    if title is not None:
        plt.suptitle(title, fontweight='bold')

    for i, label in enumerate([COLUMBIA_MAIN, COLUMBIA_EAST, POST]):
        transect = transects[label]

        # Get image coordinate at index
        x, y = transect['x'][idx], transect['y'][idx]
        model_i, model_j = model.hdr.xy_to_imagecoord(x, y)
        raw_i, raw_j = raw.hdr.xy_to_imagecoord(x, y)

        model_tseries = model._datasets['data'][:, model_i, model_j]
        raw_tseries = raw._datasets['data'][:, raw_i, raw_j]

        axs[i].set_title(snake_to_title(label) + ' Fit at (' + str(int(x)) + ', ' + str(int(y)) + ')')
        axs[i].scatter(raw.tdec, raw_tseries)
        axs[i].plot(model.tdec, model_tseries)

    for ax in axs:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.set_tight_layout(True)
    plt.savefig(save, dpi=300)

    if show:
        plt.show()
