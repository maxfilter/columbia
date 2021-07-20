#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Global
import iceutils as ice
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import warnings

# Local
import sys
sys.path.append('..')
from constants import (FIG_ROOT, LANDSAT_RASTERFILE, VX_STACKFILE, VY_STACKFILE, V_STACKFILE)
from utils import save_transects, extent_to_proj_win
from visualize import plot_streamlines

# ~ Get Centerline Starting Coordinate ........................................

def get_start_points_manual(extent):
    """ Manually select a list of start points from raster to calculate
    streamlines from.
    """
    proj_win = extent_to_proj_win(extent)
    landsat = ice.Raster(rasterfile='..' + LANDSAT_RASTERFILE, projWin=proj_win)
    _, axs = plt.subplots(figsize=(9,9))
    axs.set_title('Please Select Starting Points.', fontweight='bold')
    axs.imshow(landsat.data, extent=extent, cmap='gray')
    return plt.ginput(n=-1, timeout=300)

def get_start_points_line(start=None, end=None):
    """ Gets all points in the glacier head perpendicular transect.

    Currently only works with vertical/horizontal lines.
    """
    v = ice.Stack('..' + V_STACKFILE)

    if start is None or end is None:
        fig, ax = plt.subplots(figsize=(9,9))
        ax.set_title('Please select Transect Start and End Points.', fontweight='bold')
        ax.imshow(v._datasets['data'][0], cmap='Spectral_r')
        start, end = plt.ginput(n=-1, timeout=300)[-2:]

    # Assume vertical line
    i0, i1 = int(start[1]), int(end[1])
    j0, j1 = int(start[0]), int(end[0])
    di = abs(i1-i0)
    dj = abs(j1-j0)

    start_points = []
    if dj < di:
        for i in range(min(i0, i1), max(i0, i1)):
            start_points.append(v.hdr.imagecoord_to_xy(i, j0))
    else:
        for j in range(min(j0, j1), max(j0, j1)):
            start_points.append(v.hdr.imagecoord_to_xy(i0, j))
    
    return start_points

# ~ Streamline ................................................................

def get_streamline(vx, vy, hdr, start_point):
    Ny, Nx = vx.shape
    visited = np.full(vx.shape, False, dtype=bool)

    # Stream function to integrate
    def _stream(t, xy):
        sx, sy = xy
        si, sj = hdr.xy_to_imagecoord(sx, sy)
        u = vx[si, sj]
        v = vy[si, sj]
        speed = np.sqrt(u**2 + v**2)
        return [u/speed, v/speed]

    # Initialize integrator
    r = integrate.ode(_stream).set_integrator('lsoda')
    r.set_initial_value(start_point, 0)

    streamline = {
        'x' : [],
        'y' : [],
        's' : None,
        'label' : None
    }
    # previous i, j
    pi, pj = -1, -1
    while r.successful():
        try:
            x, y = r.integrate(r.t + 1)
        except Exception:
            break
        
        # Break if either value is nan
        if np.isnan(x) or np.isnan(y) or x < -3119850:
            break

        i, j = hdr.xy_to_imagecoord(x, y)

        # Break if out of bounds
        if i < 0 or j < 0 or i >= Ny or j >= Nx:
            break

        if visited[i, j] and pi != i and pj != j:
            # If this point was not visited at a time not in the previous step,
            # terminate to prevent infinite loop
            break
        elif pi != i and pj != j:
            # Only add the coordinate if it was not already visited
            # Integrator will still consider the point even when it is 
            # not added to the streamline
            visited[i, j] = True
            pi = i
            pj = j
            streamline['x'].append(x)
            streamline['y'].append(y)

    return streamline

def resample_streamline(streamline):
    """ Resample streamline so that each point is equidistant.

    https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values
    """
    x, y = streamline['x'], streamline['y']
    M = 500

    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])

    t = np.linspace(0,u.max(), M)
    streamline['x'] = np.interp(t, u, x)
    streamline['y'] = np.interp(t, u, y)

    return streamline


def get_streamlines(vx, vy, hdr, start_points):
    streamlines = []
    for start in start_points:
        streamline = get_streamline(vx, vy, hdr, start)
        streamline = resample_streamline(streamline)
        streamlines.append(streamline)
    
    for idx, streamline in enumerate(streamlines):
        speeds = []
        for x, y in zip(streamline['x'], streamline['y']):
            i, j = hdr.xy_to_imagecoord(x, y)
            speed = np.sqrt(vx[i,j]**2 + vy[i,j]**2)
            speeds.append(speed)
        
        streamline['s'] = np.flip(speeds)

        # Reverse list so start point is glacier terminus
        streamline['x'] = np.flip(streamline['x'])
        streamline['y'] = np.flip(streamline['y'])
        streamline['label'] = str(idx)
    
    return streamlines

def get_fastest_streamline(streamlines):
    centerline = streamlines[0]
    max_avg_speed = np.mean(streamlines[0]['s'])
    for streamline in streamlines:
        mean = np.mean(streamline['s'])
        if mean > max_avg_speed:
            max_avg_speed = mean
            centerline = streamline

    return centerline


def find_centerlines(mode='line', show_figs=False, verbose=False):
    """ Determine the streamline of maximum average flow on the glacier

    Parameters
    ----------
    mode: str
        'line': calculate streamlines starting at each pixel on a line
        'manual': select each starting point manually
    show_figs: bool
        Pause execution to display figures as they are generated
    verbose: bool
        Print progress to stdout
    """
    print('Loading velocity components...')
    vx_stack = ice.Stack('..' + VX_STACKFILE)
    vy_stack = ice.Stack('..' + VY_STACKFILE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vx = ice.inpaint(np.nanmean(vx_stack._datasets['data'], axis=0))
        vy = ice.inpaint(np.nanmean(vy_stack._datasets['data'], axis=0))

    hdr = vx_stack.hdr

    if verbose:
        print('Getting start points for streamlines...')

    centerlines = []
    if mode == 'manual':
        start_points = get_start_points_manual(hdr.extent)
        centerlines = get_streamlines(vx, vy, hdr, start_points)
    else:
        # Start and end points defining the glacier head perpendicular transect
        # (j, i) image coords
        start = [(236, 236), (180,310), (180, 36)]
        end = [(236, 272), (195, 310), (180, 48)]
        labels = ['Columbia Main', 'Columbia East', 'Post']

        for s, t, label in zip(start, end, labels):
            start_points = get_start_points_line(start=s, end=t)

            if verbose:
                print('Calculating centerline for ' + label + '...')

            streamlines = get_streamlines(vx, vy, hdr, start_points)
            centerline = get_fastest_streamline(streamlines)
            centerline['label'] = label
            centerlines.append(centerline)
    
    save = '..' + FIG_ROOT + '/centerline_transect.jpg'

    if verbose:
        print('Generating figure and saving to', save + '...')

    plot_streamlines(centerlines, hdr, save,
        clabel='Mean Velocity (m/yr)',
        ref='..' + LANDSAT_RASTERFILE,
        title='Columbia Glacier Transects',
        show=show_figs)

    save_transects(centerlines)

if __name__ == '__main__':
    find_centerlines()


# end of file