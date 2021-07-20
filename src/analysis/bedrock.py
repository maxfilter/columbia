
# ~ Imports ...................................................................
# Global
import iceutils as ice
from netCDF4 import Dataset
import numpy as np

# Local 
import sys
sys.path.append('..')
from constants import (BEDROCK_NC, FIG_ROOT,
    WGS_84_NSIDC_SEA_ICE_POLAR_STEREOGRAPHIC_NORTH, WGS_84_UTM_ZONE_6N_EPSG,
    COLUMBIA_MAIN, COLUMBIA_EAST, POST)
from utils import load_timeseries, load_transects, smooth_timeseries
from visualize import view_data, view_timeseries

# ~ Load Data .................................................................
def load_bedrock_data():
    """ Load NetCDF Bedrock data and warp to Raster EPSG 3413. """
    with Dataset('..' + BEDROCK_NC, mode='r') as fid:
        x, y = fid.variables['x'], fid.variables['y']
        X, Y = np.meshgrid(x, y)
        init_hdr = ice.RasterInfo(X=X, Y=Y, epsg=WGS_84_UTM_ZONE_6N_EPSG)

        labels = ['bedrock_z', 'thickness_1957', 'thickness_2007']
        bedrock_label = labels[0]
        data = np.array(fid.variables[bedrock_label])

    raster = ice.Raster(data=data, hdr=init_hdr)
    return ice.warp(raster,
        target_epsg=WGS_84_NSIDC_SEA_ICE_POLAR_STEREOGRAPHIC_NORTH)


def view_bedrock(bedrock, transects):
    timeseries = {}
    print(transects.keys())
    for label in [COLUMBIA_EAST, COLUMBIA_MAIN, POST]:
        timeseries[label] = {}
        timeseries[label]['values'] = smooth_timeseries(load_timeseries(bedrock, transects[label]))
        timeseries[label]['dist'] = ice.compute_path_length(transects[label]['x'], transects[label]['y'])

    view_timeseries(timeseries, '..' + FIG_ROOT + '/bedrock_z_timeseries.jpg')


def analyze_bedrock():
    bedrock = load_bedrock_data()
    transects = load_transects()

    # View Bedrock Raster
    view_data(bedrock.data, bedrock.hdr, '..' + FIG_ROOT + 
        '/bedrock_z_data.jpg', cmap='gray', clabel='Elevation (m)', 
        title='Columbia Glacier Bedrock Profile', transects=transects)

    # View Bedrock Timeseries
    view_bedrock(bedrock, transects)
    

if __name__ == '__main__':
    analyze_bedrock()