""" Create Sentinel Stack from data downloaded through auto-generated script. """

# ~ Imports ...................................................................
# Global
import glob
import iceutils as ice
import numpy as np
import os
import shutil
import subprocess

# Relative
from pathlib import Path
from zipfile import ZipFile

# Local
import sys
sys.path.append('..')
from constants import SENTINEL_ROOT, SENTINEL_STACKFILE

# ~ Stack .....................................................................
def get_sentinel_id(root):
    return root.split('/')[-1].split('.')[0]
    

def get_sentinel_tdec(root):
    date = get_sentinel_id(root).split('_')[4]
    return ice.datestr2tdec(yy=int(date[0:4]), mm=int(date[4:6]), dd=int(date[6:8]))


def get_sentinel_raster(root):
    gdal_cmd = 'gdalwarp -tps -r bilinear -tr 10 10 -t_srs EPSG:3413 -of ENVI -te -3130000 642500 -3090000 680000 %s %s'
    temp_path = '..' + SENTINEL_ROOT + '/temp'
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    tif_id = get_sentinel_id(root)

    with ZipFile(root, 'r') as fid:
        names = fid.namelist()
        for name in names:
            if name.endswith('.tiff'):
                tif_path = fid.extract(name, path=temp_path)
    
    out_path = temp_path + '/%s.dat' % tif_id

    # Run command to warp tif
    if not os.path.exists(out_path):
        cmd = gdal_cmd % (tif_path, out_path)
        subprocess.run(cmd, shell=True)

    # Get data
    raster = ice.Raster(out_path)

    # Remove temporary files
    os.remove(tif_path)
    for ext in ['.dat', '.hdr', '.dat.aux.xml']:
        os.remove(temp_path + '/' + tif_id + ext)

    return raster


def create_sentinel_stack():
    files = sorted(glob.glob('..' + SENTINEL_ROOT + '/zipfiles/*.zip'))

    init_hdr = None
    init_tdec, init_data = [], []

    for path in files:
        raster = get_sentinel_raster(path)

        # Continue to next path on failure
        if raster is None:
            continue
        
        # Get date
        init_tdec.append(get_sentinel_tdec(path))

        # Get data
        data = np.array(raster.data, dtype='f')

        # Mask out invalid data
        data[data == 0] = np.nan

        # Adjust distribution
        data = 10.0 * np.log10(data + 0.1)
        init_data.append(data)
        
        if init_hdr is None:
            init_hdr = raster.hdr
    
    # Sort data before adding to Stack
    indsort = np.argsort(init_tdec)
    init_tdec = np.array(init_tdec)[indsort]
    init_data = np.array(init_data)[indsort]

    # Create stack
    stack = ice.Stack('..' + SENTINEL_STACKFILE, mode='w', init_tdec=np.array(init_tdec), init_rasterinfo=init_hdr)

    # Create dataset
    stack.fid.create_dataset('data', data=np.array(init_data))
    stack.fid.close()

    # Delete temp folder
    shutil.rmtree('..' + SENTINEL_ROOT + '/temp')


# ~ Main ......................................................................
def preprocess_sentinel():
    create_sentinel_stack()