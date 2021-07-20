""" Download Landsat data through Earth Explorer API. """

# ~ Imports ...................................................................
# Global
import glob
import iceutils as ice
import json
import numpy as np
import os
import subprocess
import tarfile

# Relative
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
from pathlib import Path

# Local
import sys
sys.path.append('..')
from constants import EE_LOGIN_JSON, LANDSAT_ROOT, LANDSAT7_STACKFILE, LANDSAT8_STACKFILE

# ~ Download ..................................................................
def download_landsat():
    Path('..' + LANDSAT_ROOT).mkdir(parents=True, exist_ok=True)

    with open(EE_LOGIN_JSON) as f:
        login = json.load(f)
        username = login['username']
        password = login['password']

    api = API(username, password)

    scenes = api.search(
                dataset='landsat_8_c1',
                latitude=61.2197,
                longitude=-146.8953,
                start_date='2011-01-01',
                end_date='2016-12-31',
                max_cloud_cover=10
            )

    scenes += api.search(
                dataset='landsat_etm_c1',
                latitude=61.2197,
                longitude=-146.8953,
                start_date='2011-01-01',
                end_date='2016-12-31',
                max_cloud_cover=10
            )

    ee = EarthExplorer(username, password)

    for scene in scenes:
        path = '..' + LANDSAT_ROOT + '/' + scene['landsat_product_id'] + '.tar.gz'
        if not os.path.exists(path):
            print('Downloading', path)
            ee.download(scene['landsat_scene_id'], output_dir='..' + LANDSAT_ROOT)
        else:
            print(path, 'exists.')

    ee.logout()

# ~ Stack .....................................................................
def get_landsat_id(root):
    return root.split('/')[-1].split('.')[0]
    
def get_landsat_tdec(root):
    date = get_landsat_id(root).split('_')[3]
    return ice.datestr2tdec(yy=int(date[0:4]), mm=int(date[4:6]), dd=int(date[6:8]))

def get_landsat_raster(root):
    gdal_cmd = 'gdalwarp -r bilinear -tr 30 30 -t_srs EPSG:3413 -of ENVI -te -3130000 642500 -3090000 680000 %s %s'
    temp_path = '..' + LANDSAT_ROOT + '/temp'
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    tif_id = get_landsat_id(root)

    with tarfile.open(root) as fid:
        try:
            fid.extract(tif_id + '_B1.TIF', temp_path)
        except:
            return None
    
    tif_path = temp_path + '/' + tif_id + '_B1.TIF'
    out_path = temp_path + '/%s.dat' % tif_id

    print(out_path)

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

def create_landsat_stack(files, stackfile):
    init_hdr = None
    init_tdec, init_data = [], []

    for path in files:
        raster = get_landsat_raster(path)
        if raster is None:
            continue

        init_tdec.append(get_landsat_tdec(path))
        data = np.array(raster.data)
        init_data.append(data)
        
        if init_hdr is None:
            init_hdr = raster.hdr
    
    # Sort data before adding to Stack
    indsort = np.argsort(init_tdec)
    init_tdec = np.array(init_tdec)[indsort]
    init_data = np.array(init_data)[indsort]

    # Create stack
    stack = ice.Stack(stackfile, mode='w', init_tdec=np.array(init_tdec), init_rasterinfo=init_hdr)

    # Create dataset
    stack.fid.create_dataset('data', data=np.array(init_data))
    stack.fid.close()

    # Delete temp folder
    os.rmdir('..' + LANDSAT_ROOT + '/temp')

# ~ Main ......................................................................
def preprocess_landsat(download=False):
    if download:
        download_landsat()

    # Landsat 7
    landsat7_files = sorted(glob.glob('..' + LANDSAT_ROOT + '/LE07_*'))
    create_landsat_stack(landsat7_files, '..' + LANDSAT7_STACKFILE)

    # Landsat 8
    landsat8_files = sorted(glob.glob('..' + LANDSAT_ROOT + '/LC08_*'))
    create_landsat_stack(landsat8_files, '..' + LANDSAT8_STACKFILE)