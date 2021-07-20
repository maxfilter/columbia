""" Download ArcticDEM using wget. """

# ~ Imports ...................................................................

# Global
import csv
import glob
import iceutils as ice
import numpy as np
import os
import ssl
import subprocess
import tarfile
import time
import wget

# Relative
from pathlib import Path
from urllib.error import HTTPError

# Local
import sys
sys.path.append('..')
from constants import ARCTICDEM_FEATURES_CSV, ARCTICDEM_ROOT, ARCTICDEM_STACKFILE

# ~ Download ..................................................................
def download_arcticdem():
    # Fix for weird certificate issue
    ssl._create_default_https_context = ssl._create_unverified_context
    arcticdem_path = '..' + ARCTICDEM_ROOT
    Path(arcticdem_path).mkdir(parents=True, exist_ok=True)

    with open('..' + ARCTICDEM_FEATURES_CSV) as fid:
        reader = csv.DictReader(fid)
        for row in reader:
            url = row['fileurl']
            name_key = list(row.keys())[0] # Name key includes weird character
            name = arcticdem_path + '/' + row[name_key]

            dem_tarfile = arcticdem_path + '/' + url.split('/')[-1]

            success = False
            exists = True
            sleep_time_seconds = 1
            if not os.path.exists(name):
                while not success and exists:
                    try:
                        wget.download(url, arcticdem_path, bar=None)
                        success = True
                    except HTTPError as e:
                        if e.code == 404:
                            # Break from loop
                            print(url, 'not in server.')
                            exists = False
                        else:
                            # Wait so server does not get overwhelmed
                            time.sleep(sleep_time_seconds)
                            sleep_time_seconds *= 2

                if exists:
                    # Unpack tarfile
                    tar = tarfile.open(dem_tarfile)
                    tar.extractall(name)
                    tar.close()

                    # Delete tarfile
                    os.remove(dem_tarfile)
            else:
                print(name, 'exists.')


# ~ Stack .....................................................................
def get_arcticdem_id(root):
    return root.split('/')[-1] + '_dem'


def get_arcticdem_tdec(root):
    date = get_arcticdem_id(root).split('_')[2]
    return ice.datestr2tdec(yy=int(date[0:4]), mm=int(date[4:6]), dd=int(date[6:8]))


def get_arcticdem_raster(root):
    gdal_cmd = 'gdalwarp -r bilinear -tr 8 8 -t_srs EPSG:3413 -of ENVI -te -3130000 642500 -3090000 680000 %s %s'
    temp_path = '..' + ARCTICDEM_ROOT + '/temp'
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    tif_id = get_arcticdem_id(root)
    tif_path = root + '/' + tif_id + '.tif'
    out_path = temp_path + '/%s.dat' % tif_id

    # Run command to warp tif
    if not os.path.exists(out_path):
        cmd = gdal_cmd % (tif_path, out_path)
        subprocess.run(cmd, shell=True)

    # Get data
    raster = ice.Raster(out_path)

    # Remove temporary files
    for ext in ['.dat', '.hdr', '.dat.aux.xml']:
        os.remove(temp_path + '/' + tif_id + ext)

    return raster


def create_arcticdem_stack():
    files = sorted(glob.glob('..' + ARCTICDEM_ROOT + '/SETSM_*'))

    init_hdr = None
    init_tdec, init_data = [], []

    # Initialize Stack, add data later to prevent memory overuse
    for path in files:
        init_tdec.append(get_arcticdem_tdec(path))
        raster = get_arcticdem_raster(path)
        # Set invalid values in data
        data = np.array(raster.data)
        data[data == -9999] = np.nan
        init_data.append(data)
        if init_hdr is None:
            init_hdr = raster.hdr

    # Sort data before adding to Stack
    indsort = np.argsort(init_tdec)
    init_tdec = np.array(init_tdec)[indsort]
    init_data = np.array(init_data)[indsort]

    # Create stack
    stack = ice.Stack('..' + ARCTICDEM_STACKFILE, mode='w', init_tdec=np.array(init_tdec), init_rasterinfo=init_hdr)

    # Create dataset
    stack.fid.create_dataset('data', data=np.array(init_data))
    stack.fid.close()

    # Delete temp folder
    os.rmdir('..' + ARCTICDEM_ROOT + '/temp')


# ~ Main ......................................................................
def preprocess_arcticdem(download=False):
    if download:
        download_arcticdem()

    create_arcticdem_stack()
