""" Constants """
# Directories
DATA_ROOT = '/data'
FIG_ROOT = '/figures'
BIN_ROOT = DATA_ROOT + '/bin'
TIF_ROOT = DATA_ROOT + '/tif'
VELOCITY_ROOT = DATA_ROOT + '/velocity'
TRANSECT_ROOT = DATA_ROOT + '/transects'
LANDSAT_ROOT = DATA_ROOT + '/landsat'
BEDROCK_ROOT = DATA_ROOT + '/bedrock'
ARCTICDEM_ROOT = DATA_ROOT + '/arcticdem'
SENTINEL_ROOT = DATA_ROOT + '/sentinel'
ICE_THICKNESS_ROOT = DATA_ROOT + '/ice_thickness'
TERMINUS_ROOT = DATA_ROOT + '/terminus'

# Stackfile names
V_STACKFILE = VELOCITY_ROOT + '/columbia_v_stack.hdf5'
VX_STACKFILE = VELOCITY_ROOT + '/columbia_vx_stack.hdf5'
VY_STACKFILE = VELOCITY_ROOT + '/columbia_vy_stack.hdf5'
EX_STACKFILE = VELOCITY_ROOT + '/columbia_ex_stack.hdf5'
EY_STACKFILE = VELOCITY_ROOT + '/columbia_ey_stack.hdf5'
E_XX_STACKFILE = VELOCITY_ROOT + '/columbia_e_xx_stack.hdf5'
E_YY_STACKFILE = VELOCITY_ROOT + '/columbia_e_yy_stack.hdf5'
E_XY_STACKFILE = VELOCITY_ROOT + '/columbia_e_xy_stack.hdf5'
ARCTICDEM_STACKFILE = ARCTICDEM_ROOT + '/columbia_arcticdem_stack.hdf5'
LANDSAT7_STACKFILE = LANDSAT_ROOT + '/columbia_landsat7_stack.hdf5'
LANDSAT8_STACKFILE = LANDSAT_ROOT + '/columbia_landsat8_stack.hdf5'
SENTINEL_STACKFILE = SENTINEL_ROOT + '/columbia_sentinel_stack.hdf5'
ICE_THICKNESS_STACKFILE = ICE_THICKNESS_ROOT + '/columbia_ice_thickness_stack.hdf5'

SECULAR_VELOCITY_MODEL_STACKFILE = VELOCITY_ROOT + '/columbia_secular_v_model_stack.hdf5'
SECULAR_VELOCITY_MODEL_RELATIVE_STACKFILE = VELOCITY_ROOT + '/columbia_secular_v_model_relative_stack.hdf5'
SEASONAL_VELOCITY_MODEL_STACKFILE = VELOCITY_ROOT + '/columbia_seasonal_v_model_stack.hdf5'
SEASONAL_VELOCITY_RAW_STACKFILE = VELOCITY_ROOT + '/columbia_seasonal_v_raw_stack.hdf5'
SECULAR_VELOCITY_RAW_STACKFILE = VELOCITY_ROOT + '/columbia_secular_v_raw_stack.hdf5'

DEM_MODEL_STACKFILE = ARCTICDEM_ROOT + '/columbia_dem_model_stack.hdf5'
DEM_MODEL_RELATIVE_STACKFILE = ARCTICDEM_ROOT + '/columbia_dem_model_relative_stack.hdf5'
ICE_THICKNESS_MODEL_STACKFILE = ICE_THICKNESS_ROOT + '/columbia_ice_thickness_model_stack.hdf5'
ICE_THICKNESS_MODEL_RELATIVE_STACKFILE = ICE_THICKNESS_ROOT + '/columbia_ice_thickness_model_relative_stack.hdf5'

# Rasterfile names
LANDSAT_RASTERFILE = LANDSAT_ROOT + '/landsat.tif'
SEASONAL_VELOCITY_AMPLITUDE_RASTERFILE = VELOCITY_ROOT + '/columbia_seasonal_v_amplitude.tif'
SEASONAL_VELOCITY_PHASE_RASTERFILE = VELOCITY_ROOT + '/columbia_seasonal_v_phase.tif'

# NetCDF File names
BEDROCK_NC = BEDROCK_ROOT + '/ColumbiaBedrock.nc'

# Extents
# Also need to change command extent in preprocess_*.py TODO: take extent as variable
EXTENT_EPSG_3413 = [-3130000, -3090000, 642500, 680000]

# EPSG
WGS_84_UTM_ZONE_6N_EPSG = 32606
WGS_84_NSIDC_SEA_ICE_POLAR_STEREOGRAPHIC_NORTH = 3413

# Misc
MCNABB_TERMINUS_SHAPEFILE = TERMINUS_ROOT + '/columbia'
EE_LOGIN_JSON = './earth_explorer.json'
ARCTICDEM_FEATURES_CSV = ARCTICDEM_ROOT + '/features.csv'
COLUMBIA_MAIN = 'columbia_main'
COLUMBIA_EAST = 'columbia_east'
POST = 'post'

# Values
GRAVITATIONAL_ACCELERATION = 9.81
ICE_DENSITY = 800