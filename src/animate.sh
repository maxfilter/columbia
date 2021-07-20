python ice_animate_stack.py ../data/landsat/columbia_landsat8_stack.hdf5 --cmap gray --title 'Columbia Landsat 8 Stack' --clabel "" --fps 10 --save ../figures/landsat8_stack.mp4

python ice_animate_stack.py ../data/landsat/columbia_landsat7_stack.hdf5 --cmap gray --title 'Columbia Landsat 7 Stack' --clabel "" --fps 10 --save ../figures/landsat7_stack.mp4

python ice_animate_stack.py ../data/arcticdem/columbia_arcticdem_stack.hdf5 --ref ../data/landsat/landsat.tif --alpha 0.7 --title 'Columbia ArcticDEM Stack' --clabel 'Elevation (m)' --fps 10 --save ../figures/arcticdem_stack.mp4

python ice_animate_stack.py ../data/sentinel/columbia_sentinel_stack.hdf5 --ref ../data/landsat/landsat.tif --alpha 0.7 --clabel "" --title 'Columbia Sentinel Stack' --fps 10 --save ../figures/sentinel_stack.mp4

python ice_animate_stack.py ../data/velocity/columbia_seasonal_velocity_model_stack.hdf5 --clim -1500 1500 --cmap coolwarm --ref ../data/landsat/landsat.tif --alpha 0.7 --title 'Columbia Seasonal Variation' --clabel 'Seasonal Variation (m/yr)' --fps 50 --save ../figures/seasonal_velocity.mp4

python ice_animate_stack.py ../data/velocity/columbia_secular_velocity_model_relative_stack.hdf5 --clim -1500 1500 --cmap coolwarm --ref ../data/landsat/landsat.tif --alpha 0.7 --title 'Columbia Long Term Variation' --clabel 'Secular Variation (m/yr)' --fps 50 --save ../figures/secular_velocity.mp4

python ice_animate_stack.py ../data/arcticdem/columbia_dem_model_relative_stack.hdf5 --clim -150 150 --cmap coolwarm --ref ../data/landsat/landsat.tif --alpha 0.7 --title 'Columbia Elevation Change' --clabel 'Elevation Change (m/yr)' --fps 50 --save ../figures/dem_model.mp4