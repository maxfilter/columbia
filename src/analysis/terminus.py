""" Analyze Terminus Positions Columbia Glacier

TODO:
- extract mcnabb terminus positions
- determine terminus positions 2012-2016 w Sobel filter?
- plot terminus position timeseries for each transect
"""

# Global
import shapefile
import matplotlib.pyplot as plt

# Local
import sys
sys.path.append('..')
from constants import MCNABB_TERMINUS_SHAPEFILE

def get_mcnabb_terminus_positions():
    sf = shapefile.Reader('..' + MCNABB_TERMINUS_SHAPEFILE)
    shapes = sf.shapes()
    print(shapes[0])
    records = sf.records()
    fields = sf.fields
    print('Fields', fields)

    fig, axs = plt.subplots(figsize=(9,9))
    for shape in shapes:
        points = np.array(shape.points)
        axs.plot(points[:,0], points[:,1])
    plt.show()
