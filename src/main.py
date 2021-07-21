# ~ Imports ...................................................................
# Relative
from pre.preprocess_velocity import preprocess_velocity
from pre.preprocess_landsat import preprocess_landsat
from pre.preprocess_arcticdem import preprocess_arcticdem
from pre.preprocess_sentinel import preprocess_sentinel
from pre.find_centerline import find_centerlines

from analysis.bedrock import analyze_bedrock
from analysis.ice_thickness import analyze_ice_thickness
from analysis.velocity import analyze_velocity
from analysis.dem import analyze_dem
from analysis.driving_stress import analyze_driving_stress
from analysis.strain_stress import analyze_strain_stress

# ~ Methods ...................................................................
def preprocess(verbose=False):
    preprocess_velocity(verbose=verbose)
    preprocess_landsat()
    preprocess_arcticdem()
    preprocess_sentinel()

def analyze():
    # analyze_velocity()
    # analyze_bedrock()
    # analyze_ice_thickness()
    # analyze_dem()
    # analyze_driving_stress()
    analyze_strain_stress()

def main():
    # verbose = True
    # preprocess(verbose=verbose)
    # find_centerlines()
    analyze()

if __name__ == '__main__':
    main()