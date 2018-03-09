import os, sys
mypath = os.path.dirname(__file__)

# Paths
HOLO_DIR = os.path.join(mypath,"../common_tools/data/")

# Verbose  mode
VERBOSE = False
# Debug mode
DEBUG = False

# Spectractor parameters
XWINDOW = 100 # window x size to search for the targetted object 
YWINDOW = 100  # window y size to search for the targetted object
XWINDOW_ROT = 50 # window x size to search for the targetted object 
YWINDOW_ROT = 50  # window y size to search for the targetted object

LAMBDA_MIN = 350 # minimum wavelength for spectrum extraction (in nm)
LAMBDA_MAX = 1100 # maxnimum wavelength for spectrum extraction (in nm)

