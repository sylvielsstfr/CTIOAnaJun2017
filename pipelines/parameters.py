# Paths
HOLO_DIR = "../common_tools/data/"

# Debug mode
VERBOSE = 0
DEBUG = False

# Spectractor parameters
XWINDOW = 100 # window x size to search for the targetted object 
YWINDOW = 100  # window y size to search for the targetted object
XWINDOW_ROT = 50 # window x size to search for the targetted object 
YWINDOW_ROT = 50  # window y size to search for the targetted object

LAMBDA_MIN = 350 # minimum wavelength for spectrum extraction (in nm)
LAMBDA_MAX = 1100 # maxnimum wavelength for spectrum extraction (in nm)

def extract_info_from_CTIO_header(obj,header):
    obj.date_obs = header['DATE-OBS']
    obj.airmass = header['AIRMASS']
    obj.expo = header['EXPTIME']
    obj.filters = header['FILTERS']
    obj.filter = header['FILTER1']
    obj.disperser = header['FILTER2']
