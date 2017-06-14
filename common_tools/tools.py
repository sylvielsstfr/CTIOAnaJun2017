import os
from scipy.optimize import curve_fit
import numpy as np

def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gauss(x,y,guess=[10,1000,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(gauss,x,y,p0=guess,bounds=bounds)
    return popt, pcov


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
        
