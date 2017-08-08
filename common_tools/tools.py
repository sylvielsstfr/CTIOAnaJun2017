import os, re
import scipy
from scipy.optimize import curve_fit
from scipy.misc import imresize
import numpy as np

def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def line(x, a, b):
    return a*x + b

def fit_gauss(x,y,guess=[10,1000,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(gauss,x,y,p0=guess,bounds=bounds)
    return popt, pcov

def fit_line(x,y,guess=[1,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(line,x,y,p0=guess,bounds=bounds)
    return popt, pcov


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
        

def resize_image(data,Nx_final,Ny_final):
    # F mode stands for 'float' : output is float and not int between 0 and 255
    data_resampled=scipy.misc.imresize(data,(Nx_final,Ny_final),mode='F')
    data_rescaled = data_resampled
    # following line is not useful with mode='F', uncomment it with mode='I'
    #data_rescaled=(np.max(data)-np.min(data))*(data_resampled-np.min(data_resampled))/(np.max(data_resampled)-np.min(data_resampled))+np.min(data)
    return(data_rescaled)
