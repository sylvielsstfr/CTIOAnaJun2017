import os, re
import scipy
from scipy.optimize import curve_fit
from scipy.misc import imresize
import numpy as np
from astropy.modeling import models, fitting
import warnings

def gauss(x,A,x0,sigma):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))

def gauss_and_bgd(x,a,b,A,x0,sigma):
    return gauss(x,A,x0,sigma) + line(x,a,b)

def multigauss_and_bgd(x,*params):
    out = line(x,params[0],params[1])
    for k in range((len(params)-2)/3) :
        out += gauss(x,*params[2+3*k:2+3*k+3])
    return out

def line(x, a, b):
    return a*x + b

def parabola(x, a, b, c):
    return a*x*x + b*x + c

def fit_gauss(x,y,guess=[10,1000,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(gauss,x,y,p0=guess,bounds=bounds)
    return popt, pcov

def fit_multigauss_and_bgd(x,y,guess=[10,1000,1,0,0,1],bounds=(-np.inf,np.inf)):
    maxfev=100000
    popt,pcov = curve_fit(multigauss_and_bgd,x,y,p0=guess,bounds=bounds,maxfev=maxfev)
    return popt, pcov

def fit_line(x,y,guess=[1,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(line,x,y,p0=guess,bounds=bounds)
    return popt, pcov

def fit_poly(x,y,degree,w=None):
    cov = -1
    if(len(x)> order):
        if(w is None):
            fit, cov = np.polyfit(x,y,degree,cov=True)
        else:
            fit, cov = np.polyfit(x,y,degree,cov=True,w=w)
        model = lambda x : np.polyval(fit,x)
    else:
        fit = [0] * (degree+1)
        model = y
    return(fit,cov,model)

def fit_poly2d(x,y,z,degree):
    # Fit the data using astropy.modeling
    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)
    return p

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx,array[idx]

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
