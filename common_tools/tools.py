import os, re
import scipy
from scipy.optimize import curve_fit
from scipy.misc import imresize
import numpy as np

def gauss(x,A,x0,sigma):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))

def gauss_and_bgd(x,A,x0,sigma,a,b):
    return gauss(x,A,x0,sigma) + line(x,a,b)

def line(x, a, b):
    return a*x + b

def parabola(x, a, b, c):
    return a*x*x + b*x + c

def fit_gauss(x,y,guess=[10,1000,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(gauss,x,y,p0=guess,bounds=bounds)
    return popt, pcov

def fit_gauss_and_bgd(x,y,guess=[10,1000,1,0,0,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(gauss_and_bgd,x,y,p0=guess,bounds=bounds,maxfev=100000)
    return popt, pcov

def fit_line(x,y,guess=[1,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(line,x,y,p0=guess,bounds=bounds)
    return popt, pcov

def fit_poly(x,y,order,w=None):
    cov = -1
    if(len(x)> order):
        if(w is None):
            fit, cov = np.polyfit(x,y,order,cov=True)
        else:
            fit, cov = np.polyfit(x,y,order,cov=True,w=w)
        model = lambda x : np.polyval(fit,x)
    else:
        fit = [0] * (order+1)
        model = y
    return(fit,cov,model)

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
