#!/usr/bin/env python 
'''
Given a seeing, find cosmics in fits images 

Author: Augustin Guyonnet
aguyonnet@fas.harvard.edu
'''

import os, sys
import numpy as np
import astropy.io.fits as pf
import logging
import argparse
from scipy import stats
import pylab as pl
from matplotlib.colors import LogNorm



def BkgdStat(data, control_plot):
    lenX  = len(data)
    lenY  = len(data[0])
    c, low, up = stats.sigmaclip(data, 5, 3)
    print 'Lower and upper value included in bkgd : ', low, up
    mean  = np.mean(c)
    sigma = np.std(c)
    print 'Bkgd mean and sigma = ', mean, sigma
    if (control_plot is True):
        check = np.zeros([lenX, lenY])
        check = np.where(((data<up) & (data>low)), 1, 0)    
        pl.pcolor(check, cmap='hot')
        pl.colorbar()
        pl.show()
    return mean, sigma
    

'''A class to implement various filters on the pixels'''
class filters(object):
    def __init__(self, fitsimage, control_plot = False, **kwargs):
        (self.filepath, self.fitsimage) = os.path.split(fitsimage)
        self.control_plot = control_plot
        
    def Cosmics(self, seeing, frame = None, **kwargs):
        data  = (pf.open(os.path.join(self.filepath, self.fitsimage)))[0].data
        if frame is not None :
            xi =  frame[0]
            xf =  frame[1]
            yi =  frame[2]
            yf =  frame[3]
            data = data[yi:yf , xi:xf]
            hdu = pf.PrimaryHDU(data)
            hdu.writeto(os.path.join(self.filepath, 'frame_'+self.fitsimage), clobber=True)
        Iter  = 0
        count = 1000000
        lenX  = len(data)
        lenY  = len(data[0])
        print 'x_size, y_size : ', lenX, lenY
        mean, sigma = BkgdStat(data, self.control_plot)
        CosmicImage = np.zeros([lenX, lenY])
        total = 0
        while ((count) and (Iter <5)): 
            print " Iter ", Iter+1
            count = self.LaplacianFilter(sigma, mean, seeing, data, CosmicImage)
            print " Number of cosmic found ", count
            total+=count
            Iter+=1

        ''' Write mask if cosmics found on frame'''
        if total :
            hdu = pf.PrimaryHDU(CosmicImage)
            hdu.writeto(os.path.join(self.filepath, 'cosmics_'+self.fitsimage), clobber=True)
        return

      
#//Laplacian filter
#/*!Cuts (based on the article -> astro-ph/0108003): 
#  -cut_lap : the laplacian operator increases the noise by a factor of 
#  "sqrt(1.25)"
#  
#  -cut_f : 2*sigma(med), where sigma(med) is the variance of the
#  sky's median calculated in a box (3*3), here. 
#  (sigma(med) = sigma(sky)*1.22/sqrt(n); n = size of the box)
#  
#  -cut_lf : calculated from the article.
#  Factor 2.35 -> to have the seeing in arc sec */

    def LaplacianFilter(self, Sigma, Mean, seeing, data, CosmicImage):
        xmax    = len(data)-1
        ymax    = len(data[0])-1
        l       = 0.0
        f       = 0.0
        med     = 0.0
        cut     = 5 * Sigma
        cut_lap = cut * np.sqrt(1.25)
        cut_f   = 2 * (Sigma*1.22/3)
        cut_lf  = 2./(seeing*2.35-1)
        count   = 0
        for j in range(1, ymax):
            for i in range(1, xmax):
                #Calculation of the laplacian and the median only for pixels > 3 sigma
                if (data[i,j] > cut+Mean ):
                    l   = data[i,j] - 0.25*(data[i-1,j]   + data[i+1,j] 
                                            + data[i,j-1] + data[i,j+1])
                    med = np.median(data[i-1:i+2,j-1:j+2])
                    f   = med - Mean  #f is invariant by addition of a constant
                    #Construction of a cosmic image
                    if((l>cut_lap) and ((f<cut_f) or ((l/f)>cut_lf))):
                       CosmicImage[i,j] = 1
                       data[i,j]        = med
                       count           += 1  
        return count


def grabargs():
    usage = "usage: [%prog] [options]\n"
    usage += "Returns a fits image with cosmic flagged as 1"
   
    parser = argparse.ArgumentParser(description='Usage',
                                     epilog="detect cosmic/hot pixels")
    parser.add_argument('-f',"--frame", nargs='+', type=int, 
		        help = "image frame[xi,xf : yi, yf] where to search for cosmics", 
		        default=None)
    parser.add_argument('-p',"--plot", 
		        help = "show control plots", 
		        action='store_true') 
    parser.add_argument('-i',"--img", type=str, 
	                help = "list of fitsimages to be processed", 
	                default=None)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args         = grabargs()
    control_plot = args.plot
    image        = args.img
    frame        = args.frame
    print frame
    cosmics      = filters(image, control_plot = control_plot)
    seeing       = 0.7
    cosmics.Cosmics(seeing, frame=frame)
