#!/usr/bin/env python 
'''
Compute im1 (-,/,*) coeff* im2

Author: Augustin Guyonnet
guyonnet@lpnhe.in2p3.fr
'''

import os, sys
import astropy.io.fits as pf
import numpy as np



def usage():
    print "Usage: Compute.py [image1] [+,/,*] [coeff] * [image2] [result_image]"
    print 
    sys.exit(1)


if __name__ == "__main__":
  
    if len(sys.argv) != 6:
        usage()

    
    f1     = pf.open(sys.argv[1])
    image1 = f1[0].data
    image2 = (pf.open(sys.argv[4]))[0].data
    op      = sys.argv[2]
    coef    = float(sys.argv[3])
    outname = sys.argv[5]

    image2[image2<=0] = 1e-21
    print "Masterflat pixels <= 0 are replaced by 1e-21 before flatfielding"
    
    if(op=='+'):
        outimg = image1 + coef * image2     
    if(op=='/'):
        outimg = image1 / (coef * image2)    
    if(op=='*'):
        outimg = image1 * (coef * image2)    
    
    hdr    = (f1[0].header).copy()
    info   = str("Image has been subtracted by :"+ str(sys.argv[2])) 
    hdr.add_comment(info)   
    pf.writeto(outname, outimg, hdr, clobber=True)
  
