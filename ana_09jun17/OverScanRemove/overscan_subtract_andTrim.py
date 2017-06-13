#!/usr/bin/env python 
'''
trim illu section in fitsimage and subtract overscan

Author: Augustin Guyonnet
guyonnet@lpnhe.in2p3.fr
'''

import os, sys
import astropy.io.fits as pf
import glob
import pylab as pl
import telinst as instru
#/Users/lpnhe/harvard/soft/python_script/build/lib/lsst/utils/

def usage():
    print 'Return a trimmed an overscan subtracted image'
    print "header.py [-r outputrep] (images)"
    print "If no -r, output is local rep. outname is 'trim_'+img_name "
    print
    
def Do_overscan_subtract_andTrim(file):
    inst   = instru.telinst(file)
    data   = inst.Image(file)
    outimg = inst.OverscanSubtract_andTrim(data)
    hdr    = (inst.header).copy()  
    hdr.add_comment("Image is trimmed")
    (filepath, filename) = os.path.split(file)
    pf.writeto( 'trim_' + filename, outimg, hdr, clobber=True)
    
if __name__ == "__main__":
    narg = len(sys.argv)
    if narg<2 :
        usage()
 
    images   = []
    k = 1
    while( k<narg ):
        if( sys.argv[k] == "-k" ):
            k += 1
            outrep = sys.argv[k]
            continue
        images = sys.argv[k:]
        break
    

    for img in images :
        inst   = instru.telinst(img)
        data   = inst.Image(img)
        outimg = inst.OverscanSubtract_andTrim(data)
        hdr    = (inst.header).copy()  
        hdr.add_comment("Image is trimmed")
        (filepath, filename) = os.path.split(img)
        pf.writeto( 'trim_' + filename
                    , outimg, hdr, clobber=True)
  
