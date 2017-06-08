#!/usr/bin/env python 
'''
grep keywords from header of a fits image

Author: Augustin Guyonnet
guyonnet@lpnhe.in2p3.fr
'''

import os, sys
import astropy.io.fits as pf
import glob


def getheader(filename, keys=[]):
    extension = 0
    if (filename.find("[") > -1):
        extension = int(filename.split('[')[1].replace("]",""))
        filename  = filename.split('[')[0]

    try:
        f = pf.open(filename)
        head = f[extension].header
    finally:
        f.close()
    if not keys:
        print(repr(head))
    else:
        print filename , [head.get(k) for k in keys]
        
   
    
if __name__ == "__main__":
    narg = len(sys.argv)
    if narg<2 :
        print "header.py [fitsimage(s)] -k [keyword(s)]"
        print "If keyword is none, print whole header"
        print
    keywords = []
    Images   = []
    k = 1
    while( k<narg ):
        if( sys.argv[k][0] != "-" ):
            Images.append( sys.argv[k] )
            k += 1
        elif( sys.argv[k] == "-k" ):
            k += 1
            keywords=sys.argv[k:] 
            break
 
    for img in Images :
        getheader(img , keys = keywords)
