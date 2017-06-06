#!/usr/bin/env python 
'''
Skylev mean and rms of illu regions and overscans of amplifiers of a list of images 

Author: Augustin Guyonnet
guyonnet@lpnhe.in2p3.fr
'''

import os, sys
import telinst as instru
import numpy as np



def usage():
    print "Usage: skylev.py [images]"
    print "Returns mean and rms of illu regions and overscans of amplifiers of a list of images."
    print
    sys.exit(1)


if __name__ == "__main__":
  
    if len(sys.argv) <=1:
        usage()

    images = sys.argv[1:]
    for img in images:
        inst = instru.telinst(img)
        data = inst.Image(img)
        for amp in ('11','12','21','22'):
            overscan  = np.mean(inst.OverscanRegion(data, amp))
            soverscan = np.std(inst.OverscanRegion(data, amp))
            illu      = np.mean(inst.IlluRegion(data, amp))
            sillu     = np.std(inst.IlluRegion(data, amp))
            print img, ' amp', amp, 'overscan(mean, std) : (%.2f, %.2f)'%(overscan, soverscan), ' illu(mean, std) : (%.2f, %.2f)'%(illu, sillu)

        print 
