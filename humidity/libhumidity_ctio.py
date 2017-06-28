#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:29:57 2017

@author: dagoret-campagnesylvie
"""


import air 
import humidity as hum
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const


# LibRadTran parameters at ground for CTIO

P0= 775.28625 *u.hPa  
T0= 273.9 *u.K  
Z0= 2.2 *u.km 

rho_Air = 2.05022e+19 / (u.cm)**3  
rho_H2O =  8.90711e+16 / (u.cm)**3

# very imprtant parameter to convert relative humidty to PWV
PWV_0= 5.180 # kg/m2  : precipitable water vapor or mm
Ha_0 = 0.002664675 # kg/m3  : humidité absolue


def HRtoPWV(hr,P,T):
    
    """
    Estimate the PWV to give to LibRadTran
    
    P in hPa
    T in K
    HR relative humidity
    """
    
    absolute_humidity=hum.rh2ah(hr,P*100.,T)
    
    pwv=PWV_0/Ha_0*absolute_humidity
    return pwv
    


if __name__ == "__main__":
    
    hr=np.linspace(0.,1.999,50)
    pwv=HRtoPWV(hr,P0/u.hPa,T0/u.K)
    pwv_sea=HRtoPWV(hr,1015.,293.0)
    
    plt.plot(hr,pwv,"r-",label="CTIO, P=775hPa, T= 274 K")
    plt.plot(hr,pwv_sea,'b-',label="sea level, P=1015 hPa, T = 293 K")
    plt.xlabel("Relative Humidity")
    plt.ylabel("PWV (mm)")
    plt.grid()
    plt.title("Relation between HR and PWV")
    plt.legend()
    plt.savefig("hrtopwv.pdf")
    plt.show()