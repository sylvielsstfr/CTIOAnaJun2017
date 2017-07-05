#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
equivalent width ratio vs PWV


Created on Wed Jul  5 21:46:05 2017

@author: dagoret-campagnesylvie
"""
import matplotlib.pyplot as plt
import re,os
import pyfits
import pandas as pd
import numpy as np

import astropy.units as u
import sys

sys.path.append('../humidity')
import humidity as hum
import libhumidity_ctio as humctio

import sys
sys.path.append('./libradtransim')
from libsimulateTranspCTIOScattAbs import ProcessSimulation

sys.path.append('../../spectrumsim')
import libCTIOTransm as ctiosim 

os.environ['PYSYN_CDBS']

import pysynphot as S

path_ctiodata='../../spectrumsim/CTIOThroughput'
qe_filename='qecurve.txt'

CTIO_COLL_SURF=0.9*(u.m)**2/(u.cm)**2  # CTIO surface
WLMIN=3000.
WLMAX=11000.

calspec_sed='hd111980_stis_003.fits'

O2WL1=740
O2WL2=750
O2WL3=780
O2WL4=790


H2OWL1=830
H2OWL2=880
H2OWL3=990
H2OWL4=1000

#-------------------------------------------------------------------------------
def ComputeEquivalentWidthNonLinear(wl,spec,wl1,wl2,wl3,wl4,ndeg=3):
    """
    ComputeEquivalentWidth : compute the equivalent width must be computed
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg,rcond=2.0e-16*len(x_cont))
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)

    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    

    # compute bin size in the band
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                                  
    
    # calculation of equivalent width    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()    
    
    return equivalent_width

#----------------------------------------------------------------------------------
def ShowEquivalentWidth2NonLinear(wl,spec,wl1,wl2,wl3,wl4,label='absortion line',ndeg=3):
    """
    ShowEquivalentWidth : show how the equivalent width must be computed
    """
    
    f, axarr = plt.subplots(1,2,figsize=(12,4))
    
    ################
    ## Figure 1
    #################
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    axarr[0].plot(wl_cut,spec_cut,marker='.',color='red')
    axarr[0].plot([wl2,wl2],[ymin,ymax],'k-.',lw=2)
    axarr[0].plot([wl3,wl3],[ymin,ymax],'k-.',lw=2)
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
    axarr[0].plot(x_cont,y_cont,marker='.',color='blue',lw=0)
    axarr[0].plot(fit_line_x,fit_line_y,'g--',lw=2)
    
    axarr[0].grid(True)
    axarr[0].set_xlabel('$\lambda$ (nm)')
    axarr[0].set_ylabel('ADU per second')
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    
    external_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    
    
    ############
    # Figure 2
    ###########
    
    axarr[1].plot(wl_cut,ratio,marker='.',color='red')
    axarr[1].plot(wl_cut[external_indexes],ratio[external_indexes],marker='.',color='blue',lw=0)
    
    axarr[1].plot([wl2,wl2],[0,1.2],'k-.',lw=2)
    axarr[1].plot([wl3,wl3],[0,1.2],'k-.',lw=2)
    axarr[1].grid(True)
    axarr[1].set_ylim(0.8*ratio.min(),1.2*ratio.max())
    
    axarr[1].set_xlabel('$\lambda$ (nm)')
    axarr[1].set_ylabel('No unit')
    
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width
    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()
    
    
    title = 'Equivalent width computation for {}'.format(label)
    f.suptitle(title)
    f.show()
    
    return equivalent_width


def EQWRatio(am,oz,p,all_pwv,sed,bp_ctio):
    """
    Compute all equivalent width ratio
    """
    
    all_eqw_o2 = []
    all_eqw_h2o = []
    
    for pwv in all_pwv:
        path,file=ProcessSimulation(am,pwv,oz,p)   
        fullfilename=os.path.join(path,file)
        atm_data=np.loadtxt(fullfilename)
        wl_atm=atm_data[:,0]
        tr_atm=atm_data[:,1]
        bp_atm = S.ArrayBandpass(wl_atm*10.,tr_atm, name='atm') 
    
        # telescope throughput and atmosphere
        bp_ctio_atm = bp_ctio*bp_atm
    
        # SED and telescope and atmosphere
        obs_ctio = S.Observation(sed,bp_ctio_atm)
        
        wl=np.array(obs_ctio.wave)/10.
        spec=np.array(obs_ctio.flux)
        
        eqw_o2=ComputeEquivalentWidthNonLinear(wl,spec,O2WL1,O2WL2,O2WL3,O2WL4,ndeg=3)
        eqw_h2o=ComputeEquivalentWidthNonLinear(wl,spec,H2OWL1,H2OWL2,H2OWL3,H2OWL4,ndeg=3)
    
        all_eqw_o2.append(eqw_o2)
        all_eqw_h2o.append(eqw_h2o)
        
    
    all_eqw_o2=np.array(all_eqw_o2)
    all_eqw_h2o=np.array(all_eqw_h2o)
    
    all_eqwratio=all_eqw_h2o/all_eqw_o2
    return all_eqwratio


#---------------------------------------------------------------------------

if __name__ == '__main__':
    
    #SED
    sed_filename = os.path.join(os.environ['PYSYN_CDBS'], 'calspec',calspec_sed)
    sed = S.FileSpectrum(sed_filename)
    
    #CTIO
    S.refs.setref(area=CTIO_COLL_SURF.decompose(), waveset=None)
    S.refs.set_default_waveset(minwave=WLMIN, maxwave=WLMAX, delta=10, log=False)
    
    
    # QE
    wl_qe,tr_qe=ctiosim.Get_QE()
    bp_ctio_qe = S.ArrayBandpass(wl_qe*10.,tr_qe, name='CTIO QE')
    # Mirrors
    wl_m,tr_m=ctiosim.Get_Mirror()
    bp_ctio_m = S.ArrayBandpass(wl_m*10.,tr_m*tr_m, name='CTIO Mirror2')  # two mirrors
    # Filter
    wl_f,tr_f=ctiosim.Get_RG175()
    bp_ctio_f = S.ArrayBandpass(wl_f*10.,tr_f, name='RG175')  # filter
    
    # Combine for telescope throughput
    bp_ctio=bp_ctio_qe*bp_ctio_m*bp_ctio_f
    
    #atmosphere
    AM=1.0
    PWV=4.
    OZ=300.
    PRESSURE=775.28625 
    
    path,file=ProcessSimulation(AM,PWV,OZ,PRESSURE)   
    fullfilename=os.path.join(path,file)
    atm_data=np.loadtxt(fullfilename)
    wl_atm=atm_data[:,0]
    tr_atm=atm_data[:,1]
    bp_atm = S.ArrayBandpass(wl_atm*10.,tr_atm, name='atm') 
    
    # telescope throughput and atmosphere
    bp_ctio_atm = bp_ctio*bp_atm
    
    # SED and telescope and atmosphere
    obs_ctio = S.Observation(sed,bp_ctio_atm)
    
    plt.figure()
    plt.plot(obs_ctio.wave,obs_ctio.flux,'r-')
    plt.xlim(WLMIN,WLMAX)
    plt.show()

    
    EQW_O2=ShowEquivalentWidth2NonLinear(np.array(obs_ctio.wave)/10.,np.array(obs_ctio.flux),O2WL1,O2WL2,O2WL3,O2WL4,label='$O_2$ absortion line',ndeg=3)
    EQW_H2O=ShowEquivalentWidth2NonLinear(np.array(obs_ctio.wave)/10.,np.array(obs_ctio.flux),H2OWL1,H2OWL2,H2OWL3,H2OWL4,label='$O_2$ absortion line',ndeg=3)
   
    print 'EQW_O2 = ',EQW_O2,' nm'
    print 'EQW_H2O = ',EQW_H2O,' nm' 
    
    
    
    all_pwv=np.arange(0,10,0.5)
    
    all_eqwratio_z1=EQWRatio(1.,OZ,PRESSURE,all_pwv,sed,bp_ctio)
    all_eqwratio_z2=EQWRatio(1.5,OZ,PRESSURE,all_pwv,sed,bp_ctio)
    all_eqwratio_z3=EQWRatio(2.0,OZ,PRESSURE,all_pwv,sed,bp_ctio)
    
    
    
    plt.figure()
    plt.plot(all_pwv,all_eqwratio_z1,'ro',lw=2,label='z=1.0')
    plt.plot(all_pwv,all_eqwratio_z2,'bo',lw=2,label='z=1.5')
    plt.plot(all_pwv,all_eqwratio_z3,'go',lw=2,label='z=2.0')
    plt.title('EQW ratio H2O:O2 in libradtran')
    plt.xlabel('pwv in mm')
    plt.ylabel('ratio')
    plt.grid()
    plt.legend()
    plt.show()
    