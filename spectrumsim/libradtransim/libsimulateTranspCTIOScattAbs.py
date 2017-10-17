################################################################
#
# Script to simulate air transparency with LibRadTran
# With scattering and absorption
# Tune CTIO astmosphere as wished
# author: sylvielsstfr
# creation date : June 27th  2017
# 
#
#################################################################
import os
import re
import math
import numpy as np
import pandas as pd
from astropy.io import fits
import sys,getopt

import UVspec



# Definitions and configuration
#-------------------------------------

# LibRadTran installation directory
home = os.environ['HOME']+'/'
libradtranpath = os.environ['LIBRADTRANDIR']+'/'       
#libradtranpath = home+'MacOSX/External/libRadtran/libRadtran-2.0.1/'

# Filename : RT_LS_pp_us_sa_rt_z15_wv030_oz30.txt
#          : Prog_Obs_Rte_Atm_proc_Mod_zXX_wv_XX_oz_XX
  
Prog='RT'  #definition the simulation programm is libRadTran
Obs='CT'   # definition of observatory site (LS,CT,OH,MK,...)
Rte='pp'   # pp for parallel plane of ps for pseudo-spherical
Atm=['us']   # short name of atmospheric sky here US standard and  Subarctic winter
Proc='sa'  # light interaction processes : sc for pure scattering,ab for pure absorption
           # sa for scattering and absorption, ae with aerosols default, as with aerosol special
Mod='rt'   # Models for absorption bands : rt for REPTRAN, lt for LOWTRAN, k2 for Kato2
ZXX='z'        # XX index for airmass z :   XX=int(10*z)
WVXX='wv'      # XX index for PWV       :   XX=int(pwv*10)
OZXX='oz'      # XX index for OZ        :   XX=int(oz/10)



CTIO_Altitude = 2.200  # in k meters from astropy package (Cerro Pachon)
OBS_Altitude = str(CTIO_Altitude)

TOPDIR='simulations/RT/2.0.1/CT'



############################################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
#########################################################################


def usage():
    print "*******************************************************************"
    print sys.argv[0],' -z <airmass> -w <pwv> -o <oz> -p <P>'
    print ' \t - airmass from 1.0 to 3.0, typical z=1 '
    print ' \t - pwv is precipitable watr vapor in kg per m2 or mm, typical pwv = 5.18 mm'
    print ' \t - Pressure in hPa, typical P=775.3 hPa  '
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    
    
    print "*******************************************************************"
    
    

 
#-----------------------------------------------------------------------------


def ProcessSimulation(airmass_num,pwv_num,oz_num,press_num):    
    
    
    print '--------------------------------------------'
    print ' 2) airmass = ', airmass_num
    print ' 2) pwv = ', pwv_num
    print ' 3) oz = ', oz_num
    print ' 4) pressure  = ',press_num
    print '--------------------------------------------'    
   
    
    ensure_dir(TOPDIR)

    
    # build the part 1 of filename
    BaseFilename_part1=Prog+'_'+Obs+'_'+Rte+'_'
    

    # Set up type of run
    runtype='clearsky' #'no_scattering' #aerosol_special #aerosol_default# #'clearsky'#     
    if Proc == 'sc':
        runtype='no_absorption'
        outtext='no_absorption'
    elif Proc == 'ab':
        runtype='no_scattering'
        outtext='no_scattering'
    elif Proc == 'sa':
        runtype=='clearsky'
        outtext='clearsky'
    elif Proc == 'ae':   
        runtype='aerosol_default'
        outtext='aerosol_default'
    elif Proc == 'as':   
        runtype='aerosol_special'
        outtext='aerosol_special'
    else:
        runtype=='clearsky'
        outtext='clearsky'

#   Selection of RTE equation solver        
    if Rte == 'pp': # parallel plan
        rte_eq='disort'
    elif Rte=='ps':   # pseudo spherical
        rte_eq='sdisort'
        
 
#   Selection of absorption model 
    molmodel='reptran'
    if Mod == 'rt':
        molmodel='reptran'
    if Mod == 'lt':
        molmodel='lowtran'
    if Mod == 'kt':
        molmodel='kato'
    if Mod == 'k2':
        molmodel='kato2'
    if Mod == 'fu':
        molmodel='fu'    
    if Mod == 'cr':
        molmodel='crs'     
               


    	  
    # for simulation select only two atmosphere   
    #theatmospheres = np.array(['afglus','afglms','afglmw','afglt','afglss','afglsw'])
    atmosphere_map=dict()  # map atmospheric names to short names 
    atmosphere_map['afglus']='us'
    atmosphere_map['afglms']='ms'
    atmosphere_map['afglmw']='mw'  
    atmosphere_map['afglt']='tp'  
    atmosphere_map['afglss']='ss'  
    atmosphere_map['afglsw']='sw'  
      
    theatmospheres= []
    for skyindex in Atm:
        if re.search('us',skyindex):
            theatmospheres.append('afglus')
        if re.search('sw',skyindex):
            theatmospheres.append('afglsw')
            
   
   

    # 1) LOOP ON ATMOSPHERE
    for atmosphere in theatmospheres:
        #if atmosphere != 'afglus':  # just take us standard sky
        #    break
        atmkey=atmosphere_map[atmosphere]
       
        # manage input and output directories and vary the ozone
        TOPDIR2=TOPDIR+'/'+Rte+'/'+atmkey+'/'+Proc+'/'+Mod
        ensure_dir(TOPDIR2)
        INPUTDIR=TOPDIR2+'/'+'in'
        ensure_dir(INPUTDIR)
        OUTPUTDIR=TOPDIR2+'/'+'out'
        ensure_dir(OUTPUTDIR)
    
    
        # loop on molecular model resolution
        #molecularresolution = np.array(['COARSE','MEDIUM','FINE']) 
        # select only COARSE Model
        molecularresolution = np.array(['COARSE'])    
        for molres in molecularresolution:
            if molres=='COARSE':
                molresol ='coarse'
            elif molres=='MEDIUM':
                molresol ='medium'
            else:
                molresol ='fine'
         
            
         
        
        #water vapor   
        pwv_val=pwv_num
        pwv_str='H2O '+str(pwv_val)+ ' MM'
        wvfileindex=int(10*pwv_val)
           
           
        # airmass
        airmass=airmass_num
        amfileindex=int(airmass_num*10)
        
        # Ozone    
        oz_val=oz_num
        oz_str='O3 '+str(oz_num)+ ' DU'
        ozfileindex=int(oz_val/10.)
        
            
        BaseFilename=BaseFilename_part1+atmkey+'_'+Proc+'_'+Mod+'_z'+str(amfileindex)+'_'+WVXX+str(wvfileindex) +'_'+OZXX+str(ozfileindex)                   
                    
        verbose=True
        uvspec = UVspec.UVspec()
        uvspec.inp["data_files_path"]  =  libradtranpath+'data'
                
        uvspec.inp["atmosphere_file"] = libradtranpath+'data/atmmod/'+atmosphere+'.dat'
        uvspec.inp["albedo"]           = '0.2'
    
        uvspec.inp["rte_solver"] = rte_eq
            
            
                
        if Mod == 'rt':
            uvspec.inp["mol_abs_param"] = molmodel + ' ' + molresol
        else:
            uvspec.inp["mol_abs_param"] = molmodel

        # Convert airmass into zenith angle 
        am=airmass
        sza=math.acos(1./am)*180./math.pi

        # Should be no_absorption
        if runtype=='aerosol_default':
            uvspec.inp["aerosol_default"] = ''
        elif runtype=='aerosol_special':
            uvspec.inp["aerosol_set_tau_at_wvl"] = '500 0.02'
                        
        if runtype=='no_scattering':
            uvspec.inp["no_scattering"] = ''
        if runtype=='no_absorption':
            uvspec.inp["no_absorption"] = ''
     
        # set up the ozone value               
        uvspec.inp["mol_modify"] = pwv_str
        uvspec.inp["mol_modify2"] = oz_str
        
        
        # rescale pressure   if reasonable pressure values are provided
        if press_num>600. and press_num<1015.:
            uvspec.inp["pressure"] = press_num
                    
                
        uvspec.inp["output_user"] = 'lambda edir'
        uvspec.inp["altitude"] = OBS_Altitude   # Altitude LSST observatory
        uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_1.0nm.dat'
        #uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_0.1nm.dat'
        uvspec.inp["sza"]        = str(sza)
        uvspec.inp["phi0"]       = '0'
        uvspec.inp["wavelength"]       = '250.0 1200.0'
        uvspec.inp["output_quantity"] = 'reflectivity' #'transmittance' #
#       uvspec.inp["verbose"] = ''
        uvspec.inp["quiet"] = ''

  

        if "output_quantity" in uvspec.inp.keys():
            outtextfinal=outtext+'_'+uvspec.inp["output_quantity"]

           
            
        inputFilename=BaseFilename+'.INP'
        outputFilename=BaseFilename+'.OUT'
        inp=os.path.join(INPUTDIR,inputFilename)
        out=os.path.join(OUTPUTDIR,outputFilename)
                    
            
        uvspec.write_input(inp)
        uvspec.run(inp,out,verbose,path=libradtranpath)
        
        
    return OUTPUTDIR,outputFilename

#---------------------------------------------------------------------------
#####################################################################
# The program simulation start here
#
####################################################################
#
# Typical values at ground ::  
# P0= 775.28625 hPa  
# T0= 273.9 K  
# Z0= 2.2 km 
#
# At ground :
#
# Air = 2.05022e+19 molecules / cm3  
# H2O =  8.90711e+16 molecules / cm3
# M_H2O_0= 0.002664675 kg / m3 humidite absolue dans LibRadTran
# M_H20_0= 0.0025525645 kg/m3 calcule par la formule de l humidite absolue 
#                         pour 50 pourcent  d humidite relative, a P0 et T0
# PWV= 5.180 kg/m2   dans LibRadTran
#########################################################################


if __name__ == "__main__":
    
    
    airmass_str=""
    pwv_str=""
    oz_str=""
    press_str=""
    
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hz:w:o:p:",["z=","w=","o=","p="])
    except getopt.GetoptError:
        print ' Exception bad getopt with :: '+sys.argv[0]+ ' -z <airmass> -w <pwv> -o <oz> -p <pr>'
        print ' - pwv in kg / m2 or mm '
        print ' - oz in DbU '
        print ' - P in hPa '
        sys.exit(2)
        
        
        
    #print 'opts = ',opts
    #print 'args = ',args    
        
        
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-z", "--airmass"):
            airmass_str = arg
        elif opt in ("-w", "--pwv"):
            pwv_str = arg
        elif opt in ("-o", "--oz"):
            oz_str = arg  
        elif opt in ("-p", "--pr"):
            press_str = arg      
        else:
            print 'Do not understand arguments : ',argv
            
         
    print '--------------------------------------------'     
    print '1) airmass = ', airmass_str
    print '2) pwv = ', pwv_str
    print "3) oz = ", oz_str
    print "4) pr = ", press_str
    print '--------------------------------------------' 

    if airmass_str=="":
        usage()
        sys.exit()

    if pwv_str=="":
        usage()
        sys.exit()

    if oz_str=="":
        usage()
        sys.exit()
        
    
    if press_str=="":
        usage()
        sys.exit()
        
	
	
    airmass_nb=float(airmass_str)
    pwv_nb=float(pwv_str)
    oz_nb=float(oz_str)	
    pr_nb=float(press_str)	
    
    if airmass_nb<1 or airmass_nb >3 :
        print "bad airmass value z=",airmass_nb
        sys.exit()
        
    if pwv_nb<0 or pwv_nb >50 :
        print "bad PWV value pwv=",pwv_nb
        sys.exit()
        
    if oz_nb<0 or oz_nb >600 :
        print "bad Ozone value oz=",oz_nb
        sys.exit()
        
    if pr_nb<0 or pr_nb >1015 :
        print "bad Pressure value pr=",pr_nb
        sys.exit()
        
    # do the simulation now    
    
    path, outputfile=ProcessSimulation(airmass_nb,pwv_nb,oz_nb,pr_nb)
    
    print '*****************************************************'
    print ' path       = ', path
    print ' outputfile =  ', outputfile 
    print '*****************************************************'
       
   
