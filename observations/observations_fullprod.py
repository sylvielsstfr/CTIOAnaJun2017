import matplotlib.pyplot as plt
import re,os
import pyfits
import pandas as pd
import numpy as np

import sys



  

#-------------------------------------------------------------------------------
def MakeFileList(dirlist,filetag,min_imgnumber,max_imgnumber):
    """
    MakeFileList : Make The List of files to open
    =============
    
    - we select the files which are of interest.
    - In particular select the number range
    
    """
    filelist_fitsimages = []
    indices = []
    count=0
    for d in dirlist: # loop on directories, one per image   (if necessary)    
        listfiles=os.listdir(d) # build the name of leaf directory
        for filename in listfiles:
            #print filename
            if re.search(filetag,filename):  #example of filename filter
                str_index=re.findall(filetag,filename)
                count=count+1
                index=int(str_index[0])
                if index >= min_imgnumber and index <= max_imgnumber:
                    #print 'OK'
                    indices.append(index)         
                    shortfilename=d+'/'+filename
                    filelist_fitsimages.append(shortfilename)
    return filelist_fitsimages, indices
#----------------------------------------------------------------------------------

def BuildHeaderInfo(filenames):
    """
    BuildRawImages
    ===============
    """

    all_dates= []
    all_airmass = []
    
    
    
    all_exposures = []
    all_ut = []
    all_ra = []
    all_dec = []
    all_epoch = []
    all_zenith = []
    all_ha = []
    all_st = []
    all_alt = []
    all_focus = []
    all_temp = []
    all_press = []
    all_hum = []
    all_windsp = []
    all_seeing = []
    all_seeingam = []
    all_filter1 = []
    all_filter2 = []
   
    all_obj = []

  

    for idx,f in np.ndenumerate(filenames):  
        

        
        hdu_list=pyfits.open(f)
        
        header=hdu_list[0].header
        
        date_obs = header['DATE-OBS']
        airmass = header['AIRMASS']
        expo= float(header['EXPTIME'])
        ut=header['UT']
        ra=header['RA']
        dec=header['DEC']
        epoch=float(header['EPOCH'])
        zd = float(header['ZD'])
        ha = header['HA']
        st = header['ST']
        alt = float(header['ALT'])
        fcl = float(header['TELFOCUS'])
        obj= header['OBJECT']
        filter1 = header['FILTER1']
        filter2 = header['FILTER2']
        
        # Missing info 
        if re.search('data_04jun17/20170604',f):
            temp=0
            press=0
            windsp=0
            rhumid=0
            seeing=0
            seeingam=0
            
        
        else:
            temp= float(header['OUTTEMP'])
            press= float(header['OUTPRESS'])
            rhumid= float(header['OUTHUM'])
            windsp=float(header['WNDSPEED'])       
            seeing=float(header['SEEING'])
            seeingam=float(header['SAIRMASS'])
        
        
        
    
        all_dates.append(date_obs)
        all_airmass.append(airmass)
        all_obj.append(obj)
        all_exposures.append(expo)
        all_ut.append(ut)
        all_ra.append(ra)
        all_dec.append(dec)
        all_epoch.append(epoch)
        all_zenith.append(zd)
        all_ha.append(ha)
        all_st.append(st)
        all_alt.append(alt)
        all_focus.append(fcl)
        all_temp.append(temp)
        all_press.append(press)
        all_hum.append(rhumid)
        all_windsp.append(windsp)
        all_seeing.append(seeing)
        all_seeingam.append(seeingam)
        all_filter1.append(filter1)
        all_filter2.append(filter2)
       

        
        hdu_list.close()
        
        
    return all_dates, all_airmass,all_obj,all_exposures,all_ut,all_ra,all_dec,all_epoch,all_zenith,all_ha,all_st,all_alt,all_focus,all_temp,all_press,all_hum,all_windsp,all_seeing,all_seeingam,all_filter1,all_filter2
#----------------------------------------------------------------------------------


#--------------------------------------------------------------------------------


top_input_rawimage='/Volumes/LaCie2/CTIODataJune2017'


# put which subdirs to which perform overscan and trim

# remove 4th jun and 13 jun
subdirs=['data_26may17','data_28may17', 'data_29may17','data_30may17', 'data_31may17',
         'data_01jun17','data_02jun17','data_03jun17','data_05jun17',
         'data_08jun17','data_09jun17','data_10jun17','data_12jun17','data_13jun17']

#subdirs=['data_26may17','data_28may17', 'data_29may17','data_30may17', 'data_31may17',
#         'data_01jun17','data_02jun17','data_03jun17','data_04jun17','data_05jun17',
#         'data_08jun17','data_09jun17','data_10jun17','data_12jun17','data_13jun17']


#subdirs=['data_04jun17']

NBNIGHTS=len(subdirs)         
		

MIN_IMGNUMBER=1
MAX_IMGNUMBER=500
SearchTagRe='^2017[0-9]+_([0-9]+).fits$'


if __name__ == '__main__':
    
    NumberOfFiles=0
    night_index=0
    
  
    
    for subdir in subdirs:
        
        print '==============================================================='
        print subdir
        print '==============================================================='
        
        inputdir=os.path.join(top_input_rawimage,subdir)
        

        filelist_fitsimages, indexes_files = MakeFileList([inputdir],SearchTagRe,MIN_IMGNUMBER, MAX_IMGNUMBER)
        
       
        
        (all_dates,all_airmass,all_obj,all_exposures,all_ut,all_ra,all_dec,all_epoch,all_zenith,all_ha,all_st,all_alt,all_focus,all_temp,all_press,all_hum,all_windsp,all_seeing,all_seeingam,all_filter1,all_filter2)=BuildHeaderInfo(filelist_fitsimages) 
        
        dd = {'date': all_dates, 'object': all_obj,'airmass':all_airmass,'seeing':all_seeing,'filter':all_filter1,'disperser':all_filter2}
        
        df=pd.DataFrame(dd)
        
        print df.describe()
        
        
        if night_index==0:
            all_data=df
        else:
            all_data=all_data.append(df,ignore_index=True)
        
        NumberOfFiles+=len(filelist_fitsimages)
        
        night_index+=1
        
     
        
        
        
    print 'Total Number Of Fits Files = ', NumberOfFiles
    
    print all_data.describe()
    all_data.to_csv('ctioobsinfo_jun2017.csv')
    
    
    all_data.plot('date','airmass',figsize=(20,6),rot=45,grid=True,title='airmasses vs date',color='b',marker='o',linewidth=2)
    plt.savefig('am.pdf') 
    
    
    
    