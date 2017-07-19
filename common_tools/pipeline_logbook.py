import re,os
import pyfits
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
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

def CheckKey(thekey,header,allkeys,filename):
    
    shortfile=os.path.basename(filename)
    
    value=-3
    
    if re.search('20170604',filename) or re.search('20170613',filename) : # bad files
        #print " >>>>>>>  filename {}  has allkeys= {} ".format(filename,allkeys)
        if re.search('OUTTEMP',thekey) or re.search('OUTPRESS',thekey) or re.search('OUTHUM',thekey) or re.search('WNDSPEED',thekey): # bad key
            print " ++++++  Bad key {} skipped for file {}".format(thekey,shortfile)
            value=-4
            return value
        else: #other keys
            if thekey in allkeys:
                value=float(header[thekey])
            else:
                print '>>>>>>    Missing Key ',thekey, ' in file ',shortfile
                value=-2.
    else: # Good files
        if thekey in allkeys:
            value=float(header[thekey])
        else:
            print '>>>>>>    Missing Key ',thekey, ' in file ',shortfile
            value=-1.
    return value

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
        
        allkeys=header.keys()
       
        
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
        
        #sometimes these datacard are missing
        seeing=CheckKey('SEEING',header,allkeys,f)
        temp=CheckKey('OUTTEMP',header,allkeys,f)
        press=CheckKey('OUTPRESS',header,allkeys,f)
        rhumid=CheckKey('OUTHUM',header,allkeys,f)
        windsp=CheckKey('WNDSPEED',header,allkeys,f)
        seeingam=CheckKey('SAIRMASS',header,allkeys,f)
       
        
        # fill data
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

#--------------------------------------------------------------------------------




if __name__ == '__main__':
    
    # CCIN2P3 disk mounted through network using sshfs
    top_input_images='/Users/dagoret/MacOSX/data/AtmosphericCalibration/CTIODataJune2017_ovsctrim'

    # put which subdirs to which perform overscan and trim

    subdirs=['data_26may17','data_28may17', 'data_29may17','data_30may17', 'data_31may17',
         'data_01jun17','data_02jun17','data_03jun17','data_04jun17','data_05jun17',
         'data_06jun17','data_08jun17','data_09jun17','data_10jun17','data_12jun17','data_13jun17']

    #subdirs=['data_26may17','data_28may17','data_04jun17','data_13jun17']
		

    MIN_IMGNUMBER=1
    MAX_IMGNUMBER=1000
    SearchTagRe='^trim_2017[0-9]+_([0-9]+).fits$'
    
    NumberOfFiles=0
    night_index=0
    
  
    all_df= []
    
    # loop on subdirectories
    for subdir in subdirs:
        
        print '==============================================================='
        print subdir
        print '==============================================================='
        
        inputdir=os.path.join(top_input_images,subdir)
        
        #print inputdir
        
        # get the list of fits file in the directory
        filelist_fitsimages, indexes_files = MakeFileList([inputdir],SearchTagRe,MIN_IMGNUMBER, MAX_IMGNUMBER)
        
        filelist_shortfilename=[os.path.basename(path) for path in filelist_fitsimages]
        
        
        # extract info from header
        (all_dates,all_airmass,all_obj,all_exposures,all_ut,all_ra,all_dec,all_epoch,all_zenith,all_ha,all_st,all_alt,all_focus,all_temp,all_press,all_hum,all_windsp,all_seeing,all_seeingam,all_filter1,all_filter2)=BuildHeaderInfo(filelist_fitsimages) 
        
        
        all_days=np.repeat(subdir,len(all_dates))
        
        
        # create the dictionnary
        dd = {'date': all_dates, 
              'subdir':all_days,
              'index':indexes_files,
              'object': all_obj,
              'airmass':all_airmass,
              'seeing':all_seeing,
              'filter':all_filter1,
              'disperser':all_filter2,
              'exposure':all_exposures,
              'focus':all_focus,
              'file':filelist_shortfilename,
              'P':all_press,
              'T':all_temp,
              'RH':all_hum,
              'W':all_windsp
              }
        # create the dataframe from the dictionnary
        df=pd.DataFrame(dd)
        df=df.sort_values(by='index', ascending=1)
        
        # append in the list the dataframe
        all_df.append(df)
        
        NumberOfFiles+=len(filelist_fitsimages)
        night_index+=1
        
     
    all_data=pd.concat(all_df)    
    
    # order the columns
    all_data = all_data.reindex_axis(['date','subdir','index','object','filter','disperser','airmass','exposure','focus','seeing','P','T','RH','W','file'], axis=1)
        
    print '==============================================='    
    print 'Total Number Of Fits Files = ', NumberOfFiles
    print '==============================================='
    
    
    # save in CSV format
    print all_data
    print all_data.describe()
    
    all_data.to_csv('ctiofulllogbook_jun2017.csv')
    
    #save in Excel
    all_data.to_excel('ctiofulllogbook_jun2017.xlsx')
    
    # convert to astropy Table
    t = Table.from_pandas(all_data)
    t.write('ctiofulllogbook_jun2017.fits',format='fits',overwrite=True)
    
    
  