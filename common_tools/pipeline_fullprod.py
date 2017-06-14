import sys, os, re
import pyfits
import ccdproc
import matplotlib.pyplot as plt
from tools import *
import commands

import overscan_subtract_andTrim as ovsubtrim

#---------------------------------------------------------------------------------
def overscan_and_trim_allimages(files,outputdir):
    for f in files :
        ovsubtrim.Do_overscan_subtract_andTrim(f)
    commands.getoutput('mv *.fits '+outputdir)
#-------------------------------------------------------------------------------    

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

def BuildRawImages(filenames):
    """
    BuildRawImages
    ===============
    """

    all_dates= []
    all_airmass = []
    all_images = []
    all_titles = []
    all_header = []

    for idx,f in np.ndenumerate(filenames):   
        hdu_list=pyfits.open(f)
        header=hdu_list[0].header
        date_obs = header['DATE-OBS']
        airmass = header['AIRMASS']
        num=sorted_numbers[idx[0]]
        title=object_name+" z= {:3.2f} Nb={}".format(float(airmass),num)
        image_corr=hdu_list[0].data
        image=image_corr
        all_dates.append(date_obs)
        all_airmass.append(float(airmass))
        all_images.append(image)
        all_titles.append(title)
        all_header.append(header)
        hdu_list.close()
    return all_dates, all_airmass, all_images, all_titles, all_header
#----------------------------------------------------------------------------------

def remove_bad_pixels(image):
    bad_pixels = np.where(image==0)
#--------------------------------------------------------------------------------


top_input_rawimage='/Users/dagoret-campagnesylvie/MacOsX/LSST/MyWork/GitHub/CTIODataJune2017'
top_output_trimimage='/Users/dagoret-campagnesylvie/MacOsX/LSST/MyWork/GitHub/CTIODataJune2017_ovsctrim'

# put which subdirs to which perform overscan and trim

subdirs=['data_26may17','data_28may17', 'data_29may17','data_30may17', 'data_31may17',
         'data_01jun17','data_02jun17','data_03jun17','data_04jun17','data_05jun17',
         'data_08jun17','data_09jun17','data_10jun17','data_12jun17','data_13jun17']
         
		

MIN_IMGNUMBER=1
MAX_IMGNUMBER=500
SearchTagRe='^2017[0-9]+_([0-9]+).fits$'


if __name__ == '__main__':
    
    NumberOfFiles=0
    
    for subdir in subdirs:
        
        print subdir
        
        inputdir=os.path.join(top_input_rawimage,subdir)
        outputdir=os.path.join(top_output_trimimage,subdir)

        print 'inputdir=',inputdir,'  outputdir=',outputdir

        ensure_dir(outputdir)
        filelist_fitsimages, indexes_files = MakeFileList([inputdir],SearchTagRe,MIN_IMGNUMBER, MAX_IMGNUMBER)
        
        NumberOfFiles+=len(filelist_fitsimages)
        
        overscan_and_trim_allimages(filelist_fitsimages,outputdir)
        
        
        
    print 'Total Number Of Fits Files = ', NumberOfFiles