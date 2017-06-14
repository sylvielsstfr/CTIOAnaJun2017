import sys, os, re
import pyfits
import ccdproc
import matplotlib.pyplot as plt
from tools import *
import commands

import overscan_subtract_andTrim as ovsubtrim

def overscan_and_trim_allimages(files,outputdir):
    for f in files :
        ovsubtrim.Do_overscan_subtract_andTrim(f)
    commands.getoutput('mv *.fits '+outputdir)


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


def remove_bad_pixels(image):
    bad_pixels = np.where(image==0)





outputdir="./trim_images"
ensure_dir(outputdir)

##### MAKE MASTER BIAS ######
rootpath_rawimage=['/Volumes/LACIE SHARE/data_04jun17']
outputfile="MasterBias_CTIO_20170604.fits"
MIN_IMGNUMBER=1
MAX_IMGNUMBER=20
SearchTagRe='20170604_([0-9]+).fits$'

filelist_fitsimages, indexes_files = MakeFileList(rootpath_rawimage,SearchTagRe,MIN_IMGNUMBER, MAX_IMGNUMBER)
print 'Number of files :',len(filelist_fitsimages)
print filelist_fitsimages[:5]

overscan_and_trim_allimages(filelist_fitsimages,outputdir)

object_name='Master Bias June 4th 2017'
SelectTagRe='^trim_20170604_([0-9]+).fits$' # regular expression to select the file
NBIMGPERLROW=4

filelist_fitsimages, indexes_files = MakeFileList([outputdir],SearchTagRe,MIN_IMGNUMBER, MAX_IMGNUMBER)
print 'Number of files :',len(filelist_fitsimages)
print filelist_fitsimages[:5]


all_dates, all_airmass, all_images, all_titles, all_header  = BuildRawImages(filelist_fitsimages)

NBIMAGES=len(all_images)
all_rawflat = []
for index in np.arange(0,NBIMAGES):
    rawflat=ccdproc.CCDData(all_images[index],unit='adu')
    all_rawflat.append(rawflat)

all_flat_bias_subtracted = []
for flat in all_rawflat:
    flat_bias_subtracted = ccdproc.subtract_bias(flat,master_bias,add_keyword={'calib': 'subtracted bias'})
    all_flat_bias_subtracted.append(flat_bias_subtracted)

    
#master_bias = pyfits.getdata('../ana_04jun17/BuildMasterBias/MasterBias_CTIO_20170604.fits')
#master_bias = ccdproc.CCDData(master_bias,unit='adu') 

#master_flat = pyfits.getdata('../ana_04jun17/BuildMasterFlat_CB/MasterFlat_CTIO_20170530.fits')
#master_flat = ccdproc.CCDData(master_flat,unit='adu') 

raw_image = pyfits.getdata(' /Users/jneveu/Pictures/Chili/Astro/jneveu/sombrero004.fits')    
raw_image = ccdproc.CCDData(raw_image,unit='adu') 


image_bias_subtracted = ccdproc.subtract_bias(raw_image,master_bias,add_keyword={'calib': 'subtracted bias'})

image = ccdproc.flat_correct(image_bias_subtracted, master_flat)


plt.figure(figsize=(12, 12))
#plt.imshow(master_bias, vmax=bias_mean + bias_std, vmin=bias_mean - bias_std)
im=plt.imshow(image,origin='lower',cmap='rainbow',vmin=-5, vmax=5)
plt.grid(color='white', ls='solid')
plt.grid(True)
plt.show()

