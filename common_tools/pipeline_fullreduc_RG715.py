import sys, os, re
import pyfits
import ccdproc
from ccdproc import CCDData, Combiner
import matplotlib.pyplot as plt
from tools import *
import commands
import os
from astropy.io import fits
from astropy import units as u
import numpy as np
import overscan_subtract_andTrim as ovsubtrim
import bottleneck as bn  # numpy's masked median is slow...really slow (in version 1.8.1 and lower)
print 'bottleneck version',bn.__version__


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
def SortFileList(indexes,filelist):
    
    # transform in np array
    indexes=np.array(indexes)
    filelist=np.array(filelist)    


    sorted_indexes=np.argsort(indexes) # sort the file indexes
    sorted_files=filelist[sorted_indexes]
    sorted_numbers=indexes[sorted_indexes]

    return sorted_files,sorted_numbers
    
    
#------------------------------------------------------------------------------------------

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
        
        
        hdu_list=fits.open(f)
        header=hdu_list[0].header
        #hdu_list=pyfits.open(f)
        #header=hdu_list[0].header
        
        date_obs = header['DATE-OBS']
        airmass = header['AIRMASS']
        num=sorted_numbers[idx[0]]
        object_name=header['OBJECT']
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

imstats = lambda dat: (dat.min(), dat.max(), dat.mean(), dat.std())
scaling_func= lambda arr: 1/np.ma.median(arr)


#--------------------------------------------------------------------------------
def bn_median(masked_array, axis=None):
    """
    Perform fast median on masked array
    
    Parameters
    ----------
    
    masked_array : `numpy.ma.masked_array`
        Array of which to find the median.
    
    axis : int, optional
        Axis along which to perform the median. Default is to find the median of
        the flattened array.
    """
    data = masked_array.filled(fill_value=np.NaN)
    med = bn.nanmedian(data, axis=axis)
    # construct a masked array result, setting the mask from any NaN entries
    return np.ma.array(med, mask=np.isnan(med))
#-----------------------------------------------------------------------------
def avg_over_images(masked_arr, axis=0):
    """
    Calculate average pixel value along specified axis
    """
    return ma.mean(masked_arr, axis=axis)

def med_over_images(masked_arr, axis=0):
    """
    Calculate median pixel value along specified axis
    
    Uses bottleneck.nanmedian for speed
    """
    
    dat = masked_arr.data.copy()
    dat[masked_arr.mask] = np.NaN
    return bn.nanmedian(dat, axis=axis)
#---------------------------------------------------------------------



top_datapath='/sps/lsst/data/AtmosphericCalibration/'
top_datapath='.'

top_input_trimimage=top_datapath+'CTIODataJune2017_ovsctrim_v2'
top_output_reducimage='./CTIODataJune2017_reduced_RG715_v2'

# do the processing in /scratch/dagoret on cca002.in2p3.fr

master_bias_file=top_datapath+'../'+'Flats_et_bias/FinalMasterBias_CTIO.fits'
master_flat_file=top_datapath+'../'+'Flats_et_bias/FinalMasterFlat_CTIO_RG715.fits'


# put which subdirs to which perform overscan and trim

#subdirs=['data_26may17','data_28may17', 'data_29may17','data_30may17', 'data_31may17',
#         'data_01jun17','data_02jun17','data_03jun17','data_04jun17','data_05jun17',
#         'data_06jun17','data_08jun17','data_09jun17','data_10jun17','data_12jun17','data_13jun17']

subdirs=['data_26may17','data_28may17', 'data_29may17','data_30may17', 'data_31may17',
         'data_01jun17','data_02jun17','data_03jun17','data_04jun17','data_05jun17',
         'data_06jun17','data_08jun17','data_09jun17','data_10jun17','data_12jun17','data_13jun17']
 
 
#subdirs=['data_31may17']
         
		

MIN_IMGNUMBER=1
MAX_IMGNUMBER=1000
SearchTagRe='^trim_2017[0-9]+_([0-9]+).fits$'


if __name__ == '__main__':
    
    
    master_bias = ccdproc.CCDData.read(master_bias_file,unit=u.adu)
    master_flat = ccdproc.CCDData.read(master_flat_file,unit=u.adu)
    
    print 'stat on bias : min max mean std' , imstats(np.asarray(master_bias))
    print 'stat on flat : min max mean std' , imstats(np.asarray(master_flat))
    
    
    flat_min, flat_max, flat_mean, flat_std = imstats(np.asarray(master_flat))
    
    
    if flat_mean < 0.9 or flat_mean > 1.1:
        
        flat_combiner = Combiner([master_flat])
        flat_combiner.sigma_clipping(func=med_over_images)
        flat_combiner.scaling = scaling_func
        master_flat = flat_combiner.median_combine(median_func=bn_median)
        
        print 'stat on normalized flat min max mean std ' , imstats(np.asarray(master_flat))
        
    
    NumberOfFiles=0
    
    for subdir in subdirs:
        
        print subdir
        
        inputdir=os.path.join(top_input_trimimage,subdir)
        outputdir=os.path.join(top_output_reducimage,subdir)

        print 'inputdir=',inputdir,'  outputdir=',outputdir

        ensure_dir(outputdir)
        filelist_fitsimages, indexes_files = MakeFileList([inputdir],SearchTagRe,MIN_IMGNUMBER, MAX_IMGNUMBER)
        
        sorted_files,sorted_numbers=SortFileList(indexes_files,filelist_fitsimages)
        
        
        all_dates, all_airmass, all_images, all_titles, all_header=BuildRawImages(sorted_files)
        
        
        print "put images in CCDPROC"
        NBIMAGES=len(all_images)
       
        all_rawimage = []
        for index in np.arange(0,NBIMAGES):
            rawccd=ccdproc.CCDData(all_images[index],unit='adu')
            all_rawimage.append(rawccd)
           
        
        print "subtract bias"
        #subtract the masterbias
        all_bias_subtracted = []
        id=0
        for raw_image in all_rawimage:
            bias_subtracted = ccdproc.subtract_bias(raw_image, master_bias)
            all_bias_subtracted.append(bias_subtracted)
            #sprint id
            id+=1
        
        
        # normalize by the masterflat
        print "normalize with flat"
        all_reduced = []
        id=0
        for bias_sub in all_bias_subtracted:
            reduced_image = ccdproc.flat_correct(bias_sub, master_flat,min_value=0.9)
            all_reduced.append(reduced_image)
            #print id
            id+=1
        
        
        
        
        
        NumberOfFiles+=len(filelist_fitsimages)
        
        # generate output filename
        #--------------------------
        newfullfilenames=[]
        for idx,file in np.ndenumerate(sorted_files):
    
            short_infilename=os.path.basename(file)
            short_partfilename=re.findall('^trim_(.*)',short_infilename)
            short_outfilename='reduc_'+short_partfilename[0]
            newfullfilename=os.path.join(outputdir,short_outfilename)
            newfullfilenames.append(newfullfilename)
        
        
        # Save file in output
        #--------------------
        for idx,file in np.ndenumerate(newfullfilenames):
            idex=idx[0]
            #print idex,"  ",file
            #print all_header[idex]
            prihdu = fits.PrimaryHDU(header=all_header[idex],data=all_reduced[idex])
            thdulist = fits.HDUList(prihdu)
            thdulist.writeto(file,overwrite=True)
        
        
    print 'Total Number Of Fits Files = ', NumberOfFiles
