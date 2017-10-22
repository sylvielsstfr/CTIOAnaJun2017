import re, os, sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from astropy.modeling import models
from astropy import units as u
from astropy import nddata
from astropy.io import fits
from astropy.modeling import models
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.modeling import models, fitting

import ccdproc

from scipy import stats  
from scipy import ndimage
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
from IPython.display import Image

import bottleneck as bn  # numpy's masked median is slow...really slow (in version 1.8.1 and lower)

import photutils
from photutils import daofind
from photutils import CircularAperture
#from photutils.background import Background2D
from photutils import Background2D, SigmaClip, MedianBackground

from skimage.feature import hessian_matrix

from tools import *
from scan_holo import *
from targets import *
from fwhm_profiles import *

import math as m

# lib about pdf
from matplotlib.backends.backend_pdf import PdfPages 
# libs about time handling
import datetime
from datetime import timedelta
from dateutil import parser  # very usefull time format smart parser

 
import matplotlib as mpl 
from matplotlib.dates import MonthLocator, WeekdayLocator,DateFormatter
from matplotlib.dates import MONDAY
mondays = WeekdayLocator(MONDAY)
months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
monthsFmt = DateFormatter("%b '%y")
import matplotlib.dates as mdates
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
days=mdates.DayLocator()  # every day
hour=mdates.HourLocator()  # every day
yearsFmt = mdates.DateFormatter('%Y')


# Definitions of some constants
#------------------------------------------------------------------------------
Filt_names= ['dia Ron400', 'dia Thor300', 'dia HoloPhP', 'dia HoloPhAg', 'dia HoloAmAg', 'dia Ron200','Unknown']
Disp_names= ['Ron400', 'Thor300', 'HoloPhP', 'HoloPhAg', 'HoloAmAg', 'Ron200','Unknown']

#-------------------------------------------------------------------------------
def init_notebook():
    print 'ccdproc version',ccdproc.__version__
    print 'bottleneck version',bn.__version__
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)
    # to enlarge the sizes
    params = {'legend.fontsize': 'x-large',
         'figure.figsize': (15, 15),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)

    print os.getcwd()
#-------------------------------------------------------------------------------
class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)
#----------------------------------------------------------------------------------    
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])    
#-----------------------------------------------------------------------------------
def MakeFileList(dirlist_all,MIN_IMGNUMBER=0,MAX_IMGNUMBER=1e10,SelectTagRe='',SearchTagRe=''):
    """
    MakeFileList : Make The List of files to open
    =============
    
    - we select the files which are of interest.
    - In particular select the number range
    
    """
    count=0
    indexes_files= [] 
    filelist_fitsimages= []  

    for d in dirlist_all: # loop on directories, one per image   (if necessary)    
            dir_leaf= d # build the name of leaf directory
            listfiles=os.listdir(dir_leaf) 
            for filename in listfiles:
                if re.search(SearchTagRe,filename):  #example of filename filter
                    str_index=re.findall(SelectTagRe,filename)
                    count=count+1
                    index=int(str_index[0])
                    if index >= MIN_IMGNUMBER and index <= MAX_IMGNUMBER: 
                        indexes_files.append(index)         
                        shortfilename=dir_leaf+'/'+filename
                        filelist_fitsimages.append(shortfilename)

    indexes_files=np.array(indexes_files)
    filelist_fitsimages=np.array(filelist_fitsimages)
    sorted_indexes=np.argsort(indexes_files) # sort the file indexes
    sorted_numbers=indexes_files[sorted_indexes]
    #sorted_files= [filelist_fitsimages[index] for index in sorted_indexes] # sort files
    sorted_files=filelist_fitsimages[sorted_indexes]
                    
    return sorted_numbers,sorted_files
#-------------------------------------------------------------------------------

def BuildImages(sorted_filenames,sorted_numbers,object_name):
    """
    BuildRawImages
    ===============
    """

    
    all_dates = []
    all_airmass = []
    all_images = []
    all_titles = []
    all_header = []
    all_expo = []
    all_filt = []
    all_filt1 = []
    all_filt2 = []
   
    NBFILES=sorted_filenames.shape[0]

    for idx in range(NBFILES):  
        
        file=sorted_filenames[idx]    
        
        hdu_list=fits.open(file)
        header=hdu_list[0].header
        #print header
        date_obs = header['DATE-OBS']
        airmass = header['AIRMASS']
        expo = header['EXPTIME']
        filters = header['FILTERS']
        filter1 = header['FILTER1']
        filter2 = header['FILTER2']
        
    
        num=sorted_numbers[idx]
        title=object_name+" z= {:3.2f} Nb={}".format(float(airmass),num)
        image_corr=hdu_list[0].data
        image=image_corr
        
        all_dates.append(date_obs)
        all_airmass.append(float(airmass))
        all_images.append(image)
        all_titles.append(title)
        all_header.append(header)
        all_expo.append(expo)
        all_filt.append(filters)
        all_filt1.append(filter1)
        all_filt2.append(filter2)
        
        hdu_list.close()
        
    return all_dates,all_airmass,all_images,all_titles,all_header,all_expo,all_filt,all_filt1,all_filt2
#-------------------------------------------------------------------------------------------
    
def BuildRawSpec(sorted_filenames,sorted_numbers,object_name):
    """
    BuildRawSpec
    ===============
    """

    all_dates = []
    all_airmass = []
    all_leftspectra = []
    all_rightspectra = []
    all_totleftspectra = []
    all_totrightspectra = []
    all_titles = []
    all_header = []
    all_expo = []
    all_filt= []
    all_filt1 = []
    all_filt2 = []
    all_elecgain = []
   
    NBFILES=sorted_filenames.shape[0]

    for idx in range(NBFILES):  
        
        file=sorted_filenames[idx]    
        
        hdu_list=fits.open(file)
        header=hdu_list[0].header
        #print header
        date_obs = header['DATE-OBS']
        airmass = header['AIRMASS']
        expo = header['EXPTIME']
        num=sorted_numbers[idx]
        filters = header['FILTERS']
        filter1 = header['FILTER1']
        filter2 = header['FILTER2']
        
        
        title=object_name+" z= {:3.2f} Nb={}".format(float(airmass),num)
        gain = 0.25*(float(header['GTGAIN11'])+float(header['GTGAIN12'])+float(header['GTGAIN21'])+float(header['GTGAIN22']))
        
        # now reads the spectra
        
        table_data=hdu_list[1].data
        
        left_spectrum=table_data.field('RawLeftSpec')
        right_spectrum=table_data.field('RawRightSpec')
        tot_left_spectrum=table_data.field('TotLeftSpec')
        tot_right_spectrum=table_data.field('TotRightSpec')
        
        all_dates.append(date_obs)
        all_airmass.append(float(airmass))
        all_leftspectra.append(left_spectrum)
        all_rightspectra.append(right_spectrum)
        all_totleftspectra.append(tot_left_spectrum)
        all_totrightspectra.append(tot_right_spectrum)
        all_titles.append(title)
        all_header.append(header)
        all_expo.append(expo)
        all_filt.append(filters)
        all_filt1.append(filter1)
        all_filt2.append(filter2)
        all_elecgain.append(gain)
        hdu_list.close()
        
    return all_dates,all_airmass,all_titles,all_header,all_expo,all_leftspectra,all_rightspectra,all_totleftspectra,all_totrightspectra,all_filt,all_filt1,all_filt2,all_elecgain



#--------------------------------------------------------------------------------------------

def ShowImages(all_images,all_titles,all_filt,object_name,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowRawImages: Show the raw images without background subtraction
    ==============
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,5*MAXIMGROW))
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        if verbose : print 'Processing image %d...' % index        
        im=axarr[iy,ix].imshow(all_images[index][::downsampling,::downsampling],cmap='rainbow',vmin=vmin, vmax=vmax, aspect='auto',origin='lower')
        axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].grid(color='white', ls='solid')
        axarr[iy,ix].text(5.,5,all_filt[index],verticalalignment='bottom', horizontalalignment='left',color='yellow', fontweight='bold',fontsize=16)
        
        thetitle="{}".format(index)
        axarr[iy,ix].set_title(thetitle)
    
    f.colorbar(im, orientation="horizontal")
    title='Images of {}'.format(object_name)
    plt.suptitle(title,size=16)    
    
#--------------------------------------------------------------------------------------------------------------------------    



#--------------------------------------------------------------------------------------------------------------------------
def ShowImagesinPDF(all_images,all_titles,object_name,dir_top_img,all_filt,date ,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowRawImages: Show the raw images without background subtraction
    ==============
    write images in pdf
    """
    
     
   
    NBIMAGES=len(all_images)
    MAXIMGROW=max(2,m.ceil(NBIMAGES/NBIMGPERROW))
    
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,'intput_images.pdf')
    pp = PdfPages(figfilename) # create a pdf file
    
    
    title='Images of {}, date : {}'.format(object_name,date)
    
    #spec_index_min=100  # cut the left border
    #spec_index_max=1900 # cut the right border
    #star_halfwidth=70

    #thex0 = []  # container of central position
  
    #f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,4*MAXIMGROW))
    #f.tight_layout()
    for index in np.arange(0,NBIMAGES):
        
        
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(title,size=20)
            
        # index of image in the page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        
        #ix=index%NBIMGPERROW
        #iy=index/NBIMGPERROW
        image = all_images[index][:,0:right_edge]
        
        #xprofile=np.sum(image,axis=0)
        #x0=np.where(xprofile==xprofile.max())[0][0]
        #thex0.append(x0)
        
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        
        im=axarr[iy,ix].imshow(image,origin='lower',cmap='rainbow',vmin=vmin,vmax=vmax)
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        axarr[iy,ix].grid(color='white', ls='solid')
      
        #axarr[iy,ix].text(1000.,275.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='yellow', fontweight='bold',fontsize=16)
        
        # save a new page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f.colorbar(im, orientation="horizontal")
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
        
          
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()
  
    



#--------------------------------------------------------------------------------------------------------------------------
 
 
def ShowOneOrderinPDF(all_images,all_titles,thex0,they0,object_name,all_expo,dir_top_img,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowRawImages: Show the raw images without background subtraction
    ==============
    """
    
    NBIMAGES=len(all_images)
    MAXIMGROW=max(2,m.ceil(NBIMAGES/NBIMGPERROW))
    
    spec_index_min=100  # cut the left border
    spec_index_max=right_edge # cut the right border
    star_halfwidth=70
    
    
    
    figfilename=os.path.join(dir_top_img,figname)   
    title='Images of {}, date : {}'.format(object_name,date)
    
    
     # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    
    for index in np.arange(0,NBIMAGES):
        
        
        x0=int(thex0[index])
        y0=int(they0[index])
        
        
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(title,size=20)
            
        # index of image in the page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
                
        full_image=np.copy(all_images[index])
        
        
        if(all_expo[index]<=0 ): #special case of the first image
            reduc_image=full_image[y0-20:y0+20,x0+150:spec_index_max]  
        else:
            reduc_image=full_image[y0-20:y0+20,x0+150:spec_index_max]/all_expo[index] 
            
        themax=reduc_image.flatten().max()    
            
        X,Y=np.meshgrid(np.arange(0,reduc_image.shape[1]),np.arange(reduc_image.shape[0]))       
        im = axarr[iy,ix].pcolormesh(X,Y,reduc_image, cmap='rainbow',vmin=vmin,vmax=themax*0.5)    # pcolormesh is impracticalble in pdf file 
        
        
        #im = axarr[iy,ix].imshow(reduc_image,origin='lower', cmap='rainbow',vmin=0,vmax=100)    
        axarr[iy,ix].axis([X.min(), X.max(), Y.min(), Y.max()]); axarr[iy,ix].grid(True)
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
    
        axarr[iy,ix].grid(color='white', ls='solid')
        #axarr[iy,ix].text(700,2.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='yellow', fontweight='bold',fontsize=16)
        
        
        # save a new page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f.colorbar(im, orientation="horizontal")
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            print "pdf Page written ",PageNum
            f.show()
        
          
    
    f.savefig(pp, format='pdf') 
    print "Final pdf Page written ",PageNum
    f.show()
    pp.close()
  
    
#------------------------------------------------------------------------------------------------------------------
def ShowOneOrder_contourinPDF(all_images,all_titles,thex0,they0,object_name,all_expo,dir_top_img,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowRawImages_contour: Show the raw images without background subtraction
    ==============
    """
   
    NBIMAGES=len(all_images)
    MAXIMGROW=max(2,m.ceil(NBIMAGES/NBIMGPERROW))
    
    spec_index_min=100  # cut the left border
    spec_index_max=right_edge # cut the right border
    star_halfwidth=70
    
    
   
    figfilename=os.path.join(dir_top_img,figname)  
    titlepage='Images of {}, date : {}'.format(object_name,date)

    
     # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    
    for index in np.arange(0,NBIMAGES):
        
        
        x0=int(thex0[index])
        y0=int(they0[index])
        
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
                
        
        full_image=np.copy(all_images[index])
        
      
      
        
        if(all_expo[index]<=0 ): #special case of the first image
            reduc_image=full_image[y0-20:y0+20,x0+150:spec_index_max]  
        else:
            reduc_image=full_image[y0-20:y0+20,x0+150:spec_index_max]/all_expo[index] 
            
        themax=reduc_image.flatten().max()    
            
        X,Y=np.meshgrid(np.arange(0,reduc_image.shape[1]),np.arange(reduc_image.shape[0]))  
        
        T=np.transpose(reduc_image)
        
        #im = axarr[iy,ix].pcolormesh(X,Y,reduc_image, cmap='rainbow',vmin=0,vmax=themax*0.5)    # pcolormesh is impracticalble in pdf file 
        axarr[iy,ix].contourf(X, Y, reduc_image, 8, alpha=.75, cmap='jet')
        C = axarr[iy,ix].contour(X, Y, reduc_image , 8, colors='black', linewidth=.5)
        
            
        axarr[iy,ix].axis([X.min(), X.max(), Y.min(), Y.max()]); axarr[iy,ix].grid(True)
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
    
        axarr[iy,ix].grid(color='white', ls='solid')
        #axarr[iy,ix].text(700,2.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='black', fontweight='bold',fontsize=16)
        
        
        # save a new page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            print "pdf Page written ",PageNum
            f.show()
        
          
    
    f.savefig(pp, format='pdf') 
    print "Final pdf Page written ",PageNum
    f.show()
    pp.close()
#-------------------------------------------------------------------------------------------------------------------------------------  
    
def ShowTransverseProfileinPDF(all_images,thex0,all_titles,object_name,all_expo,dir_top_img,all_filt,date,figname,w=10,Dist=50,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowTransverseProfile: Show the raw images without background subtraction
    =====================
    The goal is to see in y, where is the spectrum maximum. Returns they0
    
    """

    NBIMAGES=len(all_images)
    MAXIMGROW=max(2,m.ceil(NBIMAGES/NBIMGPERROW))
    
    spec_index_min=100  # cut the left border
    spec_index_max=right_edge # cut the right border
    star_halfwidth=70  
    
    
    titlepage='Spectrum tranverse profile object : {} date : {}'.format(object_name, date)
    
     # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    
    

    ############       Criteria for spectrum region selection #####################
    #DeltaX=1000
    #w=10
    #ws=80
    #Dist=3*w
    
    ##############################################################################
    
    # containers
    thespectra= []
    thespectraUp=[]
    thespectraDown=[]
    
    they0 = []  # container for the y value
    
    # loop on images
    #-----------------
    for index in np.arange(0,NBIMAGES):
        
        
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
        
        
        x0=int(thex0[index])
        
        # index of image in the page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        
        # only order one
        data=np.copy(all_images[index])[:,x0+spec_index_min:right_edge]
        #order 0 and order 1
        data2=np.copy(all_images[index])[:,0:right_edge]

        # Do not erase central star please
        #data[:,x0-100:x0+100]=0 ## TURN OFF CENTRAL STAR
        
        if(all_expo[index]<=0):            
            yprofile=np.sum(data,axis=1)
        else:
            yprofile=np.sum(data,axis=1)/all_expo[index]
            
            
        ymin=1
        ymax=yprofile.max()
        y0=np.where(yprofile==ymax)[0][0]
        they0.append(y0)
        
        #y0 = they0[index]
        #im=axarr[iy,ix].imshow(data,vmin=-10,vmax=50)
        axarr[iy,ix].semilogy(yprofile,color='blue',lw=2)
        
        axarr[iy,ix].semilogy([y0,y0],[ymin,ymax],'r-',lw=2)
        
        axarr[iy,ix].semilogy([y0-w,y0-w],[ymin,ymax],'k-')
        axarr[iy,ix].semilogy([y0+w,y0+w],[ymin,ymax],'k-')
        
        axarr[iy,ix].semilogy([y0-w-Dist,y0-w-Dist],[ymin,ymax],'g-')
        axarr[iy,ix].semilogy([y0+w-Dist,y0+w-Dist],[ymin,ymax],'g-')
        
        axarr[iy,ix].semilogy([y0-w+Dist,y0-w+Dist],[ymin,ymax],'g-')
        axarr[iy,ix].semilogy([y0+w+Dist,y0+w+Dist],[ymin,ymax],'g-')
        
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        ##########################################################
        #### Here extract the spectrum around the central star
        #####   Take the sum un bins along y
        #############################################################
        
        spectrum2D=np.copy(data2[y0-w:y0+w,:])
        xprofile=np.sum(spectrum2D,axis=0)

        ###-----------------------------------------
        ### Lateral bands to remove sky background
        ### ---------------------------------------
        
        # region Up at average distance Dist of width 2*w
        spectrum2DUp=np.copy(data2[y0-w+Dist:y0+w+Dist,:])
        xprofileUp=np.median(spectrum2DUp,axis=0)*2.*float(w)

        # region Down at average distance -Dist of width 2*w
        spectrum2DDown=np.copy(data2[y0-w-Dist:y0+w-Dist,:])
        xprofileDown=np.median(spectrum2DDown,axis=0)*2.*float(w)
        
        
        if(all_expo[index]<=0):
            thespectra.append(xprofile)
            thespectraUp.append(xprofileUp)
            thespectraDown.append(xprofileDown)
        else:  ################## HERE I NORMALISE WITH EXPO TIME ####################################
            thespectra.append(xprofile/all_expo[index])
            thespectraUp.append(xprofileUp/all_expo[index]) 
            thespectraDown.append(xprofileDown/all_expo[index]) 

        #axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].grid(True)
    
        # save a new page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
        
          
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()
  
    
    return thespectra,thespectraUp,thespectraDown,they0    
#------------------------------------------------------------------------------------------------------------
def ShowLongitBackgroundinPDF(spectra,spectraUp,spectraDown,spectraAv,all_titles,object_name,dir_top_images,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    Show the background to be removed to the spectrum
    
    Implemented to write in a pdf file
    
    """
    NBSPEC=len(spectra)
    MAXIMGROW=max(2,m.ceil(NBSPEC/NBIMGPERROW))
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    figfilename=os.path.join(dir_top_images,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    titlepage='Longitudinal background Up/Down for obj : {} date : {} '.format(object_name,date)
    
    
    spec_index_min=100  # cut the left border
    spec_index_max=right_edge # cut the right border
    star_halfwidth=70
    
    for index in np.arange(0,NBSPEC):
        
        
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        # plot what is wanted
        axarr[iy,ix].plot(spectra[index],'r-')
        axarr[iy,ix].plot(spectraUp[index],'b-')
        axarr[iy,ix].plot(spectraDown[index],'g-')
        axarr[iy,ix].plot(spectraAv[index],'m-')
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        axarr[iy,ix].grid(True)
        
        star_pos=np.where(spectra[index][:spec_index_max]==spectra[index][:spec_index_max].max())[0][0]
        max_y_to_plot=(spectra[index][star_pos+star_halfwidth:spec_index_max]).max()*1.2
        
        
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        #axarr[iy,ix].text(spec_index_min,max_y_to_plot*1.1/1.2, all_filt[index],verticalalignment='top', horizontalalignment='center',color='blue',fontweight='bold', fontsize=20)
        
        
        # save a new page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
        
        
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()
    
    
#-----------------------------------------------------------------------------------------------------------------------

def ShowTheBackgroundProfileUpDowninPDF(spectra,spectra2,all_titles,object_name,dir_top_img,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowSpectrumProfile: Show the raw images without background subtraction
    =====================
    """
    
    
    spec_index_min=100  # cut the left border
    spec_index_max=right_edge # cut the right border
    star_halfwidth=70
    
    
    NBIMGPERROW=2
    NBSPEC=len(spectra)
    MAXIMGROW=max(2,m.ceil(NBSPEC/NBIMGPERROW))
   
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    titlepage='Spectrum 1D background profiles up dand down and av for obj: {} , date : {} (backg. rem.)'.format(object_name,date)
    
    
    
    #f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,5*MAXIMGROW))
    #f.tight_layout()
    for index in np.arange(0,NBSPEC):
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        
        axarr[iy,ix].plot(spectra[index],'r-')
        axarr[iy,ix].plot(spectra2[index],'b-')
        #axarr[iy,ix].plot(spectra3[index],'g-')
        
        
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        axarr[iy,ix].grid(True)
        
        
        star_pos=np.where(spectra[index][:spec_index_max]==spectra[index][:spec_index_max].max())[0][0]
        max_y_to_plot=(spectra[index][star_pos+star_halfwidth:spec_index_max]).max()*1.5
        
        
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        #axarr[iy,ix].text(spec_index_min,max_y_to_plot*1.1/1.5, all_filt[index],verticalalignment='top', horizontalalignment='center',color='blue',fontweight='bold', fontsize=20)
        
        
        # save the pdf page at the botton end of the page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
   
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()    


#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def ShowTheBackgroundProfileinPDF(spectra,all_titles,object_name,dir_top_img,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowSpectrumProfile: Show the raw images without background subtraction
    =====================
    """
    
    
    spec_index_min=100  # cut the left border
    spec_index_max=right_edge # cut the right border
    star_halfwidth=70
    
    
    NBIMGPERROW=2
    NBSPEC=len(spectra)
    MAXIMGROW=max(2,m.ceil(NBSPEC/NBIMGPERROW))
   
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    titlepage='Spectrum 1D background profile for obj: {} , date : {} (backg. rem.)'.format(object_name,date)
    
    
    
    #f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,5*MAXIMGROW))
    #f.tight_layout()
    for index in np.arange(0,NBSPEC):
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        
        axarr[iy,ix].plot(spectra[index],'r-')
        
        
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        axarr[iy,ix].grid(True)
        
        
        star_pos=np.where(spectra[index][:spec_index_max]==spectra[index][:spec_index_max].max())[0][0]
        max_y_to_plot=(spectra[index][star_pos+star_halfwidth:spec_index_max]).max()*1.5
        
        
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        #axarr[iy,ix].text(spec_index_min,max_y_to_plot*1.1/1.5, all_filt[index],verticalalignment='top', horizontalalignment='center',color='blue',fontweight='bold', fontsize=20)
        
        
        # save the pdf page at the botton end of the page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
   
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()
    
            

#-----------------------------------------------------------------------------------------------------------------------
    
def ShowCorrectedSpectrumProfileinPDF(spectra,all_titles,object_name,dir_top_img,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowSpectrumProfile: Show the raw images without background subtraction
    =====================
    """
    
    
    spec_index_min=100  # cut the left border
    spec_index_max=right_edge # cut the right border
    star_halfwidth=70
    
    
    NBIMGPERROW=2
    NBSPEC=len(spectra)
    MAXIMGROW=max(2,m.ceil(NBSPEC/NBIMGPERROW))
   
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    titlepage='Spectrum 1D profile for obj: {} , date : {} (backg. rem.)'.format(object_name,date)
    
    
    
    #f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,5*MAXIMGROW))
    #f.tight_layout()
    for index in np.arange(0,NBSPEC):
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        
        axarr[iy,ix].plot(spectra[index],'r-')
        
        
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        axarr[iy,ix].grid(True)
        
        
        star_pos=np.where(spectra[index][:spec_index_max]==spectra[index][:spec_index_max].max())[0][0]
        max_y_to_plot=(spectra[index][star_pos+star_halfwidth:spec_index_max]).max()*1.5
        
        
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        #axarr[iy,ix].text(spec_index_min,max_y_to_plot*1.1/1.5, all_filt[index],verticalalignment='top', horizontalalignment='center',color='blue',fontweight='bold', fontsize=20)
        
        
        # save the pdf page at the botton end of the page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
   
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()
    
#---------------------------------------------------------------------------------------------------------------------        
    
def ShowSpectrumRightProfileinPDF(spectra,thex0,all_titles,object_name,dir_top_img,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowSpectrumProfile: Show the raw images without background subtraction
    =====================
    """
    NBIMGPERROW=2
    NBSPEC=len(spectra)
    
    MAXIMGROW=max(2,int(m.ceil(float(NBSPEC)/float(NBIMGPERROW))))
    
    spec_index_min=100  # cut the left border
    spec_index_max=right_edge # cut the right border
    
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    
    
   
    
    titlepage='Right spectra for obj: {} , date : {} (backg. rem.)'.format(object_name,date)
    
        
    for index in np.arange(0,NBSPEC):
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        x0=int(thex0[index])
        cut_spectra=np.copy(spectra[index][x0:])
        cut_spectra_order1=np.copy(spectra[index][x0+100:])
        
                           
        axarr[iy,ix].plot(cut_spectra,'r-')
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        axarr[iy,ix].grid(True)
        
        max_y_to_plot=cut_spectra_order1.max()*1.2
        
        
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        #axarr[iy,ix].text(0.,max_y_to_plot*1.1/1.2, all_filt[index],verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
    
    
        # save the pdf page at the botton end of the page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()
    

    

#---------------------------------------------------------------------------------------------------------------------

def ShowSpectrumLeftProfileinPDF(spectra,thex0,all_titles,object_name,dir_top_img,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    ShowSpectrumProfile: Show the raw images without background subtraction
    =====================
    """
    NBIMGPERROW=2
    NBSPEC=len(spectra)
    
    MAXIMGROW=max(2,int(m.ceil(float(NBSPEC)/float(NBIMGPERROW))))
    
    
    spec_index_min=100  # cut the left border
    spec_index_max=right_edge # cut the right border
    
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    titlepage='Left spectra for obj: {} , date : {} (backg. rem.)'.format(object_name,date)
    
   
    for index in np.arange(0,NBSPEC):
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        x0=int(thex0[index])
        
        cut_spectra=np.copy(spectra[index][:x0])
        cut_spectra_orderm1=np.copy(spectra[index][:x0-spec_index_min])
        
       
  
        axarr[iy,ix].plot(cut_spectra,'r-')
        
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        axarr[iy,ix].grid(True)
        
        
        max_y_to_plot=cut_spectra_orderm1.max()*1.2
        
                
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)

        axarr[iy,ix].text(0.,max_y_to_plot*1.1/1.2, all_filt[index],verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
    
        # save the pdf page at the botton end of the page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()
    
#----------------------------------------------------------------------------------------------------------------------

def Find_CentralStar_positioninPDF(spectra,thex0,all_titles,object_name,dir_top_img,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    
    Find_CentralStar_position
    =========================
    
    Find star postion by gausian fit
    
    """
    NBIMGPERROW=2
    NBSPEC=len(spectra)
    
    MAXIMGROW=max(2,int(m.ceil(float(NBSPEC)/float(NBIMGPERROW))))

    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    titlepage='Central star for obj: {} , date : {} (backg. rem.)'.format(object_name,date)
    
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    
    all_mean = []  # list of mean and sigma for the main central star
    all_sigma= []
    
   
    for index in np.arange(0,NBSPEC):
        
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
    
        
        star_index=int(thex0[index])
        
        cutspectra=np.copy(spectra[index][star_index-star_halfwidth:star_index+star_halfwidth]) 
        X=np.arange(cutspectra.shape[0])+star_index-star_halfwidth
        
        # fast fit of a gaussian, bidouille
        
        x = np.sum(X*cutspectra)/np.sum(cutspectra)
        width = 0.5*np.sqrt(np.abs(np.sum((X-x)**2*cutspectra)/np.sum(cutspectra)))
        themax = cutspectra.max()
        
        all_mean.append(int(x))
        all_sigma.append(int(width))
        
        fit = lambda t : themax*np.exp(-(t-x)**2/(2*width**2))
        
        
        #print 'mean,width, max =',x,width,themax
        thelabel='fit m={}, $\sigma$= {}'.format(int(x),int(width))
        axarr[iy,ix].plot(X,cutspectra,'r-',label='data')
        axarr[iy,ix].plot(X,fit(X), 'b-',label=thelabel)
        
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        axarr[iy,ix].grid(True)
        
        max_y_to_plot=cutspectra.max()*1.2
        
        
        axarr[iy,ix].set_ylim(0,max_y_to_plot)
        axarr[iy,ix].legend(loc=1)

        axarr[iy,ix].text(star_index-star_halfwidth/2,max_y_to_plot*1.1/1.2, all_filt[index],verticalalignment='top', horizontalalignment='right',color='blue',fontweight='bold', fontsize=20)
    
        # save the pdf page at the botton end of the page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()



    return all_mean,all_sigma
        
#-------------------------------------------------------------------------------------------------------------------        

def SpectrumAmplitudeRatio(spectra,star_indexes):
    """
    SpectrumAmplitudeRatio: ratio of amplitudes
    =====================
    """
    
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=100
    
    
    ratio_list= []
    
    NBSPEC=len(spectra)
    
    for index in np.arange(0,NBSPEC):
        star_index=int(star_indexes[index])  # position of star
        
        max_right=spectra[index][star_index+star_halfwidth:spec_index_max].max()
        max_left=spectra[index][spec_index_min:star_index-star_halfwidth].max()
        
        ratio=max_right/max_left
        ratio_list.append(ratio) 
        
    return np.array(ratio_list)    

#--------------------------------------------------------------------------------------------

def GetSpectrumBackground(inspectra,start,stop,skybg):
    '''
    Return the background    
    '''
    cropedbg=inspectra[start:stop]
    purebg=cropedbg[np.where(cropedbg!=skybg)]  # remove region of the bing star
    
    return purebg

#-------------------------------------------------------------------------------------------
    
def SeparateSpectra(inspectra,x0):
    '''
    Cut the two spectra
    '''
    rightspectra=inspectra[x0:]
    revleftspectra=inspectra[:x0]
    leftspectra=   revleftspectra[::-1]
    #rightspectra=rightspectra[np.where(rightspectra>0)]
    #leftspectra=leftspectra[np.where(leftspectra>0)]
    
    return leftspectra,rightspectra

#------------------------------------------------
def DiffSpectra(spec1,spec2,bg):
    '''
    Make the difference of the tow spectra 
    
    '''
    N1=spec1.shape[0]
    N2=spec2.shape[0]
    N=np.min([N1,N2])
    spec1_croped=spec1[0:N]
    spec2_croped=spec2[0:N]
    diff_spec=np.average((spec1_croped-spec2_croped)**2)/bg**2
    return diff_spec  

#---------------------------------------------------------
def FindCenter(fullspectrum,xmin,xmax,specbg,factor=1):
    '''
    - spec1 is left spectra
    - spec2 is right spectra
    
    In case of asymetric orders one can use a multiplication factor
    '''   
    all_x0=np.arange(xmin,xmax,1)
    NBPOINTS=np.shape(all_x0)
    chi2=np.zeros(NBPOINTS)
    for idx,x0 in np.ndenumerate(all_x0):
        spec1,spec2=SeparateSpectra(fullspectrum,x0) #sparate the spectra in two pieces
        
        spec1_factor=spec1*factor
        #chi2[idx]=DiffSpectra(spec1,spec2,specbg)
        chi2[idx]=DiffSpectra(spec1_factor,spec2,specbg)
    return all_x0,chi2
#----------------------------------------------------------
    
def SplitSpectrumProfile(spectra,all_titles,object_name,star_indexes,dir_top_img, Thor300_ind,Ron400_ind,Ron200_ind,HoloPhP_ind,HoloPhAg_ind,HoloAmAg_ind):
    """
    SplitSpectrumProfile: Split the spectrum in two parts
    =====================
    """
    NBIMGPERROW=2
    NBSPEC=len(spectra)
    
    MAXIMGROW=max(2,int(m.ceil(float(NBSPEC)/float(NBIMGPERROW))))
    
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,'split_spectra.pdf')
    pp = PdfPages(figfilename) # create a pdf file
    
    
    title='Multiple 1D Spectra 1D for '.format(object_name)
    
    
    skybg=1
    spectra_left=[]  # container for spectra for split mode (left/right symetry)
    spectra_right=[]
    
    spectra_left_2=[] # container for spectra for cut mode (central star positon)
    spectra_right_2=[]
    
   
    
    for index in np.arange(0,NBSPEC):
        
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(title,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        
        
        star_index=star_indexes[index]

        if np.any(Thor300_ind == index):
            factor=Thor300_ratio_av
        elif np.any(Ron400_ind == index):
            factor=Ron400_ratio_av
        elif np.any(Ron200_ind == index):
            factor=Ron200_ratio_av
        elif np.any(HoloPhP_ind == index):
            factor=HoloPhP_ratio_av
        elif np.any(HoloPhAg_ind == index):
            factor=HoloPhAg_ratio_av
        elif np.any(HoloAmAg_ind == index):
            factor=HoloAmAg_ratio_av
        else:
            print "disperser not found "
            factor=1
        
        spectrum_sel=np.copy(spectra[index])
        
        spectrum_sel[star_index-star_halfwidth:star_index+star_halfwidth]=0  # erase central star
        
        
        all_candidate_center,all_chi2=FindCenter(spectrum_sel,xmin_center,xmax_center,skybg,factor)
        
        indexmin=np.where(all_chi2==all_chi2.min())[0]
        theorigin=all_candidate_center[indexmin]
        
        #print index, theorigin
        
        # old split by chi2 min
        spec1,spec2=SeparateSpectra(spectrum_sel,theorigin[0])
        
        spectra_left.append(spec1)
        spectra_right.append(spec2)
        
        # new split by central star position
        spec3,spec4=SeparateSpectra(spectrum_sel,star_index)
        
        spectra_left_2.append(spec3)
        spectra_right_2.append(spec4)
        
        axarr[iy,ix].plot(spec4,'r-',lw=2,label='cut right spec')
        axarr[iy,ix].plot(spec3*factor,'b-',lw=1,label='renorm cut right left')
        
        axarr[iy,ix].plot(spec2,'m-',lw=2,label='split right spec')
        axarr[iy,ix].plot(spec1*factor,'k-',lw=1,label='renorm split left spec')
        
    
        max_y_to_plot=spec2[:spec_index_max].max()*1.2
        
        
        axarr[iy,ix].legend(loc=1)                  
        axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].grid(True)
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        axarr[iy,ix].text(0.,max_y_to_plot*1.1/1.2, all_filt[index],verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
    
   
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()    
  
    
    return spectra_left,spectra_right,spectra_left_2,spectra_right_2

#--------------------------------------------------------------------------------------------
    

def SplitSpectrumProfileSimpleinPDF(spectra,thex0,all_titles,object_name,dir_top_img,all_filt,date,figname,right_edge = 1900,NBIMGPERROW=2,vmin=0,vmax=2000,downsampling=1,verbose=False):
    """
    SplitSpectrumProfile: Split the spectrum in two parts
    =====================
    """
    NBIMGPERROW=2
    NBSPEC=len(spectra)
    
    MAXIMGROW=max(2,int(m.ceil(float(NBSPEC)/float(NBIMGPERROW))))
    
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
        
 
    titlepage='Left/Right corr spectra for obj: {} , date : {} (backg. rem.)'.format(object_name,date)
    
    
    spectra_left=[] # container for spectra for cut mode (central star positon)
    spectra_right=[]
    
   
    
    for index in np.arange(0,NBSPEC):
        
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        
        # found central star
        star_index=int(thex0[index])
        
        spectrum_sel=np.copy(spectra[index])      
        spectrum_sel[star_index-star_halfwidth:star_index+star_halfwidth]=0  # erase central star
        
        
  
        # new split by central star position
        spec1,spec2=SeparateSpectra(spectrum_sel,star_index)
        
        spectra_left.append(spec1)
        spectra_right.append(spec2)
        
        axarr[iy,ix].plot(spec2,'r-',lw=2,label='cut right spec')
        axarr[iy,ix].plot(spec1,'b-',lw=1,label='cut right left')
        
    
        max_y_to_plot=spec2[:spec_index_max].max()*1.2
        
        
        axarr[iy,ix].legend(loc=1)                  
        
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        axarr[iy,ix].grid(True)
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        axarr[iy,ix].text(0.,max_y_to_plot*1.1/1.2, all_filt[index],verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
    
   
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()    
  
    
    return spectra_left,spectra_right

#---------------------------------------------------------------------------------------------------------------------
    
def CompareSpectrumProfileinPDF(spectra,all_airmass,all_titles,object_name,dir_top_img,all_filt,date,figname,grating_name,list_idx,right_edge = 1900):
    """
    CompareSpectrumProfile
    =====================
    
    """
    
    
    
    
    titlepage="Compare spectra of {}  at {} with disperser {}".format(object_name,date,grating_name)
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    
    f, axarr = plt.subplots(1,1,figsize=(10,5))
    f.suptitle(titlepage,color='blue',fontweight='bold',fontsize=16)
    
    NBSPEC=len(spectra)
    
    min_z=min(all_airmass)
    max_z=max(all_airmass)
    
    maxim_y_to_plot= []

    texte='airmass : {} - {} '.format(min_z,max_z)
    
    for index in np.arange(0,NBSPEC):
                
        if index in list_idx:
            axarr.plot(spectra[index],'r-')
            maxim_y_to_plot.append(spectra[index].max())
    
    max_y_to_plot=max(maxim_y_to_plot)*1.2
    axarr.set_ylim(0,max_y_to_plot)
    axarr.text(0.,max_y_to_plot*0.9, texte ,verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
    axarr.grid(True)
    axarr.set_xlabel("pixel",fontweight='bold',fontsize=14)
    axarr.set_ylabel("ADU",fontweight='bold',fontsize=14)
    
        
    f.savefig(pp, format='pdf')
    f.show()
    
    pp.close()     
    
#-----------------------------------------------------------------------------------------------------------------------------    
    






    
#--------------------------------------------------------------------------------------------------------------------
def ShowRightOrder(all_images,thex0,they0,all_titles,object_name,all_expo,dir_top_images):
    """
    ShowRawImages: Show the raw images without background subtraction
    ==============
    """
    NBIMGPERROW=2
    NBIMAGES=len(all_images)
    MAXIMGROW=int(NBIMAGES/NBIMGPERROW)+1
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,4*MAXIMGROW))
    f.tight_layout()
    
    right_edge = 1800
    
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        full_image=np.copy(all_images[index])[:,0:right_edge]
        y_0=int(they0[index])
        x_0=int(thex0[index])

        reduc_image=full_image[y_0-20:y_0+20,x_0+100:right_edge]/all_expo[index]
        
        X,Y=np.meshgrid(np.arange(0,reduc_image.shape[1]),np.arange(reduc_image.shape[0]))
        im = axarr[iy,ix].pcolormesh(X,Y,reduc_image, cmap='rainbow',vmin=0,vmax=100)
        #axarr[iy,ix].colorbar(im, orientation='vertical')
        axarr[iy,ix].axis([X.min(), X.max(), Y.min(), Y.max()]); axarr[iy,ix].grid(True)
        
        axarr[iy,ix].set_title(all_titles[index])
        
    
    title='Right part of spectrum of {} '.format(object_name)
    plt.suptitle(title,size=16)
    figfilename=os.path.join(dir_top_images,'rightorder.pdf')
    
    #plt.savefig(figfilename)          
#--------------------------------------------------------------------------------------------
def ShowLeftOrder(all_images,thex0,they0,all_titles,object_name,all_expo,dir_top_images):
    """
    ShowRawImages: Show the raw images without background subtraction
    ==============
    """
    NBIMGPERROW=2
    NBIMAGES=len(all_images)
    MAXIMGROW=int(NBIMAGES/NBIMGPERROW)+1
    #thex0 = []
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,4*MAXIMGROW))
    f.tight_layout()

    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        full_image=np.copy(all_images[index])
        y_0=int(they0[index])
        x_0=int(thex0[index])
        
        
        reduc_image=full_image[y_0-20:y_0+20,0:x_0-100]/all_expo[index] 

        X,Y=np.meshgrid(np.arange(0,reduc_image.shape[1]),np.arange(reduc_image.shape[0]))
        im = axarr[iy,ix].pcolormesh(X,Y,reduc_image, cmap='rainbow',vmin=0,vmax=30)
        #axarr[iy,ix].colorbar(im, orientation='vertical')
        axarr[iy,ix].axis([X.min(), X.max(), Y.min(), Y.max()]); axarr[iy,ix].grid(True)
        
        axarr[iy,ix].set_title(all_titles[index])
        
    
    title='Left part of spectrum of '.format(object_name)
    plt.suptitle(title,size=16)
    figfilename=os.path.join(dir_top_images,'leftorder.pdf')
    #plt.savefig(figfilename)      
    
#-------------------------------------------------------------------------------------------------------------------    
def ShowHistograms(all_images,all_titles,all_filt,object_name,NBIMGPERROW=2,bins=100,range=(-50,10000),downsampling=1,verbose=False):
    """
    ShowHistograms
    ==============
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1

    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(20,int(5*MAXIMGROW)))
    
    for index in np.arange(0,NBIMAGES):
        if verbose : print 'Processing image %d...' % index
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        image_flat=all_images[index][::downsampling,::downsampling].flatten()
        stat_mean=image_flat.mean()
        stat_rms=image_flat.std()
        legtitle='mean={:4.2f} std={:4.2f}'.format(stat_mean,stat_rms)
        axarr[iy,ix].hist(image_flat,bins=bins,range=range,facecolor='blue', alpha=0.75,label=legtitle);
        axarr[iy,ix].set_yscale('log')
        axarr[iy,ix].grid(True)
        axarr[iy,ix].set_ylim(0.,1e10)
        axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].legend(loc='best')  #useless
    title='histograms of images {}  '.format(object_name)
    plt.suptitle(title,size=16)        

#-----------------------------------------------------------------------------
    
def ComputeStatImages(all_images,fwhm=10,threshold=300,sigma=10.0,iters=5):
    """
    ComputeStatImages: 
    ==============
    all_images : the images
    fwhm : size of the sources to search
    threshold : number fo times above std
    """
    
    img_mean=[]
    img_median=[]
    img_std=[]
    img_sources=[]
    img_=[]
    index=0
    for image in all_images:
        mean, median, std = sigma_clipped_stats(image, sigma=sigma, iters=iters)    
        print '----------------------------------------------------------------'
        print index,' mean, median, std = ',mean, median, std
        img_mean.append(mean)
        img_median.append(median)
        img_std.append(std)
        sources = daofind(image - median, fwhm=fwhm, threshold=threshold*std) 
        print sources
        img_sources.append(sources)    
        index+=1
    return img_mean,img_median,img_std,img_sources

#--------------------------------------------------------------------------------
def ShowCenterImages(thex0,they0,DeltaX,DeltaY,all_images,all_titles,all_filt,object_name,NBIMGPERROW=2,vmin=0,vmax=2000,mask_saturated=False,target_pos=None):
    """
    ShowCenterImages: Show the raw images without background subtraction
    ==============
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1
    
    croped_images = []
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,5*MAXIMGROW))
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        x0 = int(thex0[index])
        y0 = int(they0[index])
        deltax = int(DeltaX[index])
        deltay = int(DeltaY[index])
        theimage=all_images[index]
        image_cut=np.copy(theimage[max(0,y0-deltay):min(IMSIZE,y0+deltay),max(0,x0-deltax):min(IMSIZE,x0+deltax)])
        croped_images.append(image_cut)
        if mask_saturated :
            bad_pixels = np.where(image_cut>MAXADU)
            image_cut[bad_pixels] = np.nan
        #aperture=CircularAperture([positions_central[index]], r=100.)
        im=axarr[iy,ix].imshow(image_cut,cmap='rainbow',vmin=vmin,vmax=vmax,aspect='auto',origin='lower',interpolation='None')
        axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].grid(color='white', ls='solid')
        #aperture.plot(color='red', lw=5.)
        axarr[iy,ix].text(5.,5,all_filt[index],verticalalignment='bottom', horizontalalignment='left',color='yellow', fontweight='bold',fontsize=16)
        if target_pos is not None :
            xpos = max(0,target_pos[index][0]-max(0,x0-deltax))
            ypos = max(0,target_pos[index][1]-max(0,y0-deltay))
            s = 2*min(image_cut.shape)
            axarr[iy,ix].scatter(xpos,ypos,s=s,edgecolors='k',marker='o',facecolors='none',linewidths=2)
    title='Cut Images of {}'.format(object_name)
    plt.suptitle(title,size=16) 
    return croped_images

#-------------------------------------------------------------------------------
    
def ComputeMedY(data):
    """
    Compute the median of Y vs X to find later the angle of rotation
    """
    NBINSY=data.shape[0]
    NBINSX=data.shape[1]
    the_medianY=np.zeros(NBINSX)
    the_y=np.zeros(NBINSY)
    for ix in np.arange(NBINSX):
        the_ysum=np.sum(data[:,ix])
        for iy in np.arange(NBINSY):
            the_y[iy]=iy*data[iy,ix]
        if(the_ysum>0):
            med=np.sum(the_y)/the_ysum
            the_medianY[ix]=med
    return the_medianY
#--------------------------------------------------------------------------------
    
def ComputeAveY(data):
    """
    Compute the average of Y vs X to find later the angle of rotation
    """
    NBINSY=data.shape[0]
    NBINSX=data.shape[1]
    the_averY=np.zeros(NBINSX)
    the_y=np.zeros(NBINSY)
    for ix in np.arange(NBINSX):
        the_ysum=np.sum(data[:,ix])
        for iy in np.arange(NBINSY):
            the_y[iy]=iy*data[iy,ix]
        if(the_ysum>0):
            med=np.sum(the_y)/the_ysum
            the_averY[ix]=med
    return the_averY



#--------------------------------------------------------------------------------


def ComputeRotationAngle(all_images,thex0,they0,all_titles,object_name):
    """
    ComputeRotationAngle
    ====================
    
    input:
    ------
    all_images
    thex0
    they0

    output:
    ------
    param_a
    param_b
    
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1
    
    param_a=np.zeros(NBIMAGES)
    param_b=np.zeros(NBIMAGES)

    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,4*MAXIMGROW))
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        
        image=all_images[index]    
        
        image_sel=np.copy(image)
        y0=they0[index]
        x0=thex0[index]
        
        # extract a region of 200 x 1000 centered at y=100,x=500
        
        region=np.copy(image_sel[y0-100:y0+100,:])
        data=np.copy(region)
        
        xindex=np.arange(data.shape[1])
        
        #selected_indexes=np.where(np.logical_or(np.logical_and(xindex>100,xindex<200) ,np.logical_and(xindex>1410,xindex<1600))) 
        selected_indexes=np.where(np.logical_or(np.logical_and(xindex>0,xindex<150) ,np.logical_and(xindex>1500,xindex<1600)))
        # compute Y vs X
        yaver=ComputeAveY(data)
        
        XtoFit=xindex[selected_indexes]
        YtoFit=yaver[selected_indexes]
        # does the fit
        params = curve_fit(fit_func, XtoFit, YtoFit)
        [a, b] = params[0]
        
        param_a[index]=a
        param_b[index]=b
        
        print index,' y = ',a,' * x + ',b
        x_new = np.linspace(xindex.min(),xindex.max(), 50)
        y_new = fit_func(x_new,a,b)
    
        im=axarr[iy,ix].plot(XtoFit,YtoFit,'ro')
        im=axarr[iy,ix].plot(x_new,y_new,'b-')
        axarr[iy,ix].set_title(all_titles[index])
        
        axarr[iy,ix].set_ylim(0,200)
        axarr[iy,ix].grid(True)
        
    title='Fit rotation angle of '.format(object_name)    
    plt.suptitle(title,size=16)
    
    figfilename=os.path.join(dir_top_images,'fit_rotation.pdf')
    plt.savefig(figfilename)  
    
    
    return param_a,param_b
#---------------------------------------------------------------------------------------------

def ComputeRotationAngleHessian(all_images,thex0,they0,all_titles,object_name,NBIMGPERROW=2, lambda_threshold = -20, deg_threshold = 20, width_cut = 20, right_edge = 1600,margin_cut=1):
    """
    ComputeRotationAngle
    ====================
    
    input:
    ------
    all_images
    thex0
    they0
    all_titles
    object_name
    
    output:
    ------
    rotation angles
    
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1
    
    param_theta=np.zeros(NBIMAGES)
    
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,2*MAXIMGROW))
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        
        image=all_images[index]    
        
        image_sel=np.copy(image)
        y0=int(they0[index])
        x0=int(thex0[index])
        
        # extract a region 
        region=np.copy(image_sel[y0-width_cut:y0+width_cut,0:right_edge])
        data=np.copy(region)
        
        # compute hessian matrices on the image
        Hxx, Hxy, Hyy = hessian_matrix(data, sigma=3, order = 'xy')
        lambda_plus = 0.5*( (Hxx+Hyy) + np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
        lambda_minus = 0.5*( (Hxx+Hyy) - np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
        theta = 0.5*np.arctan2(2*Hxy,Hyy-Hxx)*180/np.pi
                
        # remobe the margins
        lambda_minus = lambda_minus[margin_cut:-margin_cut,margin_cut:-margin_cut]
        lambda_plus = lambda_plus[margin_cut:-margin_cut,margin_cut:-margin_cut]
        theta = theta[margin_cut:-margin_cut,margin_cut:-margin_cut]

        # thresholds
        mask = np.where(lambda_minus>lambda_threshold)
        theta_mask = np.copy(theta)
        theta_mask[mask]=np.nan

        mask2 = np.where(np.abs(theta)>deg_threshold)
        theta_mask[mask2] = np.nan
        
        theta_hist = []
        theta_hist = theta_mask[~np.isnan(theta_mask)].flatten()
        theta_median = np.median(theta_hist)
        
        param_theta[index] = theta_median
        
        xindex=np.arange(data.shape[1])
        x_new = np.linspace(xindex.min(),xindex.max(), 50)
        y_new = y0 - width_cut + (x_new-x0)*np.tan(theta_median*np.pi/180.)
    
        im=axarr[iy,ix].imshow(theta_mask,origin='lower',cmap=cm.brg,aspect='auto',vmin=-deg_threshold,vmax=deg_threshold)
        im=axarr[iy,ix].plot(x_new,y_new,'b-')
        axarr[iy,ix].set_title(all_titles[index])
        
        axarr[iy,ix].set_ylim(0,2*width_cut)
        axarr[iy,ix].grid(True)
        

    title='Fit rotation angle of '.format(object_name)    
    plt.suptitle(title,size=16)
    
    return param_theta
    
#-----------------------------------------------------------------------------------

def ComputeRotationAngleHessianAndFit(all_images,thex0,they0,all_titles,object_name, NBIMGPERROW=2, lambda_threshold = -20, deg_threshold = 20, width_cut = 20, right_edge = 1600,margin_cut=1):
    """
    ComputeRotationAngle
    ====================
    
    input:
    ------
    all_images
    thex0
    they0
    all_titles
    object_name
    
    output:
    ------
    rotation angles
    
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1

    param_theta=np.zeros(NBIMAGES)
    
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,2*MAXIMGROW))
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        
        image_sel=np.copy(all_images[index])
        y0=int(they0[index])
        x0=int(thex0[index])
        
        # extract a region of 200 x 1000 centered at y=100,x=500    
        region=np.copy(image_sel[max(0,y0-width_cut):min(y0+width_cut,IMSIZE),0:min(IMSIZE,right_edge)])
        data=np.copy(region)
        
        # compute hessian matrices on the image
        Hxx, Hxy, Hyy = hessian_matrix(data, sigma=3, order = 'xy')
        lambda_plus = 0.5*( (Hxx+Hyy) + np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
        lambda_minus = 0.5*( (Hxx+Hyy) - np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
        theta = 0.5*np.arctan2(2*Hxy,Hyy-Hxx)*180/np.pi
                
        # remobe the margins
        lambda_minus = lambda_minus[margin_cut:-margin_cut,margin_cut:-margin_cut]
        lambda_plus = lambda_plus[margin_cut:-margin_cut,margin_cut:-margin_cut]
        theta = theta[margin_cut:-margin_cut,margin_cut:-margin_cut]

        mask = np.where(lambda_minus>lambda_threshold)
        #lambda_mask = np.copy(lambda_minus)
        #lambda_mask[mask]=np.nan
        theta_mask = np.copy(theta)
        theta_mask[mask]=np.nan

        mask2 = np.where(np.abs(theta)>deg_threshold)
        theta_mask[mask2] = np.nan
        
        #theta_hist = []
        #theta_hist = theta_mask[~np.isnan(theta_mask)].flatten()
        #theta_median = np.median(theta_hist)
        
        xtofit=[]
        ytofit=[]
        for ky,y in enumerate(theta_mask):
            for kx,x in enumerate(y):
                if not np.isnan(theta_mask[ky][kx]) :
                    if np.abs(theta_mask[ky][kx])>deg_threshold : continue
                    xtofit.append(kx)
                    ytofit.append(ky)
        popt, pcov = fit_line(xtofit, ytofit)
        [a, b] = popt
        xindex=np.arange(data.shape[1])
        x_new = np.linspace(xindex.min(),xindex.max(), 50)
        y_new = line(x_new,a,b)
        
        param_theta[index] = np.arctan(a)*180/np.pi
        
        im=axarr[iy,ix].imshow(theta_mask,origin='lower',cmap=cm.brg,aspect='auto',vmin=-deg_threshold,vmax=deg_threshold)
        im=axarr[iy,ix].plot(x_new,y_new,'b-')
        axarr[iy,ix].set_title(all_titles[index])
        
        axarr[iy,ix].set_ylim(0,2*width_cut)
        axarr[iy,ix].grid(True)
        

    title='Fit rotation angle of '.format(object_name)    
    plt.suptitle(title,size=16)
        
    return param_theta
    
#------------------------------------------------------------------------------------------

def TurnTheImages(all_images,all_angles,all_titles,object_name,NBIMGPERROW=2,vmin=0,vmax=1000,oversample_factor=6):
    """
    TurnTheImages
    =============
    
    input:
    ------
    all_images:
    all_angles:
    
    
    output:
    ------
    all_rotated_images
    
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1
    
    all_rotated_images = []

    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,3*MAXIMGROW))
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        image=all_images[index]    
        angle=all_angles[index]
        if np.isnan(angle):
            all_rotated_images.append(image)
            continue
        data=np.copy(image)
        # prefilter=False and order=5 give best rotated images
        rotated_image=ndimage.interpolation.rotate(data,angle,prefilter=False,order=5)
        all_rotated_images.append(rotated_image)
        im=axarr[iy,ix].imshow(rotated_image,origin='lower',cmap='rainbow',vmin=vmin,vmax=vmax)
        axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].grid(color='white', ls='solid')
        axarr[iy,ix].grid(True)
        
    title='Rotated images for '.format(object_name)    
    plt.suptitle(title,size=16)
    
    return all_rotated_images

#--------------------------------------------------------------------------------------

def subplots_adjust(*args, **kwargs):
    """
    call signature::

      subplots_adjust(left=None, bottom=None, right=None, top=None,
                      wspace=None, hspace=None)

    Tune the subplot layout via the
    :class:`matplotlib.figure.SubplotParams` mechanism.  The parameter
    meanings (and suggested defaults) are::

      left  = 0.125  # the left side of the subplots of the figure
      right = 0.9    # the right side of the subplots of the figure
      bottom = 0.1   # the bottom of the subplots of the figure
      top = 0.9      # the top of the subplots of the figure
      wspace = 0.2   # the amount of width reserved for blank space between subplots
      hspace = 0.2   # the amount of height reserved for white space between subplots

    The actual defaults are controlled by the rc file
    """
    fig = gcf()
    fig.subplots_adjust(*args, **kwargs)
    draw_if_interactive()

#----------------------------------------------------------------------------------

def ShowOneOrder(all_images,all_titles,x0,object_name,all_expo,NBIMGPERROW=2):
    """
    ShowRawImages: Show the raw images without background subtraction
    ==============
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,5*MAXIMGROW))
    f.tight_layout()
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        full_image=np.copy(all_images[index])
        
        if(all_expo[index]<=0 ): #special case of the first image
            reduc_image=full_image[90:150,1000:1800]  
        else:
            reduc_image=full_image[90:150,1000:1800]/all_expo[index] 
        X,Y=np.meshgrid(np.arange(0,reduc_image.shape[1]),np.arange(reduc_image.shape[0]))
        im = axarr[iy,ix].pcolormesh(X,Y,reduc_image, cmap='rainbow',vmin=0,vmax=100)
        #axarr[iy,ix].colorbar(im, orientation='vertical')
        axarr[iy,ix].axis([X.min(), X.max(), Y.min(), Y.max()]); axarr[iy,ix].grid(True)
        
        axarr[iy,ix].set_title(all_titles[index])
        
    
    title='Images of {}'.format(object_name)
    plt.suptitle(title,size=16)

#--------------------------------------------------------------------------------

def ShowTransverseProfile(all_images,all_titles,object_name,all_expo,NBIMGPERROW=2,DeltaX=1000,w=10,ws=[10,20],right_edge=1800,ylim=None):
    """
    ShowTransverseProfile: Show the raw images without background subtraction
    =====================
    The goal is to see in y, where is the spectrum maximum. Returns they0
    
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1

    thespectra= []
    thespectraUp=[]
    thespectraDown=[]
    
    they0 = []
    
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,4*MAXIMGROW))
    f.tight_layout()
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        data=np.copy(all_images[index])[:,0:right_edge]
        
        if(all_expo[index]<=0):            
            yprofile=np.sum(data,axis=1)
        else:
            yprofile=np.sum(data,axis=1)/all_expo[index]            
        ymin=1
        ymax=yprofile.max()
        y0=np.where(yprofile==ymax)[0][0]
        they0.append(y0)
        axarr[iy,ix].semilogy(yprofile)
        axarr[iy,ix].semilogy([y0,y0],[ymin,ymax],'r-')
        axarr[iy,ix].semilogy([y0-w,y0-w],[ymin,ymax],'b-')
        axarr[iy,ix].semilogy([y0+w,y0+w],[ymin,ymax],'b-')
        axarr[iy,ix].semilogy([y0+ws[0],y0+ws[0]],[ymin,ymax],'k-')
        axarr[iy,ix].semilogy([y0+ws[1],y0+ws[1]],[ymin,ymax],'k-')
        axarr[iy,ix].semilogy([y0-ws[0],y0-ws[0]],[ymin,ymax],'k-')
        axarr[iy,ix].semilogy([y0-ws[1],y0-ws[1]],[ymin,ymax],'k-')
        axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].grid(True)
        if ylim is not None : axarr[iy,ix].set_ylim(ylim)
    title='Spectrum tranverse profile '.format(object_name)
    plt.suptitle(title,size=16)   
    return they0

#--------------------------------------------------------------------------------
    
def ExtractSpectra(they0,all_images,all_titles,object_name,all_expo,w=10,ws=80,right_edge=1800):
    """
    ShowTransverseProfile: Show the raw images without background subtraction
    =====================
    The goal is to see in y, where is the spectrum maximum. Returns they0
    
    - width of the band : 2 * w
    - Distance of the bad : ws[0]
    - width of the lateral band 2* ws[1]
    
    """
    NBIMAGES=len(all_images)

    thespectra= []
    thespectraUp=[]
    thespectraDown=[]
    
    for index in np.arange(0,NBIMAGES):
        data=np.copy(all_images[index])[:,0:right_edge]
        y0 = int(they0[index])
        spectrum2D=np.copy(data[y0-w:y0+w,:])
        xprofile=np.mean(spectrum2D,axis=0)
        
        ### Lateral bands to remove sky background
        ### ---------------------------------------
        Ny, Nx =  data.shape
        ymax = min(Ny,y0+ws[1])
        ymin = max(0,y0-ws[1])
        #spectrum2DUp=np.copy(data[y0-w+Dist:y0+w+Dist,:])
        spectrum2DUp=np.copy(data[y0+ws[0]:ymax,:])
        xprofileUp=np.median(spectrum2DUp,axis=0)#*float(ymax-ws[0]-y0)

        #spectrum2DDown=np.copy(data[y0-w-Dist:y0+w-Dist,:])
        spectrum2DDown=np.copy(data[ymin:y0-ws[0],:])
        xprofileDown=np.median(spectrum2DDown,axis=0)#*float(y0-ws[0]-ymin)
        
        if(all_expo[index]<=0):
            thespectra.append(xprofile)
            thespectraUp.append(xprofileUp)
            thespectraDown.append(xprofileDown)
        else:  ################## HERE I NORMALISE WITH EXPO TIME ####################################
            thespectra.append(xprofile/all_expo[index])
            thespectraUp.append(xprofileUp/all_expo[index]) 
            thespectraDown.append(xprofileDown/all_expo[index]) 
    
    return np.array(thespectra),np.array(thespectraUp),np.array(thespectraDown)


#---------------------------------------------------------------------------------

def ShowRightOrder(all_images,thex0,they0,all_titles,object_name,all_expo,dir_top_images,NBIMGPERROW=2):
    """
    ShowRawImages: Show the raw images without background subtraction
    ==============
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,4*MAXIMGROW))
    f.tight_layout()
    
    right_edge = 1800
    
    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        full_image=np.copy(all_images[index])[:,0:right_edge]
        y_0=they0[index]
        x_0=thex0[index]

        reduc_image=full_image[y_0-20:y_0+20,x_0+100:right_edge]/all_expo[index]
        
        X,Y=np.meshgrid(np.arange(0,reduc_image.shape[1]),np.arange(reduc_image.shape[0]))
        im = axarr[iy,ix].pcolormesh(X,Y,reduc_image, cmap='rainbow',vmin=0,vmax=100)
        #axarr[iy,ix].colorbar(im, orientation='vertical')
        axarr[iy,ix].axis([X.min(), X.max(), Y.min(), Y.max()]); axarr[iy,ix].grid(True)
        
        axarr[iy,ix].set_title(all_titles[index])
        
    
    title='Right part of spectrum of {} '.format(object_name)
    plt.suptitle(title,size=16)
    figfilename=os.path.join(dir_top_images,'rightorder.pdf')
    
    #plt.savefig(figfilename)  

#------------------------------------------------------------------------------------

def ShowLeftOrder(all_images,thex0,they0,all_titles,object_name,all_expo,dir_top_images,NBIMGPERROW=2):
    """
    ShowRawImages: Show the raw images without background subtraction
    ==============
    """
    NBIMAGES=len(all_images)
    MAXIMGROW=(NBIMAGES-1) / NBIMGPERROW +1
    
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,4*MAXIMGROW))
    f.tight_layout()

    for index in np.arange(0,NBIMAGES):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        full_image=np.copy(all_images[index])
        y_0=they0[index]
        x_0=thex0[index]
        
        
        reduc_image=full_image[y_0-20:y_0+20,0:x_0-100]/all_expo[index] 

        X,Y=np.meshgrid(np.arange(0,reduc_image.shape[1]),np.arange(reduc_image.shape[0]))
        im = axarr[iy,ix].pcolormesh(X,Y,reduc_image, cmap='rainbow',vmin=0,vmax=30)
        #axarr[iy,ix].colorbar(im, orientation='vertical')
        axarr[iy,ix].axis([X.min(), X.max(), Y.min(), Y.max()]); axarr[iy,ix].grid(True)
        
        axarr[iy,ix].set_title(all_titles[index])
        
    
    title='Left part of spectrum of '.format(object_name)
    plt.suptitle(title,size=16)
    figfilename=os.path.join(dir_top_images,'leftorder.pdf')
    #plt.savefig(figfilename)  

#-----------------------------------------------------------------------------------

def CleanBadPixels(spectraUp,spectraDown):
    """
    CleanBadPixels
    --------------
    
    Try to remove bad pixels on top/down 
    
    """
    
    Clean_Up= []
    Clean_Do = []
    Clean_Av = []
    eps=25.   # this is the minumum background Please check
    NBSPEC=len(spectraUp)
    for index in np.arange(0,NBSPEC):
        s_up=spectraUp[index]
        s_do=spectraDown[index]
    
        index_up=np.where(s_up<eps)
        index_do=np.where(s_do<eps)
        
        s_up[index_up]=s_do[index_up]
        s_do[index_do]=s_up[index_do]
        s_av=(s_up+s_do)/2.
        
        Clean_Up.append(s_up)
        Clean_Do.append(s_do)
        Clean_Av.append(s_av)
        
    return Clean_Up, Clean_Do,Clean_Av

#-----------------------------------------------------------------------------------

def ShowLongitBackground(spectra,spectraUp,spectraDown,spectraAv,all_titles,all_filt,object_name,NBIMGPERROW=2,right_edge=1800):
    """
    Show the background to be removed to the spectrum
    """
    NBSPEC=len(spectra)
    MAXIMGROW=(NBSPEC-1) / NBIMGPERROW +1

    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,5*MAXIMGROW))
    f.tight_layout()
    for index in np.arange(0,NBSPEC):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        axarr[iy,ix].plot(spectra[index],'r-')
        axarr[iy,ix].plot(spectraUp[index],'b-')
        axarr[iy,ix].plot(spectraDown[index],'g-')
        axarr[iy,ix].plot(spectraAv[index],'m-')
        axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].grid(True)
        axarr[iy,ix].set_ylim(0.,spectra[index][:right_edge].max()*1.2)
        axarr[iy,ix].annotate(all_filt[index],xy=(0.05,0.9),xytext=(0.05,0.9),verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
    title='Longitudinal background Up/Down'.format(object_name)
    plt.suptitle(title,size=16)

#---------------------------------------------------------------------------------
    
def CorrectSpectrumFromBackground(spectra, background):
    """
    Background Subtraction
    """
    NBSPEC=len(spectra)
        
    corrected_spectra = []
    
    for index in np.arange(0,NBSPEC):
        corrspec=spectra[index]-background[index]
        corrected_spectra.append(corrspec)
    return corrected_spectra

#--------------------------------------------------------------------------------
    
def ShowSpectrumProfile(spectra,all_titles,object_name,all_filt,NBIMGPERROW=2,xlim=None,vertical_lines=None):
    """
    ShowSpectrumProfile: Show the raw images without background subtraction
    =====================
    """
    NBSPEC=len(spectra)
    MAXIMGROW=(NBSPEC-1) / NBIMGPERROW +1
    
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,5*MAXIMGROW))
    f.tight_layout()
    for index in np.arange(0,NBSPEC):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW
        axarr[iy,ix].plot(spectra[index],'r-')
        axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].grid(True)
        axarr[iy,ix].set_ylim(0.,spectra[index][:IMSIZE].max()*1.2)
        if xlim is not None :
            if type(xlim) is not list :
                axarr[iy,ix].set_xlim(xlim)
                axarr[iy,ix].set_ylim(0.,spectra[index][xlim[0]:xlim[1]].max()*1.2)
            else :
                axarr[iy,ix].set_xlim(xlim[index])
                axarr[iy,ix].set_ylim(0.,spectra[index][xlim[index][0]:xlim[index][1]].max()*1.2)                
        axarr[iy,ix].annotate(all_filt[index],xy=(0.05,0.9),xytext=(0.05,0.9),verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
        if vertical_lines is not None :
            axarr[iy,ix].axvline(vertical_lines[index],color='k',linestyle='--',lw=2)
    title='Spectrum 1D profile and background Up/Down for {}'.format(object_name)
    plt.suptitle(title,size=16)

#----------------------------------------------------------------------------------------
    
def SpectrumAmplitudeRatio(spectra):
    """
    SpectrumAmplitudeRatio: ratio of amplitudes
    =====================
    """
    ratio_list= []
    
    NBSPEC=len(spectra)
    
    for index in np.arange(0,NBSPEC):
       
        max_right=spectra[index][700:1900].max()
        max_left=spectra[index][:700].max()
        
        ratio=max_right/max_left
        ratio_list.append(ratio) 
        
    return ratio_list

#-----------------------------------------------------------------------------------------

def ShowSpectrumProfileFit(spectra,all_titles,object_name,all_filt,NBIMGPERROW=2,xlim=(1200,1600),guess=[10,1400,200],vertical_lines=None):
    """
    ShowRightSpectrumProfile: Show the raw images without background subtraction
    =====================
    """
    NBSPEC=len(spectra)
    MAXIMGROW=(NBSPEC-1) / NBIMGPERROW +1
    
    
    f, axarr = plt.subplots(MAXIMGROW,NBIMGPERROW,figsize=(25,5*MAXIMGROW))
    f.tight_layout()
    for index in np.arange(0,NBSPEC):
        ix=index%NBIMGPERROW
        iy=index/NBIMGPERROW

        if type(xlim[0]) is list :
            left_edge = int(xlim[index][0])
            right_edge = int(xlim[index][1])
            guess[1] = 0.5*(left_edge+right_edge)
        else :
            left_edge = int(xlim[0])
            right_edge = int(xlim[1])
        xs = np.arange(left_edge,right_edge,1)
        right_spectrum = spectra[index][left_edge:right_edge]
        axarr[iy,ix].plot(xs,right_spectrum,'r-',lw=2)
        if right_edge - left_edge > 10 :
            popt, pcov = EmissionLineFit(spectra[index],left_edge,right_edge,guess=guess)
        
            axarr[iy,ix].plot(xs,gauss(xs,*popt),'b-')
            axarr[iy,ix].axvline(popt[1],color='b',linestyle='-',lw=2)    
            print '%s:\t gaussian center x=%.2f+/-%.2f' % (all_filt[index],popt[1],np.sqrt(pcov[1,1]))
        axarr[iy,ix].set_title(all_titles[index])
        axarr[iy,ix].grid(True)
        axarr[iy,ix].set_ylim(0.,right_spectrum.max()*1.2)
        axarr[iy,ix].set_xlim(left_edge,right_edge)
        axarr[iy,ix].annotate(all_filt[index],xy=(0.05,0.9),xytext=(0.05,0.9),verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
        if vertical_lines is not None :
            axarr[iy,ix].axvline(vertical_lines[index],color='k',linestyle='--',lw=2)    
    title='Spectrum 1D profile and background Up/Down for {}'.format(object_name)
    plt.suptitle(title,size=16)
    
#-------------------------------------------------------------------------------------------
#  Sylvie : October 2017
#---------------------------------------------------------------------------------------
    
def get_filt_idx(listoffilt):
    """
    get_filt_idx::
    -------------    
        sort the index of the image according the disperser used.
        it assumes the filter is called "dia".
        The diserser names are pre-defined in the array Filt_names.
        
        input: 
            - listoffilt : list of filter-disperser name from the image header
        output:
            - filt0_idx
            - filt1_idx
            - filt2_idx
            - filt3_idx
            - filt4_idx
            - filt5_idx
            - filt6_idx
            for each kind of disperser, the list index in the listoffilt
      
    """
    
    filt0_idx=[]
    filt1_idx=[]
    filt2_idx=[]
    filt3_idx=[]
    filt4_idx=[]
    filt5_idx=[]
    filt6_idx=[]
    
    index=0
    for filt in listoffilt:
        if filt == 'dia Ron400' or filt == 'RG715 Ron400' or  filt == 'FGB37 Ron400':
            filt0_idx.append(index)
        elif filt == 'dia Thor300' or filt == 'RG715 Thor300' or  filt == 'FGB37 Thor300':
            filt1_idx.append(index)
        elif filt == 'dia HoloPhP' or filt == 'RG715 HoloPhP' or  filt == 'FGB37 HoloPhP':
            filt2_idx.append(index)
        elif filt == 'dia HoloPhAg' or filt == 'RG715 HoloPhAg' or  filt == 'FGB37 HoloPhAg':
            filt3_idx.append(index)
        elif filt == 'dia HoloAmAg' or filt == 'RG715 HoloAmAg' or  filt == 'FGB37 HoloAmAg':
            filt4_idx.append(index)
        elif filt == 'dia Ron200' or filt == 'RG715 Ron200' or  filt == 'FGB37 Ron200':
            filt5_idx.append(index)
        else :
            print ' common_notebook::get_filt_idx unknown:  filter-disperser ',filt
            filt6_idx.append(index)
    
        index+=1
    
    filt0_idx=np.array(filt0_idx)
    filt1_idx=np.array(filt1_idx)
    filt2_idx=np.array(filt2_idx)
    filt3_idx=np.array(filt3_idx)
    filt4_idx=np.array(filt4_idx)
    filt5_idx=np.array(filt5_idx)
    filt6_idx=np.array(filt6_idx)
    
    return filt0_idx,filt1_idx,filt2_idx,filt3_idx,filt4_idx,filt5_idx,filt6_idx

#------------------------------------------------------------------------------------
def get_disperser_filtname(filt):
    """
    
    """
    if filt == 'dia Ron400' or filt == 'RG715 Ron400' or  filt == 'FGB37 Ron400':
        return 'Ron400'
    elif filt == 'dia Thor300' or filt == 'RG715 Thor300' or  filt == 'FGB37 Thor300':
        return 'Thor300'
    elif filt == 'dia HoloPhP' or filt == 'RG715 HoloPhP' or  filt == 'FGB37 HoloPhP':
        return 'HoloPhP'
    elif filt == 'dia HoloPhAg' or filt == 'RG715 HoloPhAg' or  filt == 'FGB37 HoloPhAg':
        return 'HoloPhAg'
    elif filt == 'dia HoloAmAg' or filt == 'RG715 HoloAmAg' or  filt == 'FGB37 HoloAmAg':
        return 'HoloAmAg'
    elif filt == 'dia Ron200' or filt == 'RG715 Ron200' or  filt == 'FGB37 Ron200':
        return 'Ron200'
    else :
        print ' common_notebook::get_filt_idx unknown:  filter-disperser ',filt
    return 'unknown'
    

#------------------------------------------------------------------------------------
def guess_init_fit(theimage,xmin=0,xmax=-1,ymin=0,ymax=-1):
    """
    guess_init_fit::
    ---------------
        quick search of a local maximum in an image
        
        input:
            - theimage : 2D numpy array
            - xmin,xmax,ymin,ymax : the subrange where to search the maximum
        output:
            -x,y coordinate where the maximum has been found in the original coordinates of the
            image
    """
    cropped_image=np.copy(theimage[ymin:ymax,xmin:xmax])
    
    profile_y=np.sum(cropped_image,axis=1)
    profile_x=np.sum(cropped_image,axis=0)
     
    theidxmax=np.where(profile_x==profile_x.max())
    theidymax=np.where(profile_y==profile_y.max())
    
    return xmin+theidxmax[0][0],ymin+theidymax[0][0]
#-----------------------------------------------------------------------------
    
def check_bad_guess(xy_guess,filt_idx, sigma_cut=10.):
    """
    function check_bad_guess(xy_guess,filt_idx)
    
    check is the x or y position are too far from the series of other x,y postion for a given disperser
    
    input :
       xy_guess : the x or the y values to be tested
       filt_idx, the list of identifiers of a given disperser
       sigma_cut : typical distance accepted from the group
       
    output:
       the_mean : average position
       the_std. : std deviation
       the_bad_idx : the list of identiers that are at more than 3 sigma 
    """
    
    # typical dispersion
    
    # extract of (x,y) from the set of disperser id 
    the_guess=xy_guess[filt_idx]
    
    # average and distance from the filt_idx group
    the_mean=np.median(the_guess)
    the_std=np.std(the_guess-the_mean)
    
    the_bad=np.where( np.abs(the_guess-the_mean)> 3.*sigma_cut)
    
    the_bad_idx=filt_idx[the_bad]
    
    return int(the_mean),int(the_std),the_bad_idx
#-----------------------------------------------------------------------------
def remove_from_bad(arr,index_to_remove):
    """
    remove_from_bad(arr,index_to_remove)
    ------------------------------------
    
    Remove the index_to_reove from array arr
    
    input:
        - array arr
        - index_to_remove 
        
    output:
        - the array with the index_to_remove removed
        
    """
    
    newarr=arr
    set_rank_to_remove=np.where(arr==index_to_remove)[0]
    if len(set_rank_to_remove)!=0:
        rank_to_remove=set_rank_to_remove[0]
        newarr=np.delete(arr,rank_to_remove)
    return newarr
#-------------------------------------------------------------------------------
    

def guess_central_position(listofimages,DeltaX,DeltaY,dwc,filt0_idx,filt1_idx,filt2_idx,filt3_idx,filt4_idx,filt5_idx,filt6_idx,check_quality_flag=False,sigmapix_quality_cut=10,qualitytag=None):
    """
    guess_central_position:
    ----------------------
    Guess the central position of the star
    
    input:
        - listofimages: list of images
        - DeltaX,DeltaY : [xmin,xmax] and [ymin,ymax] region in which we expect center
        - dwc :width around the region
        - filt0_idx,filt1_idx,filt2_idx,filt3_idx,filt4_idx,filt5_idx : set of indexes
    
    output :
         x_guess,y_guess : coordinate of the central star in the frame of the raw image
    """

    
    # Step 1 : do the 2D guassian fit
    
    x_guess = [] 
    y_guess = []
    index=0

    # loop on images
    for theimage in listofimages:
               
        index+=1
        
        
        # try to find a maximum in the region specified here
        # we expect the central star be found at x0c, y0c
        # overwrite x0c and y0c here !!!!!
        x0c,y0c=guess_init_fit(theimage,DeltaX[0],DeltaX[1],DeltaY[0],DeltaY[1])
        
        # sub-image around the maximum found at center
        # the coordinate of the lower left corner is (x0c-dwc,y0c-dwc) in original coordinate
        sub_image=np.copy(theimage[y0c-dwc:y0c+dwc,x0c-dwc:x0c+dwc]) # make a sub-image
    
        # init the gaussian fit
        NY=sub_image.shape[0]
        NX=sub_image.shape[1]
        y, x = np.mgrid[:NY,:NX]
        z=sub_image[y,x]
    
        # we expect the central star is at the center of the subimage
        x_mean=NX/2
        y_mean=NY/2
        z_max=z[y_mean,x_mean]
    
        # do the gaussian fit
        p_init = models.Gaussian2D(amplitude=z_max,x_mean=x_mean,y_mean=y_mean)
        fit_p = fitting.LevMarLSQFitter()
    
        p = fit_p(p_init, x, y, z)
    
        x_fit= p.x_mean
        y_fit= p.y_mean
        z_fit= p.amplitude
    
       
    
        # put the center found by fit in the original image coordinate system
        #--------------------------------------------------------------------
        
        x_star_original=x0c-dwc+x_fit
        y_star_original=y0c-dwc+y_fit
        
        x_guess.append(x_star_original)
        y_guess.append(y_star_original)
        
        #if index%5==0 :
        print index-1,': (x_guess,y_guess)=',x_star_original,y_star_original

    x_guess=np.array(x_guess)
    y_guess=np.array(y_guess)
    
    
    # Step 2 : check if the fit is creazy 
    # if so, use the average center for the given disperser
    
  
    
    # find bad ids for filter 1 and correct for 
        
    if check_quality_flag: 
        print ' ==========================='
        print 'Check fit quality :: '
        print ' ==========================='
        
        # filter 0
        #------------------------
        if filt0_idx.shape[0] >0:        
            aver_x0,std_x0,bad_idx_x0=check_bad_guess(x_guess,filt0_idx)
            if (bad_idx_x0.shape[0] != 0):
                print 'bad filt 0 x : ',bad_idx_x0
                x_guess[bad_idx_x0]=aver_x0    # do the correction
    
            aver_y0,std_y0,bad_idx_y0=check_bad_guess(y_guess,filt0_idx)
            if (bad_idx_y0.shape[0] != 0):
                print 'bad filt 0 y : ',bad_idx_y0
                y_guess[bad_idx_y0]=aver_y0    # do the correction     
        

        # filter 1
        #--------------------------
        if filt1_idx.shape[0]>0:    
            aver_x1,std_x1,bad_idx_x1=check_bad_guess(x_guess,filt1_idx)
            if (bad_idx_x1.shape[0] != 0):
                print 'bad filt 1 x : ',bad_idx_x1
                # !!!!!!!!!!!!!!!!!!!!!! Special for first Thorlab image
                # !!!!!!! 30 jun 17 on HD111980 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if qualitytag=="ana_30may17_hd111980":
                    idx_to_remove=0
                    print 'remove from bad idx x1 : ',idx_to_remove
                    bad_idx_x1=remove_from_bad(bad_idx_x1,idx_to_remove)           
                    print 'new bad filt 1 x : ',bad_idx_x1
                
                x_guess[bad_idx_x1]=aver_x1    # do the correction
    
            aver_y1,std_y1,bad_idx_y1=check_bad_guess(y_guess,filt1_idx)
            if (bad_idx_y1.shape[0] != 0):
                print 'bad filt 1 y : ',bad_idx_y1
                # !!!!!!!!!!!!!!!!!!!!!! Special for first Thorlab image
                # !!!!!!! 30 jun 17 on HD111980 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if qualitytag=="ana_30may17_hd111980":
                    idx_to_remove=0        
                    print 'remove from bad idx y1 : ',idx_to_remove
                    bad_idx_y1=remove_from_bad(bad_idx_y1,idx_to_remove)           
                    print 'new bad filt 1 y : ',bad_idx_y1
                    
                y_guess[bad_idx_y1]=aver_y1    # do the correction
            
    
 
        # filter 2
        #-------------------
        if filt2_idx.shape[0]>0:          
            aver_x2,std_x2,bad_idx_x2=check_bad_guess(x_guess,filt2_idx)
            if (bad_idx_x2.shape[0] != 0):
                print 'bad filt 2 x : ',bad_idx_x2
                x_guess[bad_idx_x2]=aver_x2    # do the correction
    
            aver_y2,std_y2,bad_idx_y2=check_bad_guess(y_guess,filt2_idx)
            if (bad_idx_y2.shape[0] != 0):
                print 'bad filt 2 y : ',bad_idx_y2
                y_guess[bad_idx_y2]=aver_y2    # do the correction
        
        
        # filter 3
        #-----------
        if filt3_idx.shape[0]>0:  
            aver_x3,std_x3,bad_idx_x3=check_bad_guess(x_guess,filt3_idx)
            if (bad_idx_x3.shape[0] != 0):
                print 'bad bad filt 3 x : ',bad_idx_x3
                x_guess[bad_idx_x3]=aver_x3    # do the correction
    
            aver_y3,std_y3,bad_idx_y3=check_bad_guess(y_guess,filt3_idx)
            if (bad_idx_y3.shape[0] != 0):
                print 'bad filt 3 y : ',bad_idx_y3
                y_guess[bad_idx_y3]=aver_y3    # do the correction
        
        # filter 4
        #----------------
        if filt4_idx.shape[0]>0:          
            aver_x4,std_x4,bad_idx_x4=check_bad_guess(x_guess,filt4_idx)
            if (bad_idx_x4.shape[0] != 0):
                print 'bad filt 4 x : ',bad_idx_x4
                x_guess[bad_idx_x4]=aver_x4    # do the correction
    
            aver_y4,std_y4,bad_idx_y4=check_bad_guess(y_guess,filt4_idx)
            if (bad_idx_y4.shape[0] != 0):
                print 'bad filt 4 y : ',bad_idx_y4
                y_guess[bad_idx_y4]=aver_y4    # do the correction        
 
    
        # filter 5
        #-----------------
        if filt5_idx.shape[0]>0:  
          
            aver_x5,std_x5,bad_idx_x5=check_bad_guess(x_guess,filt5_idx)
            if (bad_idx_x5.shape[0] != 0):
                print 'bad filt 5 x : ',bad_idx_x5
                x_guess[bad_idx_x5]=aver_x5    # do the correction
    
            aver_y5,std_y5,bad_idx_y5=check_bad_guess(y_guess,filt5_idx)
            if (bad_idx_y5.shape[0] != 0):
                print 'bad filt 5 y : ',bad_idx_y5
                y_guess[bad_idx_y5]=aver_y5    # do the correction  
            
            # filter 6 
            #---------------
        if filt6_idx.shape[0]> 0:       
            aver_x6,std_x6,bad_idx_x6=check_bad_guess(x_guess,filt6_idx)
            if (bad_idx_x6.shape[0] != 0):
                print 'bad filt 6 x : ',bad_idx_x6
                x_guess[bad_idx_x6]=aver_x6    # do the correction
    
            aver_y6,std_y6,bad_idx_y6=check_bad_guess(y_guess,filt6_idx)
            if (bad_idx_y6.shape[0] != 0):
                print 'bad filt 6 y : ',bad_idx_y6
                y_guess[bad_idx_y6]=aver_y6    # do the correction  
    
    
    return x_guess,y_guess

#-----------------------------------------------------------------------------------
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    
    For example for the PSF
    
    x=pixel number
    y=Intensity in pixel
    
    values-x
    weights=y=f(x)
    
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))

#---------------------------------------------------------------------------------

def check_central_star(all_images,x_star0,y_star0,all_titles,all_filt,Dx=100,Dy=50):
    """
    check_central_star(all_images,x_star0,y_star0,all_titles)
    --------------------------------------------------------
    
    Try to localize very precisely the order 0 central star.
    We calculate the average, by giving the very high weigh to the pixels having 
    a great intentity. (power 4 of the intensity)
    
    input:
    - all_images : the list of images
    - x_star0, y_star_0 : original guess of the central star, not very accurate within Dx,Dy
    - all_titles, all_filt : info for the title
    - Dx,Dy : range allowed aroud the original central star (do not include dispersed spectrum wing)
    
    output : arrays of accurate X and Y positions
    
    
    """
    index=0
    
    x_star = []
    y_star = []
    
    for image in all_images:
        x0=int(x_star0[index])
        y0=int(y_star0[index])
        
        old_x0=x0-(x0-Dx)
        old_y0=y0-(y0-Dy)
        
        sub_image=np.copy(image[y0-Dy:y0+Dy,x0-Dx:x0+Dx])
        NX=sub_image.shape[1]
        NY=sub_image.shape[0]
        
        profile_X=np.sum(sub_image,axis=0)
        profile_Y=np.sum(sub_image,axis=1)
        X_=np.arange(NX)
        Y_=np.arange(NY)
    
        profile_X_max=np.max(profile_X)*1.2
        profile_Y_max=np.max(profile_Y)*1.2
    
        avX,sigX=weighted_avg_and_std(X_,profile_X**4) ### better if weight squared
        avY,sigY=weighted_avg_and_std(Y_,profile_Y**4) ### really avoid plateau contribution
        #print index,'\t',avX,avY,'\t',sigX,sigY
    
        f, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(20,4))

        ax1.imshow(sub_image,origin='lower',vmin=0,vmax=10000,cmap='rainbow')
        ax1.plot([avX],[avY],'ko')
        ax1.grid(True)
        ax1.set_xlabel('X - pixel')
        ax1.set_ylabel('Y - pixel')
    
        ax2.plot(X_,profile_X,'r-',lw=2)
        ax2.plot([old_x0,old_x0],[0,profile_X_max],'y-',label='old',lw=2)
        ax2.plot([avX,avX],[0,profile_X_max],'b-',label='new',lw=2)
        
        
        ax2.grid(True)
        ax2.set_xlabel('X - pixel')
        ax2.legend(loc=1)
        
        ax3.plot(Y_,profile_Y,'r-',lw=2)
        ax3.plot([old_y0,old_y0],[0,profile_Y_max],'y-',label='old',lw=2)
        ax3.plot([avY,avY],[0,profile_Y_max],'b-',label='new',lw=2)
        
        ax3.grid(True)
        ax3.set_xlabel('Y - pixel')
        ax3.legend(loc=1)
        
    
        thetitle="{} : {} , {} ".format(index,all_titles[index],all_filt[index])
        f.suptitle(thetitle, fontsize=16)
    
        theX=x0-Dx+avX
        theY=y0-Dy+avY
        
        x_star.append(theX)
        y_star.append(theY)
    
    
        index+=1
        
    x_star=np.array(x_star)
    y_star=np.array(y_star)
        
    return x_star,y_star
#----------------------------------------------------------------------------------------------------------------
#  Ana2DShapeSpectra
#------------------------------------------------------------------------------------------------------------    

def Pixel_To_Lambdas(grating_name,X_Size_Pixels,pointing,verboseflag):
    """
    Pixel_To_Lambdas:
    -----------------
    
    Convert pixels into wavelengths
    
    input:
        - grating_name : name of the disperser in calibration tools (Hologram Class)
        - X_Size_Pixels : array of pixels numbers
        - all_pointing : position of order 0 in original raw image
        - verboseflag : Verbose flag for Hologram Class
        
    output
        - lambdas : return wavelengths
    
    """
    
    if grating_name=='Ron200':
        holo = Hologram('Ron400',verbose=verboseflag)
    else:    
        holo = Hologram(grating_name,verbose=verboseflag)
    lambdas=holo.grating_pixel_to_lambda(X_Size_Pixels,pointing)
    if grating_name=='Ron200':
        lambdas=lambdas*2.
    return lambdas
#-------------------------------------------------------------------------------------------
        


def ShowOneContour(index,all_images,all_pointing,thex0,they0,all_titles,object_name,all_expo,dir_top_img,all_filt,figname):
    """
    ShowOneContour(index,all_images,all_pointing,all_titles,object_name,all_expo,dir_top_img,all_filt,figname)
    --------------
    
    Show contour lines of 2D spectrum for one image
    
    input:
        - index: selected index
        - all_images : all set of cut and rotated images
        - all_pointing : list of reference to find hologram and grater parameter for calibration
        
        - thex0, they0 : list of where is the central star in the image
        - all_titles : list of title of the image
        - object_name : list of object name
        - all_expo : list of exposure time
        - dir_top_img : directory to save the image
        - all_filt : list of filter-disperser name
        - figname : filename of figure
        
    output: the image 
    
    """
    plt.figure(figsize=(15,6))
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    
    YMIN=-15
    YMAX=15
    
    figfilename=os.path.join(dir_top_img,figname)   
    
    #center is approximately the one on the original raw image (may be changed)
    #x0=int(all_pointing[index][0])
    x0=int(thex0[index])
   
    
    # Extract the image    
    full_image=np.copy(all_images[index])
    
    # refine center in X,Y
    star_region_X=full_image[:,x0-star_halfwidth:x0+star_halfwidth]
    
    profile_X=np.sum(star_region_X,axis=0)
    profile_Y=np.sum(star_region_X,axis=1)

    NX=profile_X.shape[0]
    NY=profile_Y.shape[0]
    
    X_=np.arange(NX)
    Y_=np.arange(NY)
    
    avX,sigX=weighted_avg_and_std(X_,profile_X**4) # take squared on purpose (weigh must be >0)
    avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)
    
    x0=int(avX+x0-star_halfwidth)
      
    
    # find the center in Y on the spectrum
    yprofile=np.sum(full_image[:,spec_index_min:spec_index_max],axis=1)
    y0=np.where(yprofile==yprofile.max())[0][0]

    # cut the image in vertical and normalise by exposition time
    reduc_image=full_image[y0-20:y0+20,x0:spec_index_max]/all_expo[index] 
    reduc_image[:,0:100]=0  # erase central star
    
    X_Size_Pixels=np.arange(0,reduc_image.shape[1])
    Y_Size_Pixels=np.arange(0,reduc_image.shape[0])
    Transverse_Pixel_Size=Y_Size_Pixels-int(float(Y_Size_Pixels.shape[0])/2.)
    
    # calibration in wavelength
    #grating_name=all_filt[index].replace('dia ','')
    grating_name=get_disperser_filtname(all_filt[index])
    
    lambdas=Pixel_To_Lambdas(grating_name,X_Size_Pixels,all_pointing[index],True)
    
    #if grating_name=='Ron200':
    #    holo = Hologram('Ron400',verbose=True)
    #else:    
    #    holo = Hologram(grating_name,verbose=True)
    #lambdas=holo.grating_pixel_to_lambda(X_Size_Pixels,all_pointing[index])
    #if grating_name=='Ron200':
    #    lambdas=lambdas*2.
        

    X,Y=np.meshgrid(lambdas,Transverse_Pixel_Size)     
    T=np.transpose(reduc_image)
        
        
    plt.contourf(X, Y, reduc_image, 20, alpha=.75, cmap='jet',origin='lower')
    C = plt.contour(X, Y, reduc_image , 20, colors='black', linewidth=.5,origin='lower')
        
    
    for line in LINES:
        if line == O2 or line == HALPHA or line == HBETA or line == HGAMMA:
            plt.plot([line['lambda'],line['lambda']],[YMIN,YMAX],'-',color='lime',lw=0.5)
            plt.text(line['lambda'],YMAX-3,line['label'],verticalalignment='bottom', horizontalalignment='center',color='lime', fontweight='bold',fontsize=16)
    
    
    
    plt.axis([X.min(), X.max(), Y.min(), Y.max()]); plt.grid(True)
    plt.title(all_titles[index])
    plt.grid(color='white', ls='solid')
    plt.text(200,-5.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='yellow', fontweight='bold',fontsize=16)
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('pixels')
    plt.ylim(YMIN,YMAX)
    plt.xlim(0.,1200.)
    plt.savefig(figfilename)
    
#-------------------------------------------------------------------------------------------------------------------------------


def ShowOneOrder_contour(all_images,all_pointing,thex0,they0,all_titles,object_name,all_expo,dir_top_img,all_filt,figname):
    """
    ShowOneOrder_contour:      
    ====================
    
    Show the contour lines of 2D-Spectrum order +1 for each images

    
    input:
        - index: selected index
        - all_images : all set of cut and rotated images
        - all_pointing : list of reference to find hologram and grater parameter for calibration
        
        - thex0, they0 : list of where is the central star in the image
        - all_titles : list of title of the image
        - object_name : list of object name
        - all_expo : list of exposure time
        - dir_top_img : directory to save the image
        - all_filt : list of filter-disperser name
        - figname : filename of figure
        
    output: 
        all the image in a pdf file 
    
    """
    NBIMGPERROW=2
    NBIMAGES=len(all_images)
    MAXIMGROW=max(2,m.ceil(NBIMAGES/NBIMGPERROW))
    
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    
    
    YMIN=-10
    YMAX=10
    
    figfilename=os.path.join(dir_top_img,figname)   
    title='Images of {}'.format(object_name)
    
    
     # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    
    
    for index in np.arange(0,NBIMAGES):
        
      
        
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(title,size=20)
            
        # index of image in the page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
         
        
        
        #center is approximately the one on the original raw image (may be changed)  
        x0=int(thex0[index])
    
        
        # Extract the image    
        full_image=np.copy(all_images[index])
        
        # refine center in X,Y
        star_region_X=np.copy(full_image[:,x0-star_halfwidth:x0+star_halfwidth])
        
        profile_X=np.sum(star_region_X,axis=0)
        profile_Y=np.sum(star_region_X,axis=1)
        
        NX=profile_X.shape[0]
        NY=profile_Y.shape[0]

        X_=np.arange(NX)
        Y_=np.arange(NY)
    
        avX,sigX=weighted_avg_and_std(X_,profile_X**4) # take squared on purpose (weigh must be >0)
        avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)
    
        x0=int(avX+x0-star_halfwidth)
       
        
    
        # find the center in Y
        yprofile=np.sum(full_image[:,spec_index_min:spec_index_max],axis=1)
        y0=np.where(yprofile==yprofile.max())[0][0]
       
        
        

        # cut the image to have right spectrum (+1 order)
        # the origin is the is the star center
        reduc_image=np.copy(full_image[y0-20:y0+20,x0:spec_index_max])/all_expo[index] 
        reduc_image[:,0:100]=0  # erase central star
    
   
    
        X_Size_Pixels=np.arange(0,reduc_image.shape[1])
        Y_Size_Pixels=np.arange(0,reduc_image.shape[0])
        
        Transverse_Pixel_Size=Y_Size_Pixels-int(float(Y_Size_Pixels.shape[0])/2.)
    
        # calibration of wavelength
        #grating_name=all_filt[index].replace('dia ','')
        grating_name=get_disperser_filtname(all_filt[index])
        lambdas=Pixel_To_Lambdas(grating_name,X_Size_Pixels,all_pointing[index],False)
        
        #if grating_name=='Ron200':
        #     holo = Hologram('Ron400',verbose=False)
        #else:    
        #    holo = Hologram(grating_name,verbose=False)
        #lambdas=holo.grating_pixel_to_lambda(X_Size_Pixels,all_pointing[index])
        #if grating_name=='Ron200':
        #    lambdas=lambdas*2.
        
    
        X,Y=np.meshgrid(lambdas,Transverse_Pixel_Size)     
        T=np.transpose(reduc_image)
                   
        
        
        axarr[iy,ix].contourf(X, Y, reduc_image, 10, alpha=.75, cmap='jet')
        C = axarr[iy,ix].contour(X, Y, reduc_image , 10, colors='black', linewidth=.5)
        
        for line in LINES:
            if line == O2 or line == HALPHA or line == HBETA or line == HGAMMA:
                axarr[iy,ix].plot([line['lambda'],line['lambda']],[YMIN,YMAX],'-',color='lime',lw=0.5)
                axarr[iy,ix].text(line['lambda'],YMAX-3,line['label'],verticalalignment='bottom', horizontalalignment='center',color='lime', fontweight='bold',fontsize=16)
        
        
        axarr[iy,ix].axis([X.min(), X.max(), Y.min(), Y.max()]); 
        axarr[iy,ix].grid(True)
        axarr[iy,ix].set_title(all_titles[index])
    
        axarr[iy,ix].grid(color='white', ls='solid')
        axarr[iy,ix].text(200,-5.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='yellow', fontweight='bold',fontsize=16)
        
        
        axarr[iy,ix].set_xlabel('$\lambda$ (nm)')
        axarr[iy,ix].set_ylabel('pixels')
        axarr[iy,ix].set_ylim(YMIN,YMAX)
        axarr[iy,ix].set_xlim(0.,1100.)
        
        
        # save a new page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            print "pdf Page written ",PageNum
            f.show()
        
          
    
    f.savefig(pp, format='pdf') 
    print "Final pdf Page written ",PageNum
    f.show()
    pp.close()  

#---------------------------------------------------------------------------------------------------------------------------------------------
    
def ShowManyTransverseSpectrum(index,all_images,all_pointing,thex0,they0,all_titles,object_name,all_expo,dir_top_img,all_filt,figname):
    """
    ShowManyTransverseSpectrum:
    ---------------------------
    
    Show the transverse profile in different wavelength bands. Notice the background is subtracted to have a correct
    FWHM calculation
    
    input:
        - index: selected index
        - all_images : all set of cut and rotated images
        - all_pointing : list of reference to find hologram and grater parameter for calibration
        
        - thex0, they0 : list of where is the central star in the image
        - all_titles : list of title of the image
        - object_name : list of object name
        - all_expo : list of exposure time
        - dir_top_img : directory to save the image
        - all_filt : list of filter-disperser name
        - figname : filename of figure
        
    output: 
        - the image of transverse spectra
    
    """
    
    
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    
    # bands in wavelength
    wlmin=np.array([400,500,600,700,800,900.])
    wlmax=np.array([500,600,700,800,900,1000.])
    
    #wlmin=np.array([400,450,500,550,600,650,700,750,800,850,900,950])
    #wlmax=np.array([450,500,550,600,650,700,750,800,850,900,950,1000])
    
    # titles
    thetitle=all_titles[index]+' '+all_filt[index]
    
    NBANDS=wlmin.shape[0]
    
    figfilename=os.path.join(dir_top_img,figname)  
    plt.figure(figsize=(16,6))
       
    
    #center is approximately the one on the original raw image (may be changed)  
    x0=int(thex0[index])
    
        
    # Extract the image    
    full_image=np.copy(all_images[index])
        
    # refine center in X,Y
    star_region_X=np.copy(full_image[:,x0-star_halfwidth:x0+star_halfwidth])
        
    profile_X=np.sum(star_region_X,axis=0)
    profile_Y=np.sum(star_region_X,axis=1)
        
    NX=profile_X.shape[0]
    NY=profile_Y.shape[0]

    X_=np.arange(NX)
    Y_=np.arange(NY)
    
    avX,sigX=weighted_avg_and_std(X_,profile_X**4) # take squared on purpose (weigh must be >0)
    avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)
    
    # redefine x0, star center
    x0=int(avX+x0-star_halfwidth)
       
     # subsample of the image (left part)
    reduc_image=full_image[:,x0:spec_index_max]/all_expo[index] 
    reduc_image[:,0:100]=0  # erase central star
    
   
     
      
    ## find the     
    yprofile=np.sum(reduc_image,axis=1)
    yy0=np.where(yprofile==yprofile.max())[0][0]    
        
    
    # wavelength calibration
    X_Size_Pixels=np.arange(0,reduc_image.shape[1])
    Y_Size_Pixels=np.arange(0,reduc_image.shape[0]) 
    # transverse size in pixel
    DY_Size_Pixels=Y_Size_Pixels-yy0
    NDY_C=int( float(DY_Size_Pixels.shape[0])/2.)
   
    #grating_name=all_filt[index].replace('dia ','')
    grating_name=get_disperser_filtname(all_filt[index])
    lambdas=Pixel_To_Lambdas(grating_name,X_Size_Pixels,all_pointing[index],False)
    
      

    
    all_Yprofile = []
    all_fwhm = []
    
    # loop on wavelength bands
    for band in np.arange(NBANDS):
        iband=band
        w1=wlmin[iband]
        w2=wlmax[iband]
        Xpixel_range=np.where(np.logical_and(lambdas>w1,lambdas<w2))[0]
        
        sub_image=np.copy(reduc_image[:,Xpixel_range])
        # transverse profile
        sub_yprofile=np.sum(sub_image,axis=1)
        sub_yprofile_background=np.median(sub_yprofile)
        sub_yprofile_clean=sub_yprofile-sub_yprofile_background
        
        mean,sig=weighted_avg_and_std(DY_Size_Pixels,np.abs(sub_yprofile_clean))
        
        # cut the tails not to bias the FWHM
        tmin=NDY_C-12
        tmax=NDY_C+12
        mean_2,sig_2=weighted_avg_and_std(DY_Size_Pixels[tmin:tmax],np.abs(sub_yprofile_clean[tmin:tmax]))
        
        
        all_Yprofile.append(sub_yprofile_clean)
        label="$\lambda$ = {:3.0f}-{:3.0f}nm, $fwhm=$ {:2.1f} pix".format(w1,w2,2.36*sig_2)
        all_fwhm.append(2.36*sig_2)
        plt.plot(DY_Size_Pixels,sub_yprofile_clean,'-',label=label,lw=2)
        plt.title("Transverse size for different wavelength")
        plt.xlabel("Y - pixel")
        
    plt.title(thetitle) 
    plt.grid(color='grey', ls='solid')
    plt.legend(loc=1)
    plt.xlim(-30.,30.)
    plt.savefig(figfilename)
    
    all_fwhm=np.array(all_fwhm)
    wl_average=np.average([wlmin,wlmax],axis=0)
    
    return wl_average,all_fwhm


#--------------------------------------------------------------------------------------------------------------
    
def ShowLongitudinalSpectraSelection(index,all_images,all_pointing,thex0,they0,all_titles,object_name,all_expo,dir_top_img,all_filt,figname):
    """
    ShowLongitudinalSpectraSelection::
        
        The goal is to compare the spectrum shape when varying the transverse selection width.
        Notice  background subtraction is performed
    
        input:
        - index: selected index
        - all_images : all set of cut and rotated images
        - all_pointing : list of reference to find hologram and grater parameter for calibration
        
        - thex0, they0 : list of where is the central star in the image
        - all_titles : list of title of the image
        - object_name : list of object name
        - all_expo : list of exposure time
        - dir_top_img : directory to save the image
        - all_filt : list of filter-disperser name
        - figname : filename of figure
        
        output: 
        - the image of longitudinal spectra for different transverse selection width
    
    
    """
    plt.figure(figsize=(15,6))
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    central_star_cut=100
    
    
    # different selection width
    #--------------------------
    wsel_set=np.array([1.,3.,5.,7.,10.])
    NBSEL=wsel_set.shape[0]
    

    figfilename=os.path.join(dir_top_img,figname)         
    thetitle=all_titles[index]+' '+all_filt[index]   
    
    #--------------
    #center is approximately the one on the original raw image (may be changed)  
    x0=int(thex0[index])
    
        
    # Extract the image    
    full_image=np.copy(all_images[index])
        
    # refine center in X,Y
    star_region_X=np.copy(full_image[:,x0-star_halfwidth:x0+star_halfwidth])
        
    profile_X=np.sum(star_region_X,axis=0)
    profile_Y=np.sum(star_region_X,axis=1)
        
    NX=profile_X.shape[0]
    NY=profile_Y.shape[0]

    X_=np.arange(NX)
    Y_=np.arange(NY)
    
    avX,sigX=weighted_avg_and_std(X_,profile_X**4) # take squared on purpose (weigh must be >0)
    avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)
    
    # redefine x0, star center
    x0=int(avX+x0-star_halfwidth)

    
    
    yprofile=np.sum(full_image[:,spec_index_min:spec_index_max],axis=1)
    y0=np.where(yprofile==yprofile.max())[0][0]

    reduc_image=full_image[y0-20:y0+20,x0:spec_index_max]/all_expo[index] 
    reduc_image[:,0:central_star_cut]=0  # erase central star
    
    X_Size_Pixels=np.arange(0,reduc_image.shape[1])
    Y_Size_Pixels=np.arange(0,reduc_image.shape[0])
    Transverse_Pixel_Size=Y_Size_Pixels-int(float(Y_Size_Pixels.shape[0])/2.)
    
    
    #grating_name=all_filt[index].replace('dia ','')
    grating_name=get_disperser_filtname(all_filt[index])
    lambdas=Pixel_To_Lambdas(grating_name,X_Size_Pixels,all_pointing[index],False)
    
   
  
    
    all_longitudinal_profile = []
    all_max = []
    for thewsel in wsel_set:
                
        y_indexsel=np.where(np.abs(Transverse_Pixel_Size)<=thewsel)[0]
                
        # extract longitudinal profile
        longitudinal_profile2d=np.copy(reduc_image[y_indexsel,:])
        longitudinal_profile1d=np.sum(longitudinal_profile2d,axis=0)
        
        #compute the background between 100-150 pixel
        magnitude_bkg=np.median(longitudinal_profile1d[central_star_cut:central_star_cut+50])
        longitudinal_profile1d_bkg=np.copy(longitudinal_profile1d)
        longitudinal_profile1d_bkg[:]=magnitude_bkg
        longitudinal_profile1d_bkg[0:central_star_cut]=0
        
        
        # longitudinal background with background subtraction
        longitudinal_profile1d_nobkg=longitudinal_profile1d-longitudinal_profile1d_bkg
        
        
        all_max.append(np.max(longitudinal_profile1d_nobkg))
        all_longitudinal_profile.append(longitudinal_profile1d_nobkg)
        
        thelabel=' abs(y) < {} '.format(thewsel)
        
        plt.plot(lambdas,longitudinal_profile1d_nobkg,'-',label=thelabel,lw=2)
        
    all_max=np.array(all_max)
    themax=np.max(all_max)
    
    YMIN=0.
    YMAX=1.2*themax
    
    for line in LINES:
        if line == O2 or line == HALPHA or line == HBETA or line == HGAMMA or line == HDELTA:
            plt.plot([line['lambda'],line['lambda']],[YMIN,YMAX],'-',color='red',lw=0.5)
            plt.text(line['lambda'],0.9*(YMAX-YMIN),line['label'],verticalalignment='bottom', horizontalalignment='center',color='red', fontweight='bold',fontsize=16)
    
    
    plt.title(thetitle)
   
    
    plt.grid(color='grey', ls='solid')
    #plt.text(100,-5.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='yellow', fontweight='bold',fontsize=16)
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('Intensity')
    
    plt.xlim(0.,1200.)
    plt.ylim(0.,YMAX)
    plt.legend(loc=1)
    plt.savefig(figfilename)
    
    
#--------------------------------------------------------------------------------------------------------------------------------------
def ShowOneAbsorptionLine(index,all_images,all_pointing,thex0,they0,all_titles,object_name,all_expo,dir_top_img,all_filt,figname):
        
    """
    ShowOneAbsorptionLine:
    ----------------------
    
    Shows the O2 absorption line as a quality test of the disperser.
    Notice  background subtraction is performed
    
        input:
        - index: selected index
        - all_images : all set of cut and rotated images
        - all_pointing : list of reference to find hologram and grater parameter for calibration
        
        - thex0, they0 : list of where is the central star in the image
        - all_titles : list of title of the image
        - object_name : list of object name
        - all_expo : list of exposure time
        - dir_top_img : directory to save the image
        - all_filt : list of filter-disperser name
        - figname : filename of figure
        
        output: 
        - the image of longitudinal spectra around O2 abs fine for different transverse width

    """
    
    # define O2 line
    O2WL1=740
    O2WL2=750
    O2WL3=782
    O2WL4=790
    
    #current analysis for O2
    wl1=O2WL1
    wl2=O2WL2
    wl3=O2WL3
    wl4=O2WL4
    
    plt.figure(figsize=(10,6))
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    central_star_cut=100 # erease central region
    
    #different transverse width
    wsel_set=np.array([1.,3.,5.,7.,10.])
    NBSEL=wsel_set.shape[0]
    
    
    figfilename=os.path.join(dir_top_img,figname)     
    thetitle=all_titles[index]+' '+all_filt[index]   
    #
    #--------------
    #center is approximately the one on the original raw image (may be changed)  
    x0=int(thex0[index])
    
        
    # Extract the image    
    full_image=np.copy(all_images[index])
        
    # refine center in X,Y
    star_region_X=np.copy(full_image[:,x0-star_halfwidth:x0+star_halfwidth])
        
    profile_X=np.sum(star_region_X,axis=0)
    profile_Y=np.sum(star_region_X,axis=1)
        
    NX=profile_X.shape[0]
    NY=profile_Y.shape[0]

    X_=np.arange(NX)
    Y_=np.arange(NY)
    
    avX,sigX=weighted_avg_and_std(X_,profile_X**4) # take squared on purpose (weigh must be >0)
    avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)
    
    # redefine x0, star center
    x0=int(avX+x0-star_halfwidth)

    
    
    yprofile=np.sum(full_image[:,spec_index_min:spec_index_max],axis=1)
    y0=np.where(yprofile==yprofile.max())[0][0]

    reduc_image=full_image[y0-20:y0+20,x0:spec_index_max]/all_expo[index] 
    reduc_image[:,0:central_star_cut]=0  # erase central star
    
    X_Size_Pixels=np.arange(0,reduc_image.shape[1])
    Y_Size_Pixels=np.arange(0,reduc_image.shape[0])
    Transverse_Pixel_Size=Y_Size_Pixels-int(float(Y_Size_Pixels.shape[0])/2.)
    
    
    
    # wavelength calibration
    #grating_name=all_filt[index].replace('dia ','')
    grating_name=get_disperser_filtname(all_filt[index])
    lambdas=Pixel_To_Lambdas(grating_name,X_Size_Pixels,all_pointing[index],False)
    
    
        
    
    
        # 1 container of full 1D Spectra
    all_longitudinal_profile = []
    # loop on transverse cut
    for thewsel in wsel_set:                
        y_indexsel=np.where(np.abs(Transverse_Pixel_Size)<=thewsel)[0]      
        longitudinal_profile2d=np.copy(reduc_image[y_indexsel,:])
        longitudinal_profile1d=np.sum(longitudinal_profile2d,axis=0)
        
        
        #compute the background between 100-150 pixel
        magnitude_bkg=np.median(longitudinal_profile1d[central_star_cut:central_star_cut+50])
        longitudinal_profile1d_bkg=np.copy(longitudinal_profile1d)
        longitudinal_profile1d_bkg[:]=magnitude_bkg
        longitudinal_profile1d_bkg[0:central_star_cut]=0
        
        
        # longitudinal background with background subtraction
        longitudinal_profile1d_nobkg=longitudinal_profile1d-longitudinal_profile1d_bkg
        
        
        
        all_longitudinal_profile.append(longitudinal_profile1d_nobkg)
        
        
        
    # 2 bins of the region around abs line    
    selected_indexes=np.where(np.logical_and(lambdas>=wl1,lambdas<=wl4))        
    wl_cut=lambdas[selected_indexes]
    
    # 3 continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(lambdas>=wl1,lambdas<=wl2),np.logical_and(lambdas>=wl3,lambdas<wl4)))
    wl_cont=lambdas[continuum_indexes]
    
    
    # 3 extract sub-spectrum
    all_absline_profile = []
    all_cont_profile = []
    fit_line_x=np.linspace(wl1,wl4,50)
    idx=0
    for thewsel in wsel_set: 
        full_spec=all_longitudinal_profile[idx]
        spec_cut=full_spec[selected_indexes]
        all_absline_profile.append(spec_cut)
        
        spec_cont=full_spec[continuum_indexes]
        z_cont_fit=np.polyfit(wl_cont, spec_cont,1)
        pol_cont_fit=np.poly1d(z_cont_fit)        
        fit_line_y=pol_cont_fit(fit_line_x)
        all_cont_profile.append(fit_line_y)        
        idx+=1
    # 4 plot
    idx=0
    for thewsel in wsel_set: 
        thelabel='abs(y) < {} '.format(thewsel)
        plt.plot(wl_cut,all_absline_profile[idx],'-',label=thelabel,lw=2)
        plt.plot(fit_line_x,all_cont_profile[idx],'k:')
        idx+=1
    
    
    
    plt.title(thetitle)
   
    
    plt.grid(color='grey', ls='solid')
    #plt.text(100,-5.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='yellow', fontweight='bold',fontsize=16)
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('Intensity')
    
    plt.xlim(wl1,wl4+20)
    #plt.ylim(0.,YMAX)
    plt.legend(loc=1)
    plt.savefig(figfilename)

#-------------------------------------------------------------------------------------------------------------------
    
    

def ShowOneEquivWidth(index,all_images,all_pointing,thex0,they0,all_titles,object_name,all_expo,dir_top_img,all_filt,figname):
    """
    
    ShowOneEquivWidth:
    -----------------
    
        Shows the O2 equivalent width as a quality test of the disperser
        Notice  background substraction done
        
        input:
        - index: selected index
        - all_images : all set of cut and rotated images
        - all_pointing : list of reference to find hologram and grater parameter for calibration
        
        - thex0, they0 : list of where is the central star in the image
        - all_titles : list of title of the image
        - object_name : list of object name
        - all_expo : list of exposure time
        - dir_top_img : directory to save the image
        - all_filt : list of filter-disperser name
        - figname : filename of figure
        
        output: 
        - the plot of equivalent width around O2 abs fine for different transverse widt
    
    """
    
    O2WL1=740
    O2WL2=750
    O2WL3=782
    O2WL4=790
    
    wl1=O2WL1
    wl2=O2WL2
    wl3=O2WL3
    wl4=O2WL4
    
    plt.figure(figsize=(10,6))
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    central_star_cut=100
    
    # transverse width selection
    wsel_set=np.array([1.,3.,5.,7.,10.])
    NBSEL=wsel_set.shape[0]
     
    figfilename=os.path.join(dir_top_img,figname)       
    thetitle=all_titles[index]+' '+all_filt[index]   
    
    
  
    #--------------
    #center is approximately the one on the original raw image (may be changed)  
    x0=int(thex0[index])
    
        
    # Extract the image    
    full_image=np.copy(all_images[index])
        
    # refine center in X,Y
    star_region_X=np.copy(full_image[:,x0-star_halfwidth:x0+star_halfwidth])
        
    profile_X=np.sum(star_region_X,axis=0)
    profile_Y=np.sum(star_region_X,axis=1)
        
    NX=profile_X.shape[0]
    NY=profile_Y.shape[0]

    X_=np.arange(NX)
    Y_=np.arange(NY)
    
    avX,sigX=weighted_avg_and_std(X_,profile_X**4) # take squared on purpose (weigh must be >0)
    avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)
    
    # redefine x0, star center
    x0=int(avX+x0-star_halfwidth)

    
    yprofile=np.sum(full_image[:,spec_index_min:spec_index_max],axis=1)
    y0=np.where(yprofile==yprofile.max())[0][0]

    reduc_image=full_image[y0-20:y0+20,x0:spec_index_max]/all_expo[index] 
    reduc_image[:,0:central_star_cut]=0  # erase central star
    
    X_Size_Pixels=np.arange(0,reduc_image.shape[1])
    Y_Size_Pixels=np.arange(0,reduc_image.shape[0])
    Transverse_Pixel_Size=Y_Size_Pixels-int(float(Y_Size_Pixels.shape[0])/2.)
    
    
    # wavelength calibration
    #grating_name=all_filt[index].replace('dia ','')
    grating_name=get_disperser_filtname(all_filt[index])
    lambdas=Pixel_To_Lambdas(grating_name,X_Size_Pixels,all_pointing[index],False)
    
  
    
    # 1 container of full 1D Spectra
    all_longitudinal_profile = []
    for thewsel in wsel_set:                
        y_indexsel=np.where(np.abs(Transverse_Pixel_Size)<=thewsel)[0]      
        longitudinal_profile2d=np.copy(reduc_image[y_indexsel,:])
        longitudinal_profile1d=np.sum(longitudinal_profile2d,axis=0)
        
        
        #compute the background between 100-150 pixel
        magnitude_bkg=np.median(longitudinal_profile1d[central_star_cut:central_star_cut+50])
        longitudinal_profile1d_bkg=np.copy(longitudinal_profile1d)
        longitudinal_profile1d_bkg[:]=magnitude_bkg
        longitudinal_profile1d_bkg[0:central_star_cut]=0
        
        # do bkg substraction
        longitudinal_profile1d_nobkg=longitudinal_profile1d-longitudinal_profile1d_bkg
        
        all_longitudinal_profile.append(longitudinal_profile1d_nobkg)
        
        
    # 2 bins of the region around abs line    
    selected_indexes=np.where(np.logical_and(lambdas>=wl1,lambdas<=wl4))        
    wl_cut=lambdas[selected_indexes]
    
    # 3 continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(lambdas>=wl1,lambdas<=wl2),np.logical_and(lambdas>=wl3,lambdas<wl4)))
    wl_cont=lambdas[continuum_indexes]
    
    
    # 3 extract sub-spectrum
    all_absline_profile = []
    all_cont_profile = []
    all_ratio = []
    fit_line_x=np.linspace(wl1,wl4,50)
    idx=0
    for thewsel in wsel_set: 
        full_spec=all_longitudinal_profile[idx]
        spec_cut=full_spec[selected_indexes]
        all_absline_profile.append(spec_cut)
        
        spec_cont=full_spec[continuum_indexes]
        z_cont_fit=np.polyfit(wl_cont, spec_cont,1)
        pol_cont_fit=np.poly1d(z_cont_fit)        
        fit_line_y=pol_cont_fit(fit_line_x)
        full_continum=pol_cont_fit(wl_cut) 
        ratio=spec_cut/full_continum
        all_cont_profile.append(fit_line_y)
        all_ratio.append(ratio)        
        idx+=1
    # 4 plot
    idx=0
    for thewsel in wsel_set:
        thelabel='abs(y) < {} '.format(thewsel)
        plt.plot(wl_cut,all_ratio[idx],'-',label=thelabel,lw=2)
        idx+=1

    
    plt.title(thetitle)
   
    
    plt.grid(color='grey', ls='solid')
    #plt.text(100,-5.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='yellow', fontweight='bold',fontsize=16)
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('Equivalent Width')
    
    plt.xlim(wl1,wl4+20)
    #plt.ylim(0.,YMAX)
    plt.legend(loc=1)
    plt.savefig(figfilename)
    
    
#---------------------------------------------------------------------------------------------


def ComputeEquivalentWidthOld(wl,spec,wl1,wl2,wl3,wl4):
    """
    ComputeEquivalentWidth : compute the equivalent width must be computed
    
    input:
        wl : array of wavelength
        spec: array of wavelength
        
        wl1,wl2,wl3,wl4 : range of wavelength
    
    
    
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
     
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,1)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
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


#-----------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
def CalculateOneAbsorptionLine(index,all_images,all_pointing,thex0,they0,all_titles,object_name,all_expo,dir_top_img,all_filt,figname):
        
    """
    CalculateOneAbsorptionLine:
    ----------------------
    
    Shows the O2 absorption line as a quality test of the disperser.
    Notice  background subtraction is performed
    
        input:
        - index: selected index
        - all_images : all set of cut and rotated images
        - all_pointing : list of reference to find hologram and grater parameter for calibration
        
        - thex0, they0 : list of where is the central star in the image
        - all_titles : list of title of the image
        - object_name : list of object name
        - all_expo : list of exposure time
        - dir_top_img : directory to save the image
        - all_filt : list of filter-disperser name
        - figname : filename of figure
        
        output: 
        - the image of longitudinal spectra around O2 abs fine for different transverse width

    """
    
    # define O2 line
    O2WL1=740
    O2WL2=750
    O2WL3=782
    O2WL4=790
    
    #current analysis for O2
    wl1=O2WL1
    wl2=O2WL2
    wl3=O2WL3
    wl4=O2WL4
    
    plt.figure(figsize=(12,6))
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    central_star_cut=100
    
    #different transverse width
    wsel_set=np.array([1.,3.,5.,7.,10.])
    NBSEL=wsel_set.shape[0]
    
    
    figfilename=os.path.join(dir_top_img,figname)     
    thetitle=all_titles[index]+' '+all_filt[index]   
    #
    #--------------
    #center is approximately the one on the original raw image (may be changed)  
    x0=int(thex0[index])
    
        
    # Extract the image    
    full_image=np.copy(all_images[index])
        
    # refine center in X,Y
    star_region_X=np.copy(full_image[:,x0-star_halfwidth:x0+star_halfwidth])
        
    profile_X=np.sum(star_region_X,axis=0)
    profile_Y=np.sum(star_region_X,axis=1)
        
    NX=profile_X.shape[0]
    NY=profile_Y.shape[0]

    X_=np.arange(NX)
    Y_=np.arange(NY)
    
    avX,sigX=weighted_avg_and_std(X_,profile_X**4) # take squared on purpose (weigh must be >0)
    avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)
    
    # redefine x0, star center
    x0=int(avX+x0-star_halfwidth)

    
    
    yprofile=np.sum(full_image[:,spec_index_min:spec_index_max],axis=1)
    y0=np.where(yprofile==yprofile.max())[0][0]

    reduc_image=full_image[y0-20:y0+20,x0:spec_index_max]/all_expo[index] 
    reduc_image[:,0:central_star_cut]=0  # erase central star
    
    X_Size_Pixels=np.arange(0,reduc_image.shape[1])
    Y_Size_Pixels=np.arange(0,reduc_image.shape[0])
    Transverse_Pixel_Size=Y_Size_Pixels-int(float(Y_Size_Pixels.shape[0])/2.)
    
    
    
    # wavelength calibration
    #grating_name=all_filt[index].replace('dia ','')
    grating_name=get_disperser_filtname(all_filt[index])
    lambdas=Pixel_To_Lambdas(grating_name,X_Size_Pixels,all_pointing[index],False)
     
    
    
    
    
        # 1 container of full 1D Spectra
    all_longitudinal_profile = []
    all_eqw=[]
    for thewsel in wsel_set:                
        y_indexsel=np.where(np.abs(Transverse_Pixel_Size)<=thewsel)[0]      
        longitudinal_profile2d=np.copy(reduc_image[y_indexsel,:])
        longitudinal_profile1d=np.sum(longitudinal_profile2d,axis=0)
        
        
        #compute the background between 100-150 pixel
        magnitude_bkg=np.median(longitudinal_profile1d[central_star_cut:central_star_cut+50])
        longitudinal_profile1d_bkg=np.copy(longitudinal_profile1d)
        longitudinal_profile1d_bkg[:]=magnitude_bkg
        longitudinal_profile1d_bkg[0:central_star_cut]=0
        
        # bkg subtraction
        longitudinal_profile1d_nobkg=longitudinal_profile1d-longitudinal_profile1d_bkg
        eqw=ComputeEquivalentWidth(lambdas,longitudinal_profile1d_nobkg,wl1,wl2,wl3,wl4)
        all_eqw.append(eqw)
        all_longitudinal_profile.append(longitudinal_profile1d_nobkg)
        
        
        
    # 2 bins of the region around abs line    
    selected_indexes=np.where(np.logical_and(lambdas>=wl1,lambdas<=wl4))        
    wl_cut=lambdas[selected_indexes]
    
    # 3 continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(lambdas>=wl1,lambdas<=wl2),np.logical_and(lambdas>=wl3,lambdas<wl4)))
    wl_cont=lambdas[continuum_indexes]
    
    
    # 3 extract sub-spectrum
    all_absline_profile = []
    all_cont_profile = []
    fit_line_x=np.linspace(wl1,wl4,50)
    idx=0
    for thewsel in wsel_set: 
        full_spec=all_longitudinal_profile[idx]
        spec_cut=full_spec[selected_indexes]
        all_absline_profile.append(spec_cut)
        
        spec_cont=full_spec[continuum_indexes]
        z_cont_fit=np.polyfit(wl_cont, spec_cont,1)
        pol_cont_fit=np.poly1d(z_cont_fit)        
        fit_line_y=pol_cont_fit(fit_line_x)
        all_cont_profile.append(fit_line_y)        
        idx+=1
    # 4 plot
    idx=0
    for thewsel in wsel_set: 
        thelabel='abs(y) < {} ; EQW={:2.2f} nm '.format(thewsel,all_eqw[idx])
        plt.plot(wl_cut,all_absline_profile[idx],'-',label=thelabel,lw=2)
        plt.plot(fit_line_x,all_cont_profile[idx],'k:')
        idx+=1
    
    
    
    plt.title(thetitle)
   
    
    plt.grid(color='grey', ls='solid')
    #plt.text(100,-5.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='yellow', fontweight='bold',fontsize=16)
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('Intensity')
    
    plt.xlim(wl1,wl4+40)
    #plt.ylim(0.,YMAX)
    plt.legend(loc=1)
    plt.savefig(figfilename)

#-------------------------------------------------------------------------------------------------------------------


def CalculateOneEquivWidth(index,all_images,all_pointing,thex0,they0,all_titles,object_name,all_expo,dir_top_img,all_filt,figname):
    """
    
    ShowOneEquivWidth:
    -----------------
    
        Shows the O2 equivalent width as a quality test of the disperser
        Notice  background substraction done
        
        input:
        - index: selected index
        - all_images : all set of cut and rotated images
        - all_pointing : list of reference to find hologram and grater parameter for calibration
        
        - thex0, they0 : list of where is the central star in the image
        - all_titles : list of title of the image
        - object_name : list of object name
        - all_expo : list of exposure time
        - dir_top_img : directory to save the image
        - all_filt : list of filter-disperser name
        - figname : filename of figure
        
        output: 
        - the plot of equivalent width around O2 abs fine for different transverse widt
    
    """
    
    O2WL1=740
    O2WL2=750
    O2WL3=782
    O2WL4=790
    
    wl1=O2WL1
    wl2=O2WL2
    wl3=O2WL3
    wl4=O2WL4
    
    plt.figure(figsize=(12,6))
    spec_index_min=100  # cut the left border
    spec_index_max=1900 # cut the right border
    star_halfwidth=70
    central_star_cut=100
    
    # transverse width selection
    wsel_set=np.array([1.,3.,5.,7.,10.])
    NBSEL=wsel_set.shape[0]
     
    figfilename=os.path.join(dir_top_img,figname)       
    thetitle=all_titles[index]+' '+all_filt[index]   
    
    
  
    #--------------
    #center is approximately the one on the original raw image (may be changed)  
    x0=int(thex0[index])
    
        
    # Extract the image    
    full_image=np.copy(all_images[index])
        
    # refine center in X,Y
    star_region_X=np.copy(full_image[:,x0-star_halfwidth:x0+star_halfwidth])
        
    profile_X=np.sum(star_region_X,axis=0)
    profile_Y=np.sum(star_region_X,axis=1)
        
    NX=profile_X.shape[0]
    NY=profile_Y.shape[0]

    X_=np.arange(NX)
    Y_=np.arange(NY)
    
    avX,sigX=weighted_avg_and_std(X_,profile_X**4) # take squared on purpose (weigh must be >0)
    avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)
    
    # redefine x0, star center
    x0=int(avX+x0-star_halfwidth)

    
    yprofile=np.sum(full_image[:,spec_index_min:spec_index_max],axis=1)
    y0=np.where(yprofile==yprofile.max())[0][0]

    reduc_image=full_image[y0-20:y0+20,x0:spec_index_max]/all_expo[index] 
    reduc_image[:,0:central_star_cut]=0  # erase central star
    
    X_Size_Pixels=np.arange(0,reduc_image.shape[1])
    Y_Size_Pixels=np.arange(0,reduc_image.shape[0])
    Transverse_Pixel_Size=Y_Size_Pixels-int(float(Y_Size_Pixels.shape[0])/2.)
    
    
    # wavelength calibration
    #grating_name=all_filt[index].replace('dia ','')
    grating_name=get_disperser_filtname(all_filt[index])
    lambdas=Pixel_To_Lambdas(grating_name,X_Size_Pixels,all_pointing[index],False)
    

    
    # 1 container of full 1D Spectra
    all_longitudinal_profile = []
    all_eqw=[]
    for thewsel in wsel_set:                
        y_indexsel=np.where(np.abs(Transverse_Pixel_Size)<=thewsel)[0]      
        longitudinal_profile2d=np.copy(reduc_image[y_indexsel,:])
        longitudinal_profile1d=np.sum(longitudinal_profile2d,axis=0)
        
        #compute the background between 100-150 pixel
        magnitude_bkg=np.median(longitudinal_profile1d[central_star_cut:central_star_cut+50])
        longitudinal_profile1d_bkg=np.copy(longitudinal_profile1d)
        longitudinal_profile1d_bkg[:]=magnitude_bkg
        longitudinal_profile1d_bkg[0:central_star_cut]=0
        
        # bkg subtraction
        longitudinal_profile1d_nobkg=longitudinal_profile1d-longitudinal_profile1d_bkg
        
        #eqw calculation
        eqw=ComputeEquivalentWidth(lambdas,longitudinal_profile1d_nobkg,wl1,wl2,wl3,wl4)
        all_eqw.append(eqw)
        all_longitudinal_profile.append(longitudinal_profile1d_nobkg)
        
        
    # 2 bins of the region around abs line    
    selected_indexes=np.where(np.logical_and(lambdas>=wl1,lambdas<=wl4))        
    wl_cut=lambdas[selected_indexes]
    
    # 3 continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(lambdas>=wl1,lambdas<=wl2),np.logical_and(lambdas>=wl3,lambdas<wl4)))
    wl_cont=lambdas[continuum_indexes]
    
    
    # 3 extract sub-spectrum
    all_absline_profile = []
    all_cont_profile = []
    all_ratio = []
    fit_line_x=np.linspace(wl1,wl4,50)
    idx=0
    for thewsel in wsel_set: 
        full_spec=all_longitudinal_profile[idx]
        spec_cut=full_spec[selected_indexes]
        all_absline_profile.append(spec_cut)
        
        spec_cont=full_spec[continuum_indexes]
        z_cont_fit=np.polyfit(wl_cont, spec_cont,1)
        pol_cont_fit=np.poly1d(z_cont_fit)        
        fit_line_y=pol_cont_fit(fit_line_x)
        full_continum=pol_cont_fit(wl_cut) 
        ratio=spec_cut/full_continum
        all_cont_profile.append(fit_line_y)
        all_ratio.append(ratio)        
        idx+=1
    # 4 plot
    idx=0
    for thewsel in wsel_set:
        #thelabel='abs(y) < {} '.format(thewsel)
        thelabel='abs(y) < {} ; EQW={:2.2f} nm '.format(thewsel,all_eqw[idx])
        plt.plot(wl_cut,all_ratio[idx],'-',label=thelabel,lw=2)
        idx+=1

    
    plt.title(thetitle)
   
    
    plt.grid(color='grey', ls='solid')
    #plt.text(100,-5.,all_filt[index],verticalalignment='bottom', horizontalalignment='center',color='yellow', fontweight='bold',fontsize=16)
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('Equivalent Width')
    
    plt.xlim(wl1,wl4+40)
    #plt.ylim(0.,YMAX)
    plt.legend(loc=1)
    plt.savefig(figfilename)
    
    
#---------------------------------------------------------------------------------------------



def ShowExtrSpectrainPDF(all_spectra,all_totspectra,all_titles,object_name,dir_top_img,all_filt,date,figname,NBIMGPERROW=2):
    """
    ShowExtrSpectrainPDF: Show the raw images without background subtraction
    ==============
    """
    
    NBSPEC=len(all_spectra)
    
    MAXIMGROW=max(2,int(m.ceil(float(NBSPEC)/float(NBIMGPERROW))))
    
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    titlepage='Raw 1D Spectra 1D for {}, date : {}'.format(object_name,date)
    
    
    
    # loop on spectra  
    for index in np.arange(0,NBSPEC):
        
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
    
  
        spectrum=all_spectra[index]
        totspectrum=all_totspectra[index]
        axarr[iy,ix].plot(spectrum,'r-',lw=2)
        axarr[iy,ix].plot(totspectrum,'k:',lw=2)
        
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        axarr[iy,ix].grid(True)
      
        max_y_to_plot=spectrum[:].max()*1.2
        
        #YMIN=0.
        #YMAX=max_y_to_plot
    
        #for line in LINES:
        #    if line == O2 or line == HALPHA or line == HBETA or line == HGAMMA or line == HDELTA:
        #        axarr[iy,ix].plot([line['lambda'],line['lambda']],[YMIN,YMAX],'-',color='red',lw=0.5)
        #        axarr[iy,ix].text(line['lambda'],0.9*(YMAX-YMIN),line['label'],verticalalignment='bottom', horizontalalignment='center',color='red', fontweight='bold',fontsize=16)
     
        
        axarr[iy,ix].set_xlabel("pixel")
        axarr[iy,ix].set_ylabel("ADU")
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        axarr[iy,ix].text(0.,max_y_to_plot*1.1/1.2, all_filt[index],verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
    
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()    
#---------------------------------------------------------------------------------------------------------    

def CALSPECAbsLineIdentificationinPDF(spectra,pointing,all_titles,object_name,dir_top_images,all_filt,date,figname,tagname,NBIMGPERROW=2):
    """
    CALSPECAbsLineIdentification show the right part of spectrum with identified lines
    =====================
    """
    
    
    NBSPEC=len(spectra)
    
    MAXIMGROW=max(2,int(m.ceil(float(NBSPEC)/float(NBIMGPERROW))))
    
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_images,figname)
   
    pp = PdfPages(figfilename) # create a pdf file
    
    
    titlepage='WL calibrated 1D Spectra 1D for obj : {} date :{}'.format(object_name,date)
    
    
    all_wl= []  # containers for wavelength
    
    
    for index in np.arange(0,NBSPEC):
        
             
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
             
        
        spec = spectra[index]
        
        # calibrate
        grating_name=get_disperser_filtname(all_filt[index])
        X_Size_Pixels=np.arange(spec.shape[0])
        lambdas = Pixel_To_Lambdas(grating_name,X_Size_Pixels,pointing[index],False)
        
        
        all_wl.append(lambdas)
        
        #plot
        axarr[iy,ix].plot(lambdas,spec,'r-',lw=2,label=tagname)
    
        thetitle="{} : {} : {} ".format(index,all_titles[index],all_filt[index])
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        
        #axarr[iy,ix].text(600.,spec.max()*1.1, all_filt[index],verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
        axarr[iy,ix].legend(loc='best',fontsize=16)
        axarr[iy,ix].set_xlabel('Wavelength [nm]', fontsize=16)
        axarr[iy,ix].grid(True)
        
        YMIN=0.
        YMAX=spec.max()*1.2
    
        for line in LINES:
            if line == O2 or line == HALPHA or line == HBETA or line == HGAMMA or line == HDELTA or line ==O2B or line == O2Y or line == O2Z:
                axarr[iy,ix].plot([line['lambda'],line['lambda']],[YMIN,YMAX],'-',color='red',lw=0.5)
                axarr[iy,ix].text(line['lambda'],0.9*(YMAX-YMIN),line['label'],verticalalignment='bottom', horizontalalignment='center',color='red', fontweight='bold',fontsize=16)
     
        
        axarr[iy,ix].set_ylim(YMIN,YMAX)
        axarr[iy,ix].set_xlim(np.min(lambdas),np.max(lambdas))
        axarr[iy,ix].set_xlim(0,1200.)
    
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()   
   
    return all_wl

#---------------------------------------------------------------------------------------------
def CompareSpectrumProfile(wl,spectra,all_titles,all_airmass,object_name,all_filt,dir_top_img,grating_name,list_of_index):
    """
    CompareSpectrumProfile
    =====================
    input:
        wl
        spectra
        all_titles
        object_name
        all_filt
        dir_top_img
        grating_name
        list_of_index
    
    
    output
    
    """
    shortfilename='CompareSpec_'+grating_name+'.pdf'
    title="Compare spectra of {} with disperser {}".format(object_name,grating_name)
    figfilename=os.path.join(dir_top_img,shortfilename)
    pp = PdfPages(figfilename) # create a pdf file
    
    
    f, axarr = plt.subplots(1,1,figsize=(10,6))
    f.suptitle(title,fontsize=16,fontweight='bold')
    
    NBSPEC=len(spectra)
    
    min_z=min(all_airmass)
    max_z=max(all_airmass)
    
    maxim_y_to_plot= []

    texte='airmass : {} - {} '.format(min_z,max_z)
    
    for index in np.arange(0,NBSPEC):
                
        if index in list_of_index:
            axarr.plot(wl[index],spectra[index],'-',lw=3)
            maxim_y_to_plot.append(spectra[index].max())
    
    max_y_to_plot=max(maxim_y_to_plot)*1.2
    axarr.set_ylim(0,max_y_to_plot)
    axarr.text(0.,max_y_to_plot*0.9, texte ,verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
    axarr.grid(True)
    
    YMIN=0.
    YMAX=max_y_to_plot
    
    for line in LINES:
        if line == O2 or line == HALPHA or line == HBETA or line == HGAMMA or line == HDELTA or line ==O2B or line == O2Y or line == O2Z:
            axarr.plot([line['lambda'],line['lambda']],[YMIN,YMAX],'-',color='red',lw=0.5)
            axarr.text(line['lambda'],0.9*(YMAX-YMIN),line['label'],verticalalignment='bottom', horizontalalignment='center',color='red', fontweight='bold',fontsize=16)
     
    
    #axarr.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    #axarr.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    axarr.grid(b=True, which='major', color='grey', linewidth=0.5)
    #axarr.grid(b=True, which='minor', color='grey', linewidth=0.5)

    
    axarr.set_ylabel("ADU",fontsize=10,fontweight='bold')
    axarr.set_xlabel("wavelength (nm)",fontsize=10,fontweight='bold')
    axarr.set_xlim(0.,1200.)
        
    f.savefig(pp, format='pdf')
    f.show()
    
    pp.close()     
    


# Study Calibrated Spectra
#---------------------------------------------------------------------------------------------
def BuildCalibSpec(sorted_filenames,sorted_numbers,object_name):
    """
    BuildRawSpec
    ===============
    """

    all_dates = []
    all_airmass = []
    
    all_leftspectra_data = []
    all_rightspectra_data = []
    
    all_leftspectra_data_stat_err = []
    all_rightspectra_data_stat_err = []
    
    all_leftspectra_wl = []
    all_rightspectra_wl = []
    all_titles = []
    all_header = []
    all_expo = []
    all_filt = []
    all_filt1= []
    all_filt2= []
   
    NBFILES=sorted_filenames.shape[0]

    for idx in range(NBFILES):  
        
        file=sorted_filenames[idx]    
        
        hdu_list=fits.open(file)
        
        #hdu_list.info()
        
        header=hdu_list[0].header
        #print header
        date_obs = header['DATE-OBS']
        airmass = header['AIRMASS']
        expo = header['EXPTIME']
        num=sorted_numbers[idx]
        title=object_name+" z= {:3.2f} Nb={}".format(float(airmass),num)
        filters = header['FILTERS']
        filters1= header['FILTER1']
        filters2= header['FILTER2']
        
        # now reads the spectra
        
        table_data=hdu_list[1].data
        
        #cols = hdu_list[1].columns
        #cols.info()
        #print hdu_list[1].columns
        #cols.names
  
        #col1=fits.Column(name='CalibLeftSpecWL',format='E',array=theleftwl_cut[idx[0]])
        #col2=fits.Column(name='CalibLeftSpecData',format='E',array=theleftspectrum_cut[idx[0]])
        #col3=fits.Column(name='CalibLeftSpecSim',format='E',array=theleftsimspec_cut[idx[0]])
        #col4=fits.Column(name='CalibRightSpecWL',format='E',array=therightwl_cut[idx[0]])
        #col5=fits.Column(name='CalibRightSpecData',format='E',array=therightspectrum_cut[idx[0]])
        #col6=fits.Column(name='CalibRightSpecSim',format='E',array=therightsimspec_cut[idx[0]])
    
    
        left_spectrum_wl=table_data.field('CalibLeftSpecWL')
        left_spectrum_data=table_data.field('CalibLeftSpec')
        left_spectrum_data_stat_err=table_data.field('CalibStatErrorLeftSpec')
        
        right_spectrum_wl=table_data.field('CalibRightSpecWL')
        right_spectrum_data=table_data.field('CalibRightSpec')
        right_spectrum_data_stat_err=table_data.field('CalibStatErrorRightSpec')
       
        
 
        
        
        all_dates.append(date_obs)
        all_airmass.append(float(airmass))
        
        all_leftspectra_data.append(left_spectrum_data)
        all_rightspectra_data.append(right_spectrum_data)
        
        all_leftspectra_wl.append(left_spectrum_wl)
        all_rightspectra_wl.append(right_spectrum_wl)
        
        all_leftspectra_data_stat_err.append(left_spectrum_data_stat_err)
        all_rightspectra_data_stat_err.append(right_spectrum_data_stat_err)
        
        all_titles.append(title)
        all_header.append(header)
        all_expo.append(expo)
        
        all_filt.append(filters)
        all_filt1.append(filters1)
        all_filt2.append(filters2)
            
        hdu_list.close()
        
    return all_dates,all_airmass,all_titles,all_header,all_expo, all_leftspectra_data,all_rightspectra_data, all_leftspectra_data_stat_err , all_rightspectra_data_stat_err ,all_leftspectra_wl,all_rightspectra_wl,all_filt,all_filt1,all_filt2
#-----------------------------------------------------------------------------------------------------


def BuildCalibandSimSpecFull(sorted_filenames,sorted_numbers):
    """
    BuildCalibSpecFull : Get everything from the fits file
    ===============
    
      called by AnaCompareDataSimSpec.ipynb
    """


    
    all_leftspectra_data = []
    all_rightspectra_data = []
    
    all_totleftspectra_data = []
    all_totrightspectra_data = []
    
    all_leftspectra_data_stat_err = []
    all_rightspectra_data_stat_err = []
    
    all_sim_spectra_wl = []
    all_sim_spectra_data = []
    
    
    all_leftspectra_wl = []
    all_rightspectra_wl = []
    all_titles = []
  
    
    all_header = []
    all_objects = []
    all_dates = []
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
    all_filt = []
    all_filt1 = []
    all_filt2 = []
    
    
    
   
    NBFILES=sorted_filenames.shape[0]

    for idx in range(NBFILES):  
        
        file=sorted_filenames[idx]    
        
        hdu_list=fits.open(file)
        
        #hdu_list.info()
        
        header=hdu_list[0].header
        #print header
        
        date_obs = header['DATE-OBS']
        airmass = float(header['AIRMASS'])
        expo= float(header['EXPTIME'])
      
        obj=header['OBJECT']
   
        ut=header['UT']
        ra=header['RA']
        dec=header['DEC']
        epoch=float(header['EPOCH'])
        zd = float(header['ZD'])
        ha = header['HA']
        st = header['ST']
        alt = float(header['ALT'])
        fcl = float(header['TELFOCUS'])
        temp= float(header['OUTTEMP'])
        press= float(header['OUTPRESS'])
        hum= float(header['OUTHUM'])
        windsp=float(header['WNDSPEED'])
        seeing=float(header['SEEING'])
        seeingam=float(header['SAIRMASS'])
 
        
    
        num=sorted_numbers[idx]
        title=obj+" z= {:3.2f} Nb={}".format(float(airmass),num)
        filters = header['FILTERS']
        filters1= header['FILTER1']
        filters2= header['FILTER2']
        
        # now reads the spectra
        
        table_data=hdu_list[1].data
        
        #cols = hdu_list[1].columns
        #cols.info()
        #print hdu_list[1].columns
        #cols.names
  
    
        left_spectrum_wl=table_data.field('CalibLeftSpecWL')
        left_spectrum_data=table_data.field('CalibLeftSpec')
        left_spectrum_data_stat_err=table_data.field('CalibStatErrorLeftSpec')
        totleft_spectrum_data=table_data.field('CalibTotLeftSpec')
        
        
        right_spectrum_wl=table_data.field('CalibRightSpecWL')
        right_spectrum_data=table_data.field('CalibRightSpec')
        right_spectrum_data_stat_err=table_data.field('CalibStatErrorRightSpec')
        totright_spectrum_data=table_data.field('CalibTotRightSpec')
        
        sim_spectrum_wl=table_data.field('SimSpecWL')
        sim_spectrum_data=table_data.field('SimSpec')




        
        all_dates.append(date_obs)
        all_objects.append(obj)
        all_airmass.append(airmass)
        
        all_header.append(header)
        
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
        all_hum.append(hum)
        all_windsp.append(windsp)
        all_seeing.append(seeing)
        all_seeingam.append(seeingam)
      
        

        
        all_leftspectra_data.append(left_spectrum_data)
        all_rightspectra_data.append(right_spectrum_data)
        
        all_leftspectra_wl.append(left_spectrum_wl)
        all_rightspectra_wl.append(right_spectrum_wl)
        
        all_leftspectra_data_stat_err.append(left_spectrum_data_stat_err)
        all_rightspectra_data_stat_err.append(right_spectrum_data_stat_err)
        
        all_totleftspectra_data.append(totleft_spectrum_data)
        all_totrightspectra_data.append(totright_spectrum_data)
        
        
        all_sim_spectra_wl.append(sim_spectrum_wl)
        all_sim_spectra_data.append(sim_spectrum_data)
        
        all_titles.append(title)
       
        
        
        all_filt.append(filters)
        all_filt1.append(filters1)
        all_filt2.append(filters2)
            
        hdu_list.close()
        
    return all_header, \
        all_dates, \
        all_objects, \
        all_airmass, \
        all_titles, \
        all_exposures, \
        all_ut, all_ra,all_dec,all_epoch,all_zenith,all_ha,all_st,all_alt,all_focus,\
        all_temp, all_press,all_hum,all_windsp,\
        all_seeing,all_seeingam,\
        all_filt,all_filt1,all_filt2,\
        all_leftspectra_data, \
        all_rightspectra_data, \
        all_leftspectra_data_stat_err ,\
        all_rightspectra_data_stat_err ,\
        all_leftspectra_wl,\
        all_rightspectra_wl, \
        all_totleftspectra_data, \
        all_totrightspectra_data, \
        all_sim_spectra_wl, \
        all_sim_spectra_data

#--------------------------------------------------------------------------------------------------------------------------
def BuildCalibSpecFull(sorted_filenames,sorted_numbers):
    """
    BuildCalibandSpecFull : Get everything from the fits file
    =========================================================
    
    called by SimulateSpectrum.ipynb 
    
    """


    
    all_leftspectra_data = []
    all_rightspectra_data = []
    
    all_totleftspectra_data = []
    all_totrightspectra_data = []
    
    all_leftspectra_data_stat_err = []
    all_rightspectra_data_stat_err = []
    
    all_leftspectra_wl = []
    all_rightspectra_wl = []
    all_titles = []
  
    
    all_header = []
    all_objects = []
    all_dates = []
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
    all_filt = []
    all_filt1 = []
    all_filt2 = []
    
    
    
   
    NBFILES=sorted_filenames.shape[0]

    for idx in range(NBFILES):  
        
        file=sorted_filenames[idx]    
        
        hdu_list=fits.open(file)
        
        #hdu_list.info()
        
        header=hdu_list[0].header
        #print header
        
        date_obs = header['DATE-OBS']
        airmass = float(header['AIRMASS'])
        expo= float(header['EXPTIME'])
      
        obj=header['OBJECT']
   
        ut=header['UT']
        ra=header['RA']
        dec=header['DEC']
        epoch=float(header['EPOCH'])
        zd = float(header['ZD'])
        ha = header['HA']
        st = header['ST']
        alt = float(header['ALT'])
        fcl = float(header['TELFOCUS'])
        temp= float(header['OUTTEMP'])
        press= float(header['OUTPRESS'])
        hum= float(header['OUTHUM'])
        windsp=float(header['WNDSPEED'])
        seeing=float(header['SEEING'])
        seeingam=float(header['SAIRMASS'])
 
        
    
        num=sorted_numbers[idx]
        title=obj+" z= {:3.2f} Nb={}".format(float(airmass),num)
        filters = header['FILTERS']
        filters1= header['FILTER1']
        filters2= header['FILTER2']
        
        # now reads the spectra
        
        table_data=hdu_list[1].data
        
        #cols = hdu_list[1].columns
        #cols.info()
        #print hdu_list[1].columns
        #cols.names
  
    
        left_spectrum_wl=table_data.field('CalibLeftSpecWL')
        left_spectrum_data=table_data.field('CalibLeftSpec')
        left_spectrum_data_stat_err=table_data.field('CalibStatErrorLeftSpec')
        totleft_spectrum_data=table_data.field('CalibTotLeftSpec')
        
        
        right_spectrum_wl=table_data.field('CalibRightSpecWL')
        right_spectrum_data=table_data.field('CalibRightSpec')
        right_spectrum_data_stat_err=table_data.field('CalibStatErrorRightSpec')
        totright_spectrum_data=table_data.field('CalibTotRightSpec')
        

        # append only once !!!!!!
        
        all_dates.append(date_obs)
        all_objects.append(obj)
        all_airmass.append(airmass)
        
        all_header.append(header)
        
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
        all_hum.append(hum)
        all_windsp.append(windsp)
        all_seeing.append(seeing)
        all_seeingam.append(seeingam)
      
        

        
        all_leftspectra_data.append(left_spectrum_data)
        all_rightspectra_data.append(right_spectrum_data)
        
        all_leftspectra_wl.append(left_spectrum_wl)
        all_rightspectra_wl.append(right_spectrum_wl)
        
        all_leftspectra_data_stat_err.append(left_spectrum_data_stat_err)
        all_rightspectra_data_stat_err.append(right_spectrum_data_stat_err)
        
        all_totleftspectra_data.append(totleft_spectrum_data)
        all_totrightspectra_data.append(totright_spectrum_data)
        
        all_titles.append(title)
       
        
        
        all_filt.append(filters)
        all_filt1.append(filters1)
        all_filt2.append(filters2)
            
        hdu_list.close()
        
    return all_header, \
        all_dates, \
        all_objects, \
        all_airmass, \
        all_titles, \
        all_exposures, \
        all_ut, all_ra,all_dec,all_epoch,all_zenith,all_ha,all_st,all_alt,all_focus,\
        all_temp, all_press,all_hum,all_windsp,\
        all_seeing,all_seeingam,\
        all_filt,all_filt1,all_filt2,\
        all_leftspectra_data, \
        all_rightspectra_data, \
        all_leftspectra_data_stat_err ,\
        all_rightspectra_data_stat_err ,\
        all_leftspectra_wl,\
        all_rightspectra_wl, \
        all_totleftspectra_data, \
        all_totrightspectra_data

#--------------------------------------------------------------------------------------------------------------------------        
        
        
        

def ShowCalibSpectrainPDF(all_spectra,all_spectra_stat_err,all_spectra_wl,all_titles,object_name,dir_top_img,all_filt,date,figname,tagname,NBIMGPERROW=2):
    """
    ShowCalibSpectrainPDF : Show the raw images without background subtraction
    ==============
    """
    
    NBSPEC=len(all_spectra)
    
    MAXIMGROW=max(2,int(m.ceil(float(NBSPEC)/float(NBIMGPERROW))))
    
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    titlepage='Calibrated 1D Spectra 1D for {}, date : {} , {} '.format(object_name,date,tagname)
    
    
    
    # loop on spectra  
    for index in np.arange(0,NBSPEC):
        
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
    
  
        spectrum=all_spectra[index]
        spectrum_err=all_spectra_stat_err[index]
        spectrum_wl=all_spectra_wl[index]
        
        
       
        axarr[iy,ix].errorbar(spectrum_wl,spectrum,yerr=spectrum_err,ecolor='grey',elinewidth=0.5)
        axarr[iy,ix].plot(spectrum_wl,spectrum,'-',color='blue',lw=1)
    
        
        thetitle="{} : {} : {} : {} ".format(index,all_titles[index],all_filt[index],tagname)
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        axarr[iy,ix].grid(True)
      
        max_y_to_plot=np.max(spectrum)*1.2
        
        YMIN=0.
        YMAX=max_y_to_plot
        for line in LINES:
            if line == O2 or line == HALPHA or line == HBETA or line == HGAMMA or line == HDELTA:
                axarr[iy,ix].plot([line['lambda'],line['lambda']],[YMIN,YMAX],'-',color='red',lw=0.5)
                axarr[iy,ix].text(line['lambda'],0.9*(YMAX-YMIN),line['label'],verticalalignment='bottom', horizontalalignment='center',color='red', fontweight='bold',fontsize=16)
        
        
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        axarr[iy,ix].set_xlim(0.,1200.)
        axarr[iy,ix].set_xlabel('$\lambda$ (nm)')
        axarr[iy,ix].set_ylabel('ADU (sum over 10 pix transv)')
        #axarr[iy,ix].text(0.,max_y_to_plot*1.1/1.2, all_filt[index],verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
    
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()    
#---------------------------------------------------------------------------------------------------------   


def ShowCalibSpectrainPDFSelect(all_spectra,all_spectra_stat_err,all_spectra_wl,all_titles,object_name,dir_top_img,all_filt,all_filt1,all_filt2,date,figname,tagname,NBIMGPERROW=2):
    """
    ShowCalibSpectrainPDF : Show the raw images without background subtraction
    ==============
    """
    
    NBSPEC=len(all_spectra)
    
    MAXIMGROW=max(2,int(m.ceil(float(NBSPEC)/float(NBIMGPERROW))))
    
    
    # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,figname)
    pp = PdfPages(figfilename) # create a pdf file
    titlepage='Calibrated 1D Spectra 1D for {}, date : {} , {} '.format(object_name,date,tagname)
    
    
    index=-1
    # loop on spectra  
    for idx in np.arange(0,NBSPEC):
        
        if all_filt1[idx]=='FGB37':
            index=index+1
        else:
            continue
        
        
        # new pdf page    
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(titlepage,size=20)
            
        # index of image in the pdf page    
        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
    
  
        spectrum=all_spectra[idx]
        spectrum_err=all_spectra_stat_err[idx]
        spectrum_wl=all_spectra_wl[idx]
        
        
       
        axarr[iy,ix].errorbar(spectrum_wl,spectrum,yerr=spectrum_err,ecolor='grey',elinewidth=0.5)
        axarr[iy,ix].plot(spectrum_wl,spectrum,'-',color='blue',lw=1)
    
        
        thetitle="{} : {} : {} : {} ".format(idx,all_titles[idx],all_filt[idx],tagname)
        axarr[iy,ix].set_title(thetitle,color='blue',fontweight='bold',fontsize=16)
        
        axarr[iy,ix].grid(True)
      
        max_y_to_plot=np.max(spectrum)*1.2
        
        
        YMIN=0.
        YMAX=max_y_to_plot
        
        for line in LINES:
            if line == O2 or line == HALPHA or line == HBETA or line == HGAMMA or line == HDELTA:
                axarr[iy,ix].plot([line['lambda'],line['lambda']],[YMIN,YMAX],'-',color='red',lw=0.5)
                axarr[iy,ix].text(line['lambda'],0.9*(YMAX-YMIN),line['label'],verticalalignment='bottom', horizontalalignment='center',color='red', fontweight='bold',fontsize=16)
        
        
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        axarr[iy,ix].set_xlim(0.,1200.)
        axarr[iy,ix].set_xlabel('$\lambda$ (nm)')
        axarr[iy,ix].set_ylabel('ADU (sum over 10 pix transv)')
        #axarr[iy,ix].text(0.,max_y_to_plot*1.1/1.2, all_filt[index],verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
    
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
    
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()    
#--------------------------------------------------------------------------------------------------------- 
    
def ShowCalibAndSimSpectrainPDF(all_spectra,all_wl,all_titles,object_name,all_filt,dir_top_img,sim_spec_data,sim_spec_wl,NBIMGPERROW=2,NormSpecRange= [790,810]):
    """
    ShowCalibAndSimSpectra: Compare spectra in data and sim in AnaCompareDataSimSpec.ipynb
    ==============
    
    
    input : NormSpecRange= [790,810] range in  which individual spectra data and sim are normalized relatively
    
    output:
        return relative calibration factors
    
    """
    
    # range where to normalize the spectra
    
    
    # number of spectra
    NBSPEC=len(all_spectra)
   
    MAXIMGROW=max(2,m.ceil(NBSPEC/NBIMGPERROW))
    #MAXIMGROW=int(MAXIMGROW)
        
    # calibration factor required
    calibdatasimfactor = []    
        
        
     # fig file specif
    NBIMGROWPERPAGE=5  # number of rows per pages
    PageNum=0          # page counter
    
    figfilename=os.path.join(dir_top_img,'input_calibratedandsim_spectra.pdf')
    pp = PdfPages(figfilename) # create a pdf file
    
    title='Calibrated spectra for {}'.format(object_name)
         
   
  
    
    for index in np.arange(0,NBSPEC):
        if index%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            f, axarr = plt.subplots(NBIMGROWPERPAGE,NBIMGPERROW,figsize=(25,30))
            f.suptitle(title,size=20,fontweight='bold')
 

        indexcut=index-PageNum*(NBIMGROWPERPAGE*NBIMGPERROW)    
        ix=indexcut%NBIMGPERROW
        iy=indexcut/NBIMGPERROW
        
        # extraction for data
        wl=all_wl[index]  # wavelength in data        
        spectrum=all_spectra[index]   # data
        maxdata_in_range=np.max(spectrum[np.where(np.logical_and(wl>NormSpecRange[0],wl<NormSpecRange[1]))]) # max in data in selected wl range
        
        # extraction for sim
        wl_sim= sim_spec_wl[index]       # wl
        spectrum_sim=sim_spec_data[index] # spec
        
        maxsim_in_range=np.max(spectrum_sim[np.where(np.logical_and(wl_sim>NormSpecRange[0],wl_sim<NormSpecRange[1]))]) # max in sim in selected wl range
        
        # calib factor per image
        calibfactor=maxdata_in_range/maxsim_in_range   # calib factor data/sim in wl range
        calibdatasimfactor.append(calibfactor) # for later analysis
        
        # plot spec data
        axarr[iy,ix].plot(wl,spectrum,'r-',lw=2,label='data') # plot data
        
        # renormalize sim
        spectrum_sim=spectrum_sim*calibfactor  # renormalize the sim to data units
        
        # plot sim
        axarr[iy,ix].plot(wl_sim,spectrum_sim,'b-',lw=1,label='sim')
        
        
        max_y_to_plot=spectrum[:].max()*1.4
        axarr[iy,ix].set_ylim(0.,max_y_to_plot)
        axarr[iy,ix].set_xlim(0.,1200.)
        axarr[iy,ix].text(0.,max_y_to_plot*1.1/1.4, all_filt[index],verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20)
       
       
        YMIN=0.
        YMAX=max_y_to_plot
    
        for line in LINES:
            if line == O2 or line == HALPHA or line == HBETA or line == HGAMMA or line == HDELTA or line ==O2B or line == O2Y or line == O2Z:
                axarr[iy,ix].plot([line['lambda'],line['lambda']],[YMIN,YMAX],'-',color='red',lw=0.5)
                axarr[iy,ix].text(line['lambda'],0.9*(YMAX-YMIN),line['label'],verticalalignment='bottom', horizontalalignment='center',color='red', fontweight='bold',fontsize=16)
        
 
        
        
                
        
        
        thetitle="{} : {}".format(index,all_titles[index])
        axarr[iy,ix].set_title(thetitle)
        #axarr[iy,ix].set_title(all_filt[index])
        axarr[iy,ix].set_xlabel("wavelength (nm)")
        
        axarr[iy,ix].get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        axarr[iy,ix].get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        axarr[iy,ix].grid(b=True, which='major' ,color='grey', linewidth=0.5)
        #axarr[iy,ix].grid(b=True, which='minor', color='grey', linewidth=0.5)
      
        axarr[iy,ix].legend()
        
        # save a new page
        if (index+1)%(NBIMGPERROW*NBIMGROWPERPAGE) == 0:
            PageNum+=1  # increase page Number
            f.savefig(pp, format='pdf')
            f.show()
            
                
          
    
    f.savefig(pp, format='pdf') 
    f.show()
    pp.close()
    return np.array(calibdatasimfactor)
#-----------------------------------------------------------------------------------------------------------



        
    
#----------------------------------------------------------------------------------------------------------
def PlotDataVsDateTime(all_dates,all_data,thetitle,thextitle,theytitle,dir_top_img,figname):
    """
    """
    
    NDATA=len(all_data)
    all_dt= [ parser.parse(all_dates[i]) for i in range(NDATA)]
    
    fig=plt.figure(figsize=(15,5))

    ax=fig.add_subplot(1,1,1)
    ax.plot_date(all_dt, all_data,marker='o',color='red',lw=0,label='airmass',linewidth=3)
    #ax.plot_date(all_dt, am,marker='.',color='blue',lw=0,label='relative airmass',linewidth=3)

    #ax.set_ylim(0.,2)

    date_range = all_dt[NDATA-1] - all_dt[0]

    if date_range > timedelta(days = 1):
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1,32), interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.get_xaxis().set_minor_locator(mdates.HourLocator(byhour=range(0,24,2)))
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,2)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_xaxis().set_minor_locator(mdates.MinuteLocator(byminute=range(0,60,5)))
    
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.grid(b=True, which='major', color='k', linewidth=1.0)
    ax.grid(b=True, which='minor', color='grey', linewidth=0.5)
    ax.set_ylabel(theytitle)
    ax.set_xlabel(thextitle)


    plt.title(thetitle)
    plt.legend(loc='best')

    figfilename=os.path.join(dir_top_img,figname)
    fig.savefig(figfilename)
#-----------------------------------------------------------------------------------------------------------

def GetSpectraFromIndexList(all_wl,all_spectra,idx_list):
    """
    GetSpectraFromIndexList(all_wl,all_spectra,idx_list)
    
    Select spectra from a list of indexes
    
    input :
        all_wl : all wavelength
        all_spectra :all input spectra
        idx_list : list of selected index
        
    output  :
        all_wl_sel : selected wavelength
        all_spectra_sel : selected spectra
    
    """
    NBSPEC=len(all_spectra)
    
    
    all_wl_sel=[]
    all_spectra_sel=[]
    
    for idx in np.arange(0,NBSPEC):
        if idx in idx_list:
            all_wl_sel.append(all_wl[idx])
            all_spectra_sel.append(all_spectra[idx])
    return all_wl_sel,all_spectra_sel
    
    
#----------------------------------------------------------------------------------------------------------    
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y
    return y[(window_len/2):-(window_len/2)] 


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def RemoveBadWavelengths(all_wl,all_spec,WLMIN=300.,WLMAX=1200.):
    """
    
    Remove wl<WLMIN=300. and wl>WLMAX
    
    """
    NSPEC=len(all_wl)
    all_wl_cut= []
    all_spec_cut = []
    
    for idx in np.arange(NSPEC):
        thewl=all_wl[idx]
        thespec=all_spec[idx]
        index_sel=np.where(np.logical_and(thewl>=WLMIN,thewl<=WLMAX))
        all_wl_cut.append(thewl[index_sel])
        all_spec_cut.append(thespec[index_sel])
    
    return all_wl_cut,all_spec_cut
#----------------------------------------------------------------- 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#    AnaEqWdtCalibSpectrum.ipynb         
#---------------------------------------------------------------------------------------------------------
def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
#------------------------------------------------------------------------------------------------------------  
    
import numpy.core.numeric as NX
from numpy.core import isscalar, abs, finfo, atleast_1d, hstack, dot
from numpy.lib.twodim_base import diag, vander
from numpy.lib.function_base import trim_zeros, sort_complex
from numpy.lib.type_check import iscomplex, real, imag
from numpy.linalg import eigvals, lstsq, inv
#---------------------------------------------------------------------------------------------------------
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """
    Least squares polynomial fit.
    Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
    to points `(x, y)`. Returns a vector of coefficients `p` that minimises
    the squared error.
    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (M,), optional
        weights to apply to the y-coordinates of the sample points.
    cov : bool, optional
        Return the estimate and the covariance matrix of the estimate
        If full is True, then cov is not returned.
    Returns
    -------
    p : ndarray, shape (M,) or (M, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.
    residuals, rank, singular_values, rcond :
        Present only if `full` = True.  Residuals of the least-squares fit,
        the effective rank of the scaled Vandermonde coefficient matrix,
        its singular values, and the specified value of `rcond`. For more
        details, see `linalg.lstsq`.
    V : ndarray, shape (M,M) or (M,M,K)
        Present only if `full` = False and `cov`=True.  The covariance
        matrix of the polynomial coefficient estimates.  The diagonal of
        this matrix are the variance estimates for each coefficient.  If y
        is a 2-D array, then the covariance matrix for the `k`-th data set
        are in ``V[:,:,k]``
    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if `full` = False.
        The warnings can be turned off by
        >>> import warnings
        >>> warnings.simplefilter('ignore', np.RankWarning)
    See Also
    --------
    polyval : Computes polynomial values.
    linalg.lstsq : Computes a least-squares fit.
    scipy.interpolate.UnivariateSpline : Computes spline fits.
    Notes
    -----
    The solution minimizes the squared error
    .. math ::
        E = \\sum_{j=0}^k |p(x_j) - y_j|^2
    in the equations::
        x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
        x[1]**n * p[0] + ... + x[1] * p[n-1] + p[n] = y[1]
        ...
        x[k]**n * p[0] + ... + x[k] * p[n-1] + p[n] = y[k]
    The coefficient matrix of the coefficients `p` is a Vandermonde matrix.
    `polyfit` issues a `RankWarning` when the least-squares fit is badly
    conditioned. This implies that the best fit is not well-defined due
    to numerical error. The results may be improved by lowering the polynomial
    degree or by replacing `x` by `x` - `x`.mean(). The `rcond` parameter
    can also be set to a value smaller than its default, but the resulting
    fit may be spurious: including contributions from the small singular
    values can add numerical noise to the result.
    Note that fitting polynomial coefficients is inherently badly conditioned
    when the degree of the polynomial is large or the interval of sample points
    is badly centered. The quality of the fit should always be checked in these
    cases. When polynomial fits are not satisfactory, splines may be a good
    alternative.
    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting
    .. [2] Wikipedia, "Polynomial interpolation",
           http://en.wikipedia.org/wiki/Polynomial_interpolation
    Examples
    --------
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> z = np.polyfit(x, y, 3)
    >>> z
    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254])
    It is convenient to use `poly1d` objects for dealing with polynomials:
    >>> p = np.poly1d(z)
    >>> p(0.5)
    0.6143849206349179
    >>> p(3.5)
    -0.34732142857143039
    >>> p(10)
    22.579365079365115
    High-order polynomials may oscillate wildly:
    >>> p30 = np.poly1d(np.polyfit(x, y, 30))
    /... RankWarning: Polyfit may be poorly conditioned...
    >>> p30(4)
    -0.80000000000000204
    >>> p30(5)
    -0.99999999999999445
    >>> p30(4.5)
    -0.10547061179440398
    Illustration:
    >>> import matplotlib.pyplot as plt
    >>> xp = np.linspace(-2, 6, 100)
    >>> _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
    >>> plt.ylim(-2,2)
    (-2, 2)
    >>> plt.show()
    """
    order = int(deg) + 1
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0

    # check arguments.
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    # set rcond
    if rcond is None:
        rcond = len(x)*finfo(x.dtype).eps

    # set up least squares equation for powers of x
    lhs = vander(x, order)
    rhs = y

    # apply weighting
    if w is not None:
        w = NX.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs *= w[:, NX.newaxis]
        if rhs.ndim == 2:
            rhs *= w[:, NX.newaxis]
        else:
            rhs *= w

    # scale lhs to improve condition number and solve
    scale = NX.sqrt((lhs*lhs).sum(axis=0))
    lhs /= scale
    
    
    print 'MyPolyfit lhs      =   ', lhs
    
    rcond =len(x)*2e-16
    
    c, resids, rank, s = lstsq(lhs, rhs, rcond)
    c = (c.T/scale).T  # broadcast scale coefficients
    
    
    print 'MyPolyfit  c, resids, rank, s   =  ',   c, resids, rank, s
    
    
    

    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order and not full:
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    if full:
        return c, resids, rank, s, rcond
    elif cov:
        Vbase = inv(dot(lhs.T, lhs))
        Vbase /= NX.outer(scale, scale)
        # Some literature ignores the extra -2.0 factor in the denominator, but
        #  it is included here because the covariance of Multivariate Student-T
        #  (which is implied by a Bayesian uncertainty analysis) includes it.
        #  Plus, it gives a slightly more conservative estimate of uncertainty.
        fac = resids / (len(x) - order - 2.0)
        if y.ndim == 1:
            return c, Vbase * fac
        else:
            return c, Vbase[:,:, NX.newaxis] * fac
    else:
        return c
#----------------------------------------------------------------------------------------------
def ShowEquivalentWidth(wl,spec,wl1,wl2,wl3,wl4,label='absortion line',fsize=(12,4)):
    """
    ShowEquivalentWidth : show how the equivalent width must be computed
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
    ##--------------
    ## Figure 1
    ##------------
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    axarr[0].plot(wl_cut,spec_cut,'b-')
    axarr[0].plot([wl2,wl2],[ymin,ymax],'r-.',lw=2)
    axarr[0].plot([wl3,wl3],[ymin,ymax],'r-.',lw=2)
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,1)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
    axarr[0].plot(x_cont,y_cont,'o')
    axarr[0].plot(fit_line_x,fit_line_y,'g--',lw=1)
    
    axarr[0].grid(True)
    axarr[0].set_xlabel('$\lambda$ (nm)')
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    
    #--------------
    # Figure 2
    #-----------
    axarr[1].plot(wl_cut,ratio,'b-')
    axarr[1].plot([wl2,wl2],[0,1.2],'r-.',lw=2)
    axarr[1].plot([wl3,wl3],[0,1.2],'r-.',lw=2)
    axarr[1].grid(True)
    
    axarr[1].set_xlabel('$\lambda$ (nm)')
    
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
    
    return equivalent_width
#-----------------------------------------------------------------------------------------------------
def ShowEquivalentWidth2(wl,spec,wl1,wl2,wl3,wl4,label='absortion line',fsize=(12,4)):
    """
    ShowEquivalentWidth : show how the equivalent width must be computed
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
    #############
    ## Figure 1
    ############
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    axarr[0].plot(wl_cut,spec_cut,'b-')
    axarr[0].plot([wl2,wl2],[ymin,ymax],'r-.',lw=2)
    axarr[0].plot([wl3,wl3],[ymin,ymax],'r-.',lw=2)
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,1)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
    axarr[0].plot(x_cont,y_cont,'o')
    axarr[0].plot(fit_line_x,fit_line_y,'g--',lw=1)
    
    axarr[0].grid(True)
    axarr[0].set_xlabel('$\lambda$ (nm)')
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    
    #-------------
    # Figure 2
    #-----------
    axarr[1].plot(wl_cut,ratio)
    axarr[1].plot([wl2,wl2],[0,1.2],'r-.',lw=2)
    axarr[1].plot([wl3,wl3],[0,1.2],'r-.',lw=2)
    axarr[1].grid(True)
    axarr[1].set_ylim(0.8*ratio.min(),1.2*ratio.max())
    
    axarr[1].set_xlabel('$\lambda$ (nm)')
    
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
    
    return equivalent_width    

#--------------------------------------------------------------------------------------------------
def ShowEquivalentWidthwthStatErrTOREMOVE(wl,spec,specerr,wl1,wl2,wl3,wl4,ndeg=3,thelabel='ShowEquivalentWidthwthStatErr'):
    """
    ******************************************************************************
    ShowEquivalentWidthwthStatErr : show how the equivalent width must be computed
    
    - with errors
    - with non linear method
    
    *********************************************************************************
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    spec_cut_err=specerr[selected_indexes]
    
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    #############
    ### Figure 1
    #############
    plt.figure()
    
    plt.plot(wl_cut,spec_cut,'b-')
    plt.errorbar(wl_cut,spec_cut,yerr=spec_cut_err,color='red',fmt='.')
    
    # vertical bars
    plt.plot([wl2,wl2],[ymin,ymax],'k-.',lw=2)
    plt.plot([wl3,wl3],[ymin,ymax],'k-.',lw=2)
    
    
    # continuum fit
    #------------------
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))

    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    y_cont_err=specerr[continuum_indexes]
    y_w=1./y_cont_err
    y_w[np.where(y_cont==0)]=0. # erase the empty bins
   
    
    popt_p , pcov_p= np.polyfit(x_cont, y_cont,ndeg,w=y_w,full=False,cov=True,rcond=2.0e-16*len(x_cont)) #rcond mandatory    
    z_cont_fit=popt_p
      

    pol_cont_fit=np.poly1d(z_cont_fit)
    
    # fitted curve with its error
    #---------------------------------
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    fit_line_y_err = []
    for thex in fit_line_x:
        dfdx = [ thex**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        fit_line_y_err.append(propagated_error)
    fit_line_y_err=np.array(fit_line_y_err)
    
    errorfill(fit_line_x,fit_line_y,fit_line_y_err, color='grey',ax=plt)
    plt.errorbar(x_cont,y_cont,yerr=y_cont_err,fmt='.',color='blue')
    plt.plot(fit_line_x,fit_line_y,'g--',lw=1)
    plt.xlabel(' wavelength (nm)')
    plt.ylabel(' ADU per second')
    
    plt.grid(True)
    plt.title(thelabel)
    
    # compute the ratio spectrum/continuum
    #-------------------------------------
    full_continum=pol_cont_fit(wl_cut)    
    
    full_continum_err= []
    for wl in wl_cut:
        dfdx = [ wl**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        full_continum_err.append(propagated_error)
    full_continum_err=np.array(full_continum_err)
    
    
    ratio=spec_cut/full_continum
    # error not correlated    
    ratio_err=ratio*np.sqrt( (spec_cut_err/spec_cut)**2+ (full_continum_err/full_continum)**2)
    
    
    
    
    ###################
    ### Second figure
    #################
    plt.figure()
    plt.plot(wl_cut,ratio,lw=2,color='blue')
    plt.errorbar(wl_cut,ratio,yerr=ratio_err,fmt='.',lw=2,color='red')
    plt.plot([wl2,wl2],[0,1.2],'k-.',lw=2)
    plt.plot([wl3,wl3],[0,1.2],'k-.',lw=2)
    plt.grid(True)
    plt.ylim(0.8*ratio.min(),1.2*ratio.max())
    plt.xlabel(' wavelength (nm)')
    plt.ylabel(' no unit')
    plt.title(thelabel)
    
    # Compute now the equavalent width
    #--------------------------------
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin in wavelength

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width and its error
    #-------------------------------------------------
    absorption_band=wl_bin_size*(1-ratio)
    absorption_band_error=wl_bin_size*ratio_err
    
    equivalent_width= absorption_band.sum()
    
    # quadratic sum of errors for each wl bin
    equivalend_width_error=np.sqrt((absorption_band_error*absorption_band_error).sum() )
    
    return equivalent_width,equivalend_width_error
#----------------------------------------------------------------------------------------------

def ShowEquivalentWidthNonLinearTOREMOVE(wl,spec,wl1,wl2,wl3,wl4,ndeg=3,thelabel='ShowEquivalentWidthNonLinear'):
    """
    ShowEquivalentWidthNonLinear : show how the equivalent width must be computed
    Do not use stats, for simulation by exammple
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
   
    
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    plt.figure()
    plt.plot(wl_cut,spec_cut,'r-')
    plt.plot([wl2,wl2],[ymin,ymax],'k-.',lw=2)
    plt.plot([wl3,wl3],[ymin,ymax],'k-.',lw=2)
    
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg,rcond=2.0e-16*len(x_cont))
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    plt.plot(x_cont,y_cont,marker='.',color='blue',lw=0)
    plt.plot(fit_line_x,fit_line_y,'g--',lw=2)
    
    plt.grid(True)
    plt.xlabel(' wavelength (nm)')
    plt.ylabel(' ADU per second')
    plt.title(thelabel)
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    
    plt.figure()
    plt.plot(wl_cut,ratio,'b-')
    plt.plot([wl2,wl2],[0,1.2],'k-.',lw=2)
    plt.plot([wl3,wl3],[0,1.2],'k-.',lw=2)
    plt.grid(True)
    plt.xlabel(' wavelength (nm)')
    plt.ylabel(' No unit')
    plt.ylim(0.8*ratio.min(),1.2*ratio.max())
    plt.title(thelabel)
    
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width
    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()
    
    
    return equivalent_width
#---------------------------------------------------------------------------------------------
def ShowEquivalentWidthNonLinear2(wl,spec,wl1,wl2,wl3,wl4,ndeg=3,label='absortion line',fsize=(12,4)):
    """
    ShowEquivalentWidth : show how the equivalent width must be computed
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
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
    
    return equivalent_width    

#---------------------------------------------------------------------------------------------    

def ShowEquivalentWidthNonLinearwthStatErr(wl,spec,specerr,wl1,wl2,wl3,wl4,ndeg=3,label='absortion line',fsize=(12,4)):
    """
    ***********************************************************************
    ShowEquivalentWidth2NonLinearwthStatErr : 
    
    Fit equivalent width:
    - Non linear
    - with errors
    
    ************************************************************************
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
    order=ndeg+1
    
    #######################
    ## Figure 1
    #########################
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    spec_cut_err=specerr[selected_indexes]
    
    
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    # plot points
    #axarr[0].plot(wl_cut,spec_cut,'b-')
    axarr[0].errorbar(wl_cut,spec_cut,yerr=spec_cut_err,color='red',fmt='.',lw=1)
    axarr[0].plot(wl_cut,spec_cut,'r-',lw=1)
    # plot vertical bars
    axarr[0].plot([wl2,wl2],[ymin,ymax],'k-.',lw=2)
    axarr[0].plot([wl3,wl3],[ymin,ymax],'k-.',lw=2)
    
    # continuum fit
    #----------------
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    

    # error for continum
    y_cont_err=specerr[continuum_indexes]
    y_w=1./y_cont_err
    y_w[np.where(y_cont==0)]=0. # erase the empty bins
    
    popt_p , pcov_p= np.polyfit(x_cont, y_cont,ndeg,w=y_w,full=False,cov=True,rcond=2.0e-16*len(x_cont)) #rcond mandatory
    
    z_cont_fit=popt_p
      
   
    pol_cont_fit=np.poly1d(z_cont_fit)
    

    # compute the fit and propagate the error
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    fit_line_y_err = []
    for thex in fit_line_x:
        dfdx = [ thex**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        fit_line_y_err.append(propagated_error)
    fit_line_y_err=np.array(fit_line_y_err)
    
    
    errorfill(fit_line_x,fit_line_y,fit_line_y_err,color='grey',ax=axarr[0])
    axarr[0].errorbar(x_cont,y_cont,yerr=y_cont_err,fmt='.',color='blue')
    
       
    axarr[0].grid(True)
    axarr[0].set_xlabel('$\lambda$ (nm)')
    axarr[0].set_ylabel('ADU per second')
    
    # compute the ratio spectrum/continuum and its error
    # -------------------------------------
    full_continum=pol_cont_fit(wl_cut)  
    
    full_continum_err= []
    external_indexes = []
    idx=0
    for wl in wl_cut:
        if wl<wl2 or wl>wl3:
            external_indexes.append(idx)
        idx+=1
        dfdx = [ wl**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        full_continum_err.append(propagated_error)
    full_continum_err=np.array(full_continum_err)
    
    
    ratio=spec_cut/full_continum
    # error not correlated    
    ratio_err=ratio*np.sqrt( (spec_cut_err/spec_cut)**2+ (full_continum_err/full_continum)**2)
    
    
    
    
    
    ##################
    # Figure 2
    ###################
    
    axarr[1].plot(wl_cut,ratio,lw=1,color='red')
    axarr[1].errorbar(wl_cut,ratio,yerr=ratio_err,fmt='.',lw=2,color='red')
    axarr[1].errorbar(wl_cut[external_indexes],ratio[external_indexes],yerr=ratio_err[external_indexes],fmt='.',lw=0,color='blue')
    
    
    axarr[1].plot([wl2,wl2],[0,1.2],'k-.',lw=2)
    axarr[1].plot([wl3,wl3],[0,1.2],'k-.',lw=2)
    axarr[1].grid(True)
    axarr[1].set_ylim(0.8*ratio.min(),1.2*ratio.max())
    axarr[1].set_xlabel('$\lambda$ (nm)')
    axarr[1].set_ylabel('ratio : no unit')
    
    
    #compute the equivalent width
    #-----------------------------
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width and its error (units nm because wl in nm)
    # ----------------------------------------------
    absorption_band=wl_bin_size*(1-ratio)
    absorption_band_error=wl_bin_size*ratio_err
    equivalent_width= absorption_band.sum()
    equivalent_width_err=np.sqrt((absorption_band_error**2).sum())
    
    
    title = 'Equivalent width computation for {}'.format(label)
    f.suptitle(title)
    
    return equivalent_width,equivalent_width_err
#--------------------------------------------------------------------------------------------
def ShowEquivalentWidthNonLinear(wl,spec,wl1,wl2,wl3,wl4,ndeg=3,label='absortion line',fsize=(12,4)):
    """
    ShowEquivalentWidth : show how the equivalent width must be computed
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
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
    
    return equivalent_width


#--------------------------------------------------------------------------------------------------
def ComputeEquivalentWidth(wl,spec,wl1,wl2,wl3,wl4):
    """
    ComputeEquivalentWidth : compute the equivalent width must be computed
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
     
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,1)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
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
#--------------------------------------------------------------------------------------------- 
def ComputeEquivalentWidthNonLinear(wl,spec,wl1,wl2,wl3,wl4,ndeg=3):
    """
    ComputeEquivalentWidth : compute the equivalent width must be computed
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
     
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg,rcond=2.0e-16*len(x_cont))
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
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
#----------------------------------------------------------------------------------------------
def ComputeEquivalentWidthNonLinearwthStatErr(wl,spec,specerr,wl1,wl2,wl3,wl4,ndeg=3):
    """
    ************************************************************************************
    ComputeEquivalentWidthNonLinearwthStatErr : compute the equivalent width must be computed
    
    *************************************************************************************
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
    
    # extract
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    spec_cut_err=specerr[selected_indexes]
    
    
    ymin=spec_cut.min()
    ymax=spec_cut.max()
     
    # continuum fit
    #---------------
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    y_cont_err=specerr[continuum_indexes]
    
    y_w=1./y_cont_err
    y_w[np.where(y_cont==0)]=0. # erase the empty bins
   
    
    popt_p , pcov_p= np.polyfit(x_cont, y_cont,ndeg,w=y_w,full=False,cov=True,rcond=2.0e-16*len(x_cont)) #rcond mandatory    
    z_cont_fit=popt_p
    
    
    
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    fit_line_y_err = []
    for thex in fit_line_x:
        dfdx = [ thex**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        fit_line_y_err.append(propagated_error)
    fit_line_y_err=np.array(fit_line_y_err)
    
    
    # compute the ratio spectrum/continuum and its error
    full_continum=pol_cont_fit(wl_cut)    
    
    full_continum_err= []
    for wl in wl_cut:
        dfdx = [ wl**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        full_continum_err.append(propagated_error)
    full_continum_err=np.array(full_continum_err)
    
    
    ratio=spec_cut/full_continum
    # error not correlated    
    ratio_err=ratio*np.sqrt( (spec_cut_err/spec_cut)**2+ (full_continum_err/full_continum)**2)
    
    

    # compute bin size in the band
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
    
    
    
    # calculation of equivalent width    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum() 
    
    # return equavalent width error
    absorption_band_error=wl_bin_size*ratio_err
    # quadratic sum of errors for each wl bin
    equivalend_width_error=np.sqrt((absorption_band_error*absorption_band_error).sum() )
    
    return equivalent_width,equivalend_width_error
#----------------------------------------------------------------------------------------------------- 
def ShowAllEquivalentWidth(all_wl,all_spec,all_filt,wl1,wl2,wl3,wl4,label='absorption line'):
        
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        
        newlabel = label+ ' spec {} with disp {} (fit BG-L)'.format(index,all_filt[index]) 
        
        ShowEquivalentWidth(wl,spectrum,wl1,wl2,wl3,wl4,label=newlabel,fsize=(9,3))
#-----------------------------------------------------------------------------------------------        
def ShowAllEquivalentWidthNonLinear(all_wl,all_spec,all_filt,wl1,wl2,wl3,wl4,ndeg=3,label='absorption line'):
        
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        
        newlabel = label+ ' spec {} with disp {} (fit BG-N)'.format(index,all_filt[index]) 
        
        ShowEquivalentWidthNonLinear(wl,spectrum,wl1,wl2,wl3,wl4,ndeg=ndeg,label=newlabel,fsize=(9,3))
        
        
#-----------------------------------------------------------------------------------------------        
def ShowAllEquivalentWidthNonLinearwthStatErr(all_wl,all_spec,all_spec_err,all_filt,wl1,wl2,wl3,wl4,ndeg=3,label='absorption line'):
        
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        err=all_spec_err[index]
        
        newlabel = label+ ' spec {} with disp {} (fit BG-N+ stat err)'.format(index,all_filt[index]) 
        
        ShowEquivalentWidthNonLinearwthStatErr(wl,spectrum,err,wl1,wl2,wl3,wl4,ndeg=ndeg,label=newlabel,fsize=(9,3))
#----------------------------------------------------------------------------------------------------        
        
#---------------------------------------------------------------------------------------------------
def ComputeAllEquivalentWidth(all_wl,all_spec,wl1,wl2,wl3,wl4):
    
    EQW_coll = []
    
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        eqw=ComputeEquivalentWidth(wl,spectrum,wl1,wl2,wl3,wl4)
        EQW_coll.append(eqw)
        
    return np.array(EQW_coll)
#--------------------------------------------------------------------------------------------
        
#---------------------------------------------------------------------------------------------------
def ComputeAllEquivalentWidthNonLinear(all_wl,all_spec,wl1,wl2,wl3,wl4,ndeg=3):
    
    EQW_coll = []
    
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        eqw=ComputeEquivalentWidthNonLinear(wl,spectrum,wl1,wl2,wl3,wl4,ndeg=ndeg)
        EQW_coll.append(eqw)
        
    return np.array(EQW_coll)
#--------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def ComputeAllEquivalentWidthNonLinearwthStatErr(all_wl,all_spec,all_spec_err,wl1,wl2,wl3,wl4,ndeg=3):
    
    EQW_coll = []
    EQWErr_coll = []
    
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        err=all_spec_err[index]
        
        eqw,eqw_err=ComputeEquivalentWidthNonLinearwthStatErr(wl,spectrum,err,wl1,wl2,wl3,wl4,ndeg=ndeg)
        EQW_coll.append(eqw)
        EQWErr_coll.append(eqw_err)
        
    EQW_coll=np.array(EQW_coll)    
    EQWErr_coll=np.array(EQWErr_coll)    
    return EQW_coll, EQWErr_coll
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PlotEquivalentWidthVsAirMass(all_eqw_width,all_eqw_width_sim,all_am,all_filt,tagabsline,dir_top_img,figname,spec_err=None):
    """
    """
    am0 = []
    eqw0 = []
    err0 = []
    
    am1 = []
    eqw1 = []
    err1 = []
    
    am2 = []
    eqw2 = []
    err2 = []
    
    am3 = []
    eqw3 = []
    err3 = []
    
    am4 = []
    eqw4 = []
    err4 = []
    
    
    for index,eqw in np.ndenumerate(all_eqw_width):
        idx=index[0]
        grating_name=all_filt[idx]
        am=all_am[idx]
        err=0
        if spec_err != None:
            err=spec_err[idx]
        
        
        if re.search(Disp_names[0],grating_name):
            am0.append(am)
            eqw0.append(eqw)
            err0.append(err)
        elif re.search(Disp_names[1],grating_name):
            am1.append(am)
            eqw1.append(eqw)
            err1.append(err)
        elif re.search(Disp_names[2],grating_name):
            am2.append(am)
            eqw2.append(eqw)
            err2.append(err)
        elif re.search(Disp_names[3],grating_name):
            am3.append(am)
            eqw3.append(eqw)
            err3.append(err)
        elif re.search(Disp_names[4],grating_name):
            am4.append(am)
            eqw4.append(eqw)
            err4.append(err)
        else:
            print 'disperser ',grating_name,' not found'
            
    fig=plt.figure(figsize=(20,8))

    ax=fig.add_subplot(1,1,1)
    
    ax.errorbar(am0,eqw0,err0, fmt='o',color='red')
    ax.plot(am0,eqw0,marker='o',color='red',lw=1,label=Disp_names[0])
    ax.errorbar(am1,eqw1,err1, fmt='o',color='blue')
    ax.plot(am1,eqw1,marker='o',color='blue',lw=1,label=Disp_names[1])
    ax.errorbar(am2,eqw2,err2, fmt='o',color='green')
    ax.plot(am2,eqw2,marker='o',color='green',lw=1,label=Disp_names[2])
    ax.errorbar(am3,eqw3,err3, fmt='o',color='cyan')
    ax.plot(am3,eqw3,marker='o',color='cyan',lw=1,label=Disp_names[3])
    ax.errorbar(am4,eqw4,err4, fmt='o',color='magenta')
    ax.plot(am4,eqw4,marker='o',color='magenta',lw=1,label=Disp_names[4])
    
    ax.plot(all_am,all_eqw_width_sim,marker='o',color='black',lw=1,label='Sim')

    #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())   
    #ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.grid(b=True, which='major', color='grey', linewidth=1.0)
    #ax.grid(b=True, which='minor', color='grey', linewidth=0.5)

    ax.set_ylabel('Equivalent width (nm)')
    ax.set_xlabel('AirMass')



    title='Equivalent Width for {} vs airmass'.format(tagabsline)

    plt.title(title)
    plt.legend(loc='best')

    figfilename=os.path.join(dir_top_img,figname)
    fig.savefig(figfilename)
 #--------------------------------------------------------------------------------------------  

def PlotEquivalentWidthVsTime(all_eqw_width,all_eqw_width_sim,all_am,all_dt,all_filt,tagabsline,dir_top_img,figname,spec_err=None):
    """
    """
    am0 = []
    eqw0 = []
    tim0 = []
    err0 = []
    
    am1 = []
    eqw1 = []
    tim1=[]
    err1 = []
    
    am2 = []
    eqw2 = []
    tim2=[]
    err2 = []
    
    am3 = []
    eqw3 = []
    tim3=[]
    err3 = []
    
    am4 = []
    eqw4 = []
    tim4=[]
    err4 = []
    
    NDATA=len(all_eqw_width)
    
    date_range = all_dt[NDATA-1] - all_dt[0]
    
    
    for index,eqw in np.ndenumerate(all_eqw_width):
        idx=index[0]
        grating_name=all_filt[idx]
        am=all_am[idx]
        err=0
        if spec_err != None:
            err=spec_err[idx]
        
        
        
        if re.search(Disp_names[0],grating_name):
            am0.append(am)
            eqw0.append(eqw)
            tim0.append(all_dt[idx])
            err0.append(err)
        elif re.search(Disp_names[1],grating_name):
            am1.append(am)
            eqw1.append(eqw)
            tim1.append(all_dt[idx])
            err1.append(err)
        elif re.search(Disp_names[2],grating_name):
            am2.append(am)
            eqw2.append(eqw)
            tim2.append(all_dt[idx])
            err2.append(err)
        elif re.search(Disp_names[3],grating_name):
            am3.append(am)
            eqw3.append(eqw)
            tim3.append(all_dt[idx])
            err3.append(err)
        elif re.search(Disp_names[4],grating_name):
            am4.append(am)
            eqw4.append(eqw)
            tim4.append(all_dt[idx])
            err4.append(err)
        else:
            print 'disperser ',grating_name,' not found'
            
    fig=plt.figure(figsize=(20,8))

    ax=fig.add_subplot(1,1,1)

    ax.errorbar(tim0,eqw0,err0, fmt='o',color='red')
    ax.plot_date(tim0,eqw0,'-',color='red',lw=1,label=Disp_names[0])
    ax.errorbar(tim1,eqw1,err1, fmt='o',color='blue')
    ax.plot_date(tim1,eqw1,'-',color='blue',lw=1,label=Disp_names[1])
    ax.errorbar(tim2,eqw2,err2, fmt='o',color='green')
    ax.plot_date(tim2,eqw2,'-',color='green',lw=1,label=Disp_names[2])
    ax.errorbar(tim3,eqw3,err3, fmt='o',color='cyan')
    ax.plot_date(tim3,eqw3,'-',color='cyan',lw=1,label=Disp_names[3])
    ax.errorbar(tim4,eqw4,err4, fmt='o',color='magenta')
    ax.plot_date(tim4,eqw4,'-',color='magenta',lw=1,label=Disp_names[4])
    
    ax.plot(all_dt,all_eqw_width_sim,'o',color='black')
    ax.plot_date(all_dt,all_eqw_width_sim,'-',color='black',lw=1,label='Sim')
    
    
    date_range = all_dt[NDATA-1] - all_dt[0]

    if date_range > datetime.timedelta(days = 1):
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1,32), interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.get_xaxis().set_minor_locator(mdates.HourLocator(byhour=range(0,24,2)))
        #ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,2)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_xaxis().set_minor_locator(mdates.MinuteLocator(byminute=range(0,60,5)))
    
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.grid(b=True, which='major', color='k', linewidth=2.0)
    ax.grid(b=True, which='minor', color='grey', linewidth=0.5)
    ax.set_ylabel('Equivalent width (nm)')
    ax.set_xlabel('time')
#    ax.set_ylim(0.,5.)



    title='Equivalent Width for {} vs time'.format(tagabsline)

    plt.title(title)
    plt.legend(loc='best')
    
    figfilename=os.path.join(dir_top_img,figname)
    fig.savefig(figfilename)
   