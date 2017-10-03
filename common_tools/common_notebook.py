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

# Definitions of some constants
#------------------------------------------------------------------------------
Filt_names= ['dia Ron400', 'dia Thor300', 'dia HoloPhP', 'dia HoloPhAg', 'dia HoloAmAg', 'dia Ron200','Unknown']

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
        hdu_list.close()
        
    return all_dates,all_airmass,all_images,all_titles,all_header,all_expo,all_filt

#------------------------------------------------------------------------------

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
    title='Images of {}'.format(object_name)
    plt.suptitle(title,size=16)    

#------------------------------------------------------------------------------
    
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
    
    return thespectra,thespectraUp,thespectraDown


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
        if filt == 'dia Ron400':
            filt0_idx.append(index)
        elif filt == 'dia Thor300':
            filt1_idx.append(index)
        elif filt == 'dia HoloPhP':
            filt2_idx.append(index)
        elif filt == 'dia HoloPhAg':
            filt3_idx.append(index)
        elif filt == 'dia HoloAmAg':
            filt4_idx.append(index)
        elif filt == 'dia Ron200':
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
    
    return filt0_idx,filt1_idx,filt2_idx,filt3_idx,filt4_idx,filt5_idx,filt6_idx

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
    

def guess_central_position(listofimages,DeltaX,DeltaY,dwc,filt0_idx,filt1_idx,filt2_idx,filt3_idx,filt4_idx,filt5_idx=None,filt6_idx=None):
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
    
    print 'Check fit quality :: '
    print ' ==========================='
    
    # find bad ids for filter 1 and correct for 
        
        
    # filter 0        
    aver_x0,std_x0,bad_idx_x0=check_bad_guess(x_guess,filt0_idx)
    if (bad_idx_x0.shape[0] != 0):
        print 'bad filt 0 x : ',bad_idx_x0
        x_guess[bad_idx_x0]=aver_x0    # do the correction
    
    aver_y0,std_y0,bad_idx_y0=check_bad_guess(y_guess,filt0_idx)
    if (bad_idx_y0.shape[0] != 0):
        print 'bad filt 0 y : ',bad_idx_y0
        y_guess[bad_idx_y0]=aver_y0    # do the correction     
        

    # filter 1        
    aver_x1,std_x1,bad_idx_x1=check_bad_guess(x_guess,filt1_idx)
    if (bad_idx_x1.shape[0] != 0):
        print 'bad filt 1 x : ',bad_idx_x1
        
        # !!!!!!!!!!!!!!!!!!!!!! Special for first Thorlab image
        idx_to_remove=0
        print 'remove from bad idx x1 : ',idx_to_remove
        bad_idx_x1=remove_from_bad(bad_idx_x1,idx_to_remove)           
        print 'new bad filt 1 x : ',bad_idx_x1
        x_guess[bad_idx_x1]=aver_x1    # do the correction
    
    aver_y1,std_y1,bad_idx_y1=check_bad_guess(y_guess,filt1_idx)
    if (bad_idx_y1.shape[0] != 0):
        
        print 'bad filt 1 y : ',bad_idx_y1
         # !!!!!!!!!!!!!!!!!!!!!! Special for first Thorlab image
        idx_to_remove=0        
        print 'remove from bad idx y1 : ',idx_to_remove
        bad_idx_y1=remove_from_bad(bad_idx_y1,idx_to_remove)           
        print 'new bad filt 1 y : ',bad_idx_y1
        
        y_guess[bad_idx_y1]=aver_y1    # do the correction
    
 
    # filter 2        
    aver_x2,std_x2,bad_idx_x2=check_bad_guess(x_guess,filt2_idx)
    if (bad_idx_x2.shape[0] != 0):
        print 'bad filt 2 x : ',bad_idx_x2
        x_guess[bad_idx_x2]=aver_x2    # do the correction
    
    aver_y2,std_y2,bad_idx_y2=check_bad_guess(y_guess,filt2_idx)
    if (bad_idx_y2.shape[0] != 0):
        print 'bad filt 2 y : ',bad_idx_y2
        y_guess[bad_idx_y2]=aver_y2    # do the correction
        
        
    # filter 3        
    aver_x3,std_x3,bad_idx_x3=check_bad_guess(x_guess,filt3_idx)
    if (bad_idx_x3.shape[0] != 0):
        print 'bad bad filt 3 x : ',bad_idx_x3
        x_guess[bad_idx_x3]=aver_x3    # do the correction
    
    aver_y3,std_y3,bad_idx_y3=check_bad_guess(y_guess,filt3_idx)
    if (bad_idx_y3.shape[0] != 0):
        print 'bad filt 3 y : ',bad_idx_y3
        y_guess[bad_idx_y3]=aver_y3    # do the correction
        
    # filter 4        
    aver_x4,std_x4,bad_idx_x4=check_bad_guess(x_guess,filt4_idx)
    if (bad_idx_x4.shape[0] != 0):
        print 'bad filt 4 x : ',bad_idx_x4
        x_guess[bad_idx_x4]=aver_x4    # do the correction
    
    aver_y4,std_y4,bad_idx_y4=check_bad_guess(y_guess,filt4_idx)
    if (bad_idx_y4.shape[0] != 0):
        print 'bad filt 4 y : ',bad_idx_y4
        y_guess[bad_idx_y4]=aver_y4    # do the correction        
 
    
    # filter 5 
    if filt5_idx != None and filt5_idx.shape[0] != 0:       
        aver_x5,std_x5,bad_idx_x5=check_bad_guess(x_guess,filt5_idx)
        if (bad_idx_x5.shape[0] != 0):
            print 'bad filt 5 x : ',bad_idx_x5
            x_guess[bad_idx_x5]=aver_x5    # do the correction
    
        aver_y5,std_y5,bad_idx_y5=check_bad_guess(y_guess,filt5_idx)
        if (bad_idx_y5.shape[0] != 0):
            print 'bad filt 5 y : ',bad_idx_y5
            y_guess[bad_idx_y5]=aver_y5    # do the correction  
            
        # filter 6 
    if filt6_idx != None and filt6_idx.shape[0] != 0:       
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



    