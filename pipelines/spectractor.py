import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import copy
sys.path.append("../common_tools/")
from tools import *
from holo_specs import *
from targets import *
from parameters import *

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

from skimage.feature import hessian_matrix

import logging

MY_FORMAT = "%(asctime)-24s %(name)-12s %(funcName)-20s %(levelname)-6s %(message)s"   #### ce sont des variables de classes
logging.basicConfig(format=MY_FORMAT, level=logging.WARNING)

def set_logger(logger):
    my_logger = logging.getLogger(logger)
    if VERBOSE > 0:
        my_logger.setLevel(logging.INFO)
    else:
        my_logger.setLevel(logging.WARNING)
    if DEBUG: my_logger.setLevel(logging.DEBUG)
    return my_logger


class Image():

    def __init__(self,filename,target=""):
        """
        Args:
            filename (:obj:`str`): path to the image
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.filename = filename
        self.load(filename)
        # Load the target if given
        self.target = None
        if target != "": self.target=Target(target,verbose=VERBOSE)
        self.err = None

    def load(self,filename):
        """
        Args:
            filename (:obj:`str`): path to the image
        """
        self.my_logger.info('Loading image %s...' % filename)
        hdu_list = fits.open(filename)
        self.header = hdu_list[0].header
        self.data = hdu_list[0].data
        self.date_obs = self.header['DATE-OBS']
        self.airmass = self.header['AIRMASS']
        self.expo = self.header['EXPTIME']
        self.filters = self.header['FILTERS']
        self.filter = self.header['FILTER1']
        self.disperser = self.header['FILTER2']
        IMSIZE = int(self.header['XLENGTH'])
        PIXEL2ARCSEC = float(self.header['XPIXSIZE'])
        if self.header['YLENGTH'] != IMSIZE:
            self.my_logger.warning('Image rectangular: X=%d pix, Y=%d pix' % (IMSIZE, self.header['YLENGTH']))
        if self.header['YPIXSIZE'] != PIXEL2ARCSEC:
            self.my_logger.warning('Pixel size rectangular: X=%d arcsec, Y=%d arcsec' % (PIXEL2ARCSEC, self.header['YPIXSIZE']))
        self.coord = SkyCoord(self.header['RA']+' '+self.header['DEC'],unit=(units.hourangle, units.deg),obstime=self.header['DATE-OBS'] )
        self.my_logger.info('Image loaded')
        # Load the disperser
        self.my_logger.info('Loading disperser %s...' % self.disperser)
        self.disperser = Hologram(self.disperser,data_dir=HOLO_DIR,verbose=VERBOSE)

    def find_target(self,guess,rotated=False):
        """
        Find precisely the position of the targeted object.
        
        Args:
            guess (:obj:`list`): [x,y] guessed position of th target
        """
        x0 = guess[0]
        y0 = guess[1]
        #y0 = int(0.5*IMSIZE + (self.coord.dec.arcsec - self.target.coord.dec.arcsec)/PIXEL2ARCSEC)
        #x0 = int(0.5*IMSIZE + (self.coord.ra.arcsec - self.target.coord.ra.arcsec)*np.cos(self.coord.dec.radian)/PIXEL2ARCSEC)
        #print x0,y0,PIXEL2ARCSEC



        #ra0 = self.coord.ra.radian
        #dec0 = self.coord.dec.radian
        #ra = self.target.coord.ra.radian
        #dec = self.target.coord.dec.radian
        #bottom = np.sin(dec)*np.sin(dec0) + np.cos(dec)*np.cos(dec0)*np.cos(ra-ra0)
        #xi = np.cos(dec) * np.sin(ra-ra0) / bottom
        #eta = (np.sin(dec)*np.cos(dec0) - np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)) / bottom
        #xi = xi * 180.0 / np.pi * 3600.
        #eta = eta * 180.0 / np.pi * 3600. 
        #x0 = int(0.5*IMSIZE - xi / PIXEL2ARCSEC )
        #y0 = int(0.5*IMSIZE - eta / PIXEL2ARCSEC)
        #print xi/60., eta/60.
        #print self.coord
        #print self.target.coord
        #print (self.coord.dec.arcmin - self.target.coord.dec.arcmin)
        #print (self.coord.ra.arcmin - self.target.coord.ra.arcmin)
        #print (self.coord.ra.arcmin - self.target.coord.ra.arcmin)*np.cos(self.coord.dec.radian)
        #print x0,y0,PIXEL2ARCSEC
        Dx = XWINDOW
        Dy = YWINDOW
        if rotated:
            angle = self.rotation_angle*np.pi/180.
            rotmat = np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            vec = np.array(self.target_pixcoords) - 0.5*np.array(self.data.shape)
            guess2 =  np.dot(rotmat,vec) + 0.5*np.array(self.data_rotated.shape)
            x0 = int(guess2[0,0])
            y0 = int(guess2[0,1])
        if rotated:
            Dx = XWINDOW_ROT
            Dy = YWINDOW_ROT
            sub_image = np.copy(self.data_rotated[y0-Dy:y0+Dy,x0-Dx:x0+Dx])
        else:
            sub_image = np.copy(self.data[y0-Dy:y0+Dy,x0-Dx:x0+Dx])
        #fig = plt.figure()
        #plt.imshow(self.data,origin='lower',vmin=0,vmax=0.02*np.max(self.data),cmap='rainbow')
        #plt.plot([x0],[y0],'ko')
        #print x0, y0
        #plt.show()
        NX=sub_image.shape[1]
        NY=sub_image.shape[0]        
        profile_X=np.sum(sub_image,axis=0)
        profile_Y=np.sum(sub_image,axis=1)
        profile_X -= np.min(profile_X)
        profile_Y -= np.min(profile_Y)
        X_=np.arange(NX)
        Y_=np.arange(NY)

        avX,sigX=weighted_avg_and_std(X_,profile_X**4) 
        avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)

        if profile_X[int(avX)] < 0.8*np.max(profile_X) :
            self.my_logger.warning('X position determination of the target probably wrong')

        if profile_Y[int(avY)] < 0.8*np.max(profile_Y) :
            self.my_logger.warning('Y position determination of the target probably wrong')

        theX=x0-Dx+avX
        theY=y0-Dy+avY
        
        if DEBUG:
            profile_X_max=np.max(profile_X)*1.2
            profile_Y_max=np.max(profile_Y)*1.2

            f, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(15,4))
            ax1.imshow(sub_image,origin='lower',vmin=0,vmax=10000,cmap='rainbow')
            ax1.plot([avX],[avY],'ko')
            ax1.grid(True)
            ax1.set_xlabel('X - pixel')
            ax1.set_ylabel('Y - pixel')

            ax2.plot(X_,profile_X,'r-',lw=2)
            ax2.plot([Dx,Dx],[0,profile_X_max],'y-',label='old',lw=2)
            ax2.plot([avX,avX],[0,profile_X_max],'b-',label='new',lw=2)
            ax2.grid(True)
            ax2.set_xlabel('X - pixel')
            ax2.legend(loc=1)

            ax3.plot(Y_,profile_Y,'r-',lw=2)
            ax3.plot([Dy,Dy],[0,profile_Y_max],'y-',label='old',lw=2)
            ax3.plot([avY,avY],[0,profile_Y_max],'b-',label='new',lw=2)
            ax3.grid(True)
            ax3.set_xlabel('Y - pixel')
            ax3.legend(loc=1)

            plt.show()

        self.my_logger.info('X,Y target position in pixels: %.3f,%.3f' % (theX,theY))
        if rotated:
            self.target_pixcoords_rotated = [theX,theY]
        else:
            self.target_pixcoords = [theX,theY]
        return [theX,theY]

    def compute_rotation_angle_hessian(self, deg_threshold = 10, width_cut = YWINDOW, right_edge = IMSIZE-200, margin_cut=12):
        x0, y0 = np.array(self.target_pixcoords).astype(int)
        # extract a region 
        data=np.copy(self.data[y0-width_cut:y0+width_cut,0:right_edge])
        # compute hessian matrices on the image
        Hxx, Hxy, Hyy = hessian_matrix(data, sigma=3, order = 'xy')
        lambda_plus = 0.5*( (Hxx+Hyy) + np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
        lambda_minus = 0.5*( (Hxx+Hyy) - np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
        theta = 0.5*np.arctan2(2*Hxy,Hyy-Hxx)*180/np.pi
        # remove the margins
        lambda_minus = lambda_minus[margin_cut:-margin_cut,margin_cut:-margin_cut]
        lambda_plus = lambda_plus[margin_cut:-margin_cut,margin_cut:-margin_cut]
        theta = theta[margin_cut:-margin_cut,margin_cut:-margin_cut]
        # thresholds
        lambda_threshold = np.min(lambda_minus)
        mask = np.where(lambda_minus>lambda_threshold)
        theta_mask = np.copy(theta)
        theta_mask[mask]=np.nan
        minimum_pixels = 0.01*2*width_cut*right_edge
        while len(theta_mask[~np.isnan(theta_mask)]) < minimum_pixels:
            lambda_threshold /= 2
            mask = np.where(lambda_minus>lambda_threshold)
            theta_mask = np.copy(theta)
            theta_mask[mask]=np.nan
            #print len(theta_mask[~np.isnan(theta_mask)]), lambda_threshold
        theta_guess = self.disperser.theta(self.target_pixcoords)
        mask2 = np.where(np.abs(theta-theta_guess)>deg_threshold)
        theta_mask[mask2] = np.nan
        theta_hist = []
        theta_hist = theta_mask[~np.isnan(theta_mask)].flatten()
        theta_median = np.median(theta_hist)
        if abs(theta_median-theta_guess)>0.1:
            self.my_logger.warning('Interpolated angle and fitted angle disagrees with more than 0.1 degree:  %.2f vs %.2f' % (theta_median,theta_guess))

        if DEBUG:
            f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
            xindex=np.arange(data.shape[1])
            x_new = np.linspace(xindex.min(),xindex.max(), 50)
            y_new = width_cut + (x_new-x0)*np.tan(theta_median*np.pi/180.)
            ax1.imshow(theta_mask,origin='lower',cmap=cm.brg,aspect='auto',vmin=-deg_threshold,vmax=deg_threshold)
            #ax1.imshow(np.log10(data),origin='lower',cmap="jet",aspect='auto')
            ax1.plot(x_new,y_new,'b-')
            ax1.set_ylim(0,2*width_cut)
            ax1.grid(True)
            n,bins, patches = ax2.hist(theta_hist,bins=int(np.sqrt(len(theta_hist))))
            ax2.plot([theta_median,theta_median],[0,np.max(n)])
            ax2.set_xlabel("Rotation angles [degrees]")
            plt.show()

        return theta_median
    

    def turn_image(self):
        self.rotation_angle = self.compute_rotation_angle_hessian()
        self.my_logger.info('Rotate the image with angle theta=%.2f degree' % self.rotation_angle)
        self.data_rotated = np.copy(self.data)
        if not np.isnan(self.rotation_angle):
            self.data_rotated=ndimage.interpolation.rotate(self.data,self.rotation_angle,prefilter=False,order=5)
        if DEBUG:
            f, (ax1,ax2) = plt.subplots(2,1,figsize=[8,8])
            y0 = int(self.target_pixcoords[1])
            ax1.imshow(np.log10(self.data[y0-YWINDOW:y0+YWINDOW,200:-200]),origin='lower',cmap='rainbow',aspect="auto")
            #ax1.imshow(np.log10(self.data),origin='lower',cmap='rainbow',aspect="auto")
            ax1.plot([0,self.data.shape[0]-200],[YWINDOW,YWINDOW],'w-')
            ax1.grid(color='white', ls='solid')
            ax1.grid(True)
            ax1.set_title('Raw image (log10 scale)')
            ax2.imshow(np.log10(self.data_rotated[y0-YWINDOW:y0+YWINDOW,200:-200]),origin='lower',cmap='rainbow',aspect="auto")
            #ax2.imshow(np.log10(self.data_rotated),origin='lower',cmap='rainbow',aspect="auto")
            ax2.plot([0,self.data_rotated.shape[0]-200],[YWINDOW,YWINDOW],'w-')
            ax2.grid(color='white', ls='solid')
            ax2.grid(True)
            ax2.set_title('Turned image (log10 scale)')
            plt.show()

    def extract_spectrum_from_image(self,w=3,ws=[8,30],right_edge=1800):
        self.my_logger.info('Extracting spectrum from image: spectrum with width 2*%d pixels and background from %d to %d pixels' % (w,ws[0],ws[1]))
        data=np.copy(self.data_rotated)[:,0:right_edge]
        if self.expo <= 0 :
            data /= self.expo
        y0 = int(self.target_pixcoords_rotated[1])
        spectrum2D=np.copy(data[y0-w:y0+w,:])
        xprofile=np.mean(spectrum2D,axis=0)
        
        ### Lateral bands to remove sky background
        ### ---------------------------------------
        Ny, Nx =  data.shape
        ymax = min(Ny,y0+ws[1])
        ymin = max(0,y0-ws[1])
        spectrum2DUp=np.copy(data[y0+ws[0]:ymax,:])
        spectrum2DUp = filter_stars_from_bgd(spectrum2DUp,margin_cut=1)
        xprofileUp=np.nanmedian(spectrum2DUp,axis=0)#*float(ymax-ws[0]-y0)
        spectrum2DDown=np.copy(data[ymin:y0-ws[0],:])
        spectrum2DDown = filter_stars_from_bgd(spectrum2DDown,margin_cut=1)
        xprofileDown=np.nanmedian(spectrum2DDown,axis=0)#*float(y0-ws[0]-ymin)
        
        #Clean_Up, Clean_Do,Clean_Av=CleanBadPixels(thespectraUp,thespectraDown)
        xprofile_background = 0.5*(xprofileUp+xprofileDown)

        spectrum = Spectrum(Image=self)
        spectrum.data = xprofile - xprofile_background
        if DEBUG:
            spectrum.plot_spectrum()
    
        return spectrum


class Spectrum():
    """ Spectrum class used to store information and methods
    relative to spectra nd their extraction.
    """
    def __init__(self,filename="",Image=None):
        """
        Args:
            filename (:obj:`str`): path to the image
            Image (:obj:`Image`): copy info from Image object
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.target = None
        if filename != "" :
            self.filename = filename
            self.load(filename)
        if Image is not None:
            self.date_obs = Image.date_obs
            self.airmass = Image.airmass
            self.expo = Image.expo
            self.filters = Image.filters
            self.filter = Image.filter
            self.disperser = Image.disperser
            self.target = Image.target
            self.target_pixcoords = Image.target_pixcoords
            self.target_pixcoords_rotated = Image.target_pixcoords_rotated
            self.my_logger.info('Spectrum info copied from Image')
        self.data = None
        self.err = None
        self.lambdas = None
        self.order = 1
        self.load_filter()

    def load_filter(self):
        global LAMBDA_MIN, LAMBDA_MAX
        print self.filter
        for f in FILTERS:
            if f['label'] == self.filter:
                
                LAMBDA_MIN = f['min']
                LAMBDA_MAX = f['max']
                self.my_logger.info('Load filter %s: lambda between %.1f and %.1f' % (f['label'],LAMBDA_MIN, LAMBDA_MAX))
                break
            

    def plot_spectrum(self,xlim=None,order=1,atmospheric_lines=True,hydrogen_only=False,emission_spectrum=False,nofit=False):
        xs = self.lambdas
        if xs is None : xs = np.arange(self.data.shape[0])
        #redshift = 0
        #if self.target is not None : redshift = self.target.redshift
        fig = plt.figure(figsize=[12,6])
        plt.plot(xs,self.data,'r-',lw=2,label='Order %d spectrum' % order)
        if self.lambdas is not None:
            plot_atomic_lines(plt.gca(),redshift=self.target.redshift,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only,fontsize=12)
        plt.grid(True)
        plt.xlim([LAMBDA_MIN,LAMBDA_MAX])
        plt.ylim(0.,np.max(self.data)*1.2)
        plt.xlabel('$\lambda$ [nm]')
        if self.lambdas is None: plt.xlabel('Pixels')
        if xlim is not None :
            plt.xlim(xlim)
            plt.ylim(0.,np.max(self.data[xlim[0]:xlim[1]])*1.2)
        if not nofit and self.lambdas is not None:
            lambda_shift = detect_lines(self.lambdas,self.data,redshift=self.target.redshift,emission_spectrum=self.target.emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only,ax=plt.gca(),verbose=False)
        plt.show()

    def calibrate(self,order=1,emission_spectrum=False,atmospheric_lines=True,hydrogen_only=False):
        self.my_logger.warning('Set redhisft and optsions for tests')
        #if self.target is not None :redshift = self.target.redshift
        #emission_spectrum = False
        atmospheric_lines = True
        self.my_logger.info('Calibrating order %d spectrum...' % order)
        self.lambdas, self.data = extract_spectrum(self.data,self.disperser,[0,self.data.shape[0]],self.target_pixcoords_rotated[0],self.target_pixcoords,order=order)
        # Cut spectra
        lambdas_indices = np.where(np.logical_and(self.lambdas > LAMBDA_MIN, self.lambdas < LAMBDA_MAX))[0]
        self.lambdas = self.lambdas[lambdas_indices]
        self.data = self.data[lambdas_indices]
        # Detect emission/absorption lines and calibrate pixel/lambda 
        D = DISTANCE2CCD-DISTANCE2CCD_ERR
        shift = 0
        shifts = []
        counts = 0
        D_step = DISTANCE2CCD_ERR / 4
        delta_pixels = lambdas_indices - int(self.target_pixcoords_rotated[0])
        while D < DISTANCE2CCD+4*DISTANCE2CCD_ERR and D > DISTANCE2CCD-4*DISTANCE2CCD_ERR and counts < 30 :
            self.disperser.D = D
            lambdas_test = self.disperser.grating_pixel_to_lambda(delta_pixels,self.target_pixcoords,order=order)
            lambda_shift = detect_lines(lambdas_test,self.data,redshift=self.target.redshift,emission_spectrum=self.target.emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only,ax=None,verbose=VERBOSE)
            shifts.append(lambda_shift)
            counts += 1
            if abs(lambda_shift)<0.1 :
                break
            elif lambda_shift > 2 :
                D_step = DISTANCE2CCD_ERR 
            elif 0.5 < lambda_shift < 2 :
                D_step = DISTANCE2CCD_ERR / 4
            elif 0 < lambda_shift < 0.5 :
                D_step = DISTANCE2CCD_ERR / 10
            elif 0 > lambda_shift > -0.5 :
                D_step = -DISTANCE2CCD_ERR / 20
            elif  lambda_shift < -0.5 :
                D_step = -DISTANCE2CCD_ERR / 6
            D += D_step
        shift = np.mean(lambdas_test - self.lambdas)
        self.lambdas = lambdas_test
        detect_lines(self.lambdas,self.data,redshift=self.target.redshift,emission_spectrum=self.target.emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only,ax=None,verbose=VERBOSE)
        if VERBOSE :
            print 'Wavelenght total shift: %.2fnm (after %d steps)' % (shift,len(shifts))
            print '\twith D = %.2f mm (DISTANCE2CCD = %.2f +/- %.2f mm, %.1f sigma shift)' % (D,DISTANCE2CCD,DISTANCE2CCD_ERR,(D-DISTANCE2CCD)/DISTANCE2CCD_ERR)
        #if DEBUG:
        self.plot_spectrum(xlim=None,order=order,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only,emission_spectrum=self.target.emission_spectrum,nofit=False)



def Spectractor(filename,outputdir,guess,target):
    """ Spectractor
    Main function to extract a spectrum from an image

    Args:
        filename (:obj:`str`): path to the image
        outputdir (:obj:`str`): path to the output directory
    """
    my_logger = set_logger(__name__)
    my_logger.info('Start SPECTRACTOR')
    # Load reduced image
    image = Image(filename,target=target)
    # Set output path
    ensure_dir(outputdir)
    # Cut the image
    
    # Find the exact target position in the raw cut image: several methods
    my_logger.info('Search for the target in the image...')
    target_pixcoords = image.find_target(guess)
    # Rotate the image: several methods
    image.turn_image()
    # Find the exact target position in the rotated image: several methods
    my_logger.info('Search for the target in the rotated image...')
    target_pixcoords_rotated = image.find_target(guess,rotated=True)
    # Subtract background and bad pixels
    spectrum = image.extract_spectrum_from_image()
    # Calibrate the spectrum
    spectrum.calibrate()
    # Subtract second order

    # Cut in wavelength

    # Load target and its spectrum

    # Run libratran ?

    # Save the spectra
    

if __name__ == "__main__":
    import commands, string, re, time, os
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-N", "--Narray", dest="Narray",
                      help="Size of map X,Y,Z (default=50,50,50).",default="50,50,50")
    parser.add_option("-d", "--debug", dest="debug",action="store_true",
                      help="Enter debug mode (more verbose and plots).",default=False)
    parser.add_option("-b", "--beta", dest="beta",
                      help="Beta values to use, can be an array (default=3.0).",default="3.0")
    parser.add_option("-v", "--verbose", dest="verbose",
                      help="Verbose.",default=0)
    (opts, args) = parser.parse_args()

    VERBOSE = int(opts.verbose)
    if opts.debug:
        DEBUG=True
        VERBOSE=1
        
        
    filename="../../CTIODataJune2017_reducedRed/data_05jun17/reduc_20170605_00.fits"
    filename="../ana_05jun17/OverScanRemove/trim_images/trim_20170605_007.fits"
    outputdir="test"
    guess = [745,643]
    target="3C273"

    #filename="../ana_05jun17/OverScanRemove/trim_images/trim_20170605_029.fits"
    #guess = [814, 585]
    #target = "PNG321.0+3.9"
    #filename="../ana_29may17/OverScanRemove/trim_images/trim_20170529_150.fits"
    #guess = [720, 670]
    #target = "HD185975"
    #filename="../ana_31may17/OverScanRemove/trim_images/trim_20170531_150.fits"
    #guess = [840, 530]
    #target = "HD205905"

    Spectractor(filename,outputdir,guess,target)
