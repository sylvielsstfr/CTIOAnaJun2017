import numpy as np
import os, sys
from scipy import ndimage
from scipy.optimize import curve_fit

# CCD characteristics
IMSIZE = 2048 # size of the image in pixel
PIXEL2MM = 24e-3 # pixel size in mm
PIXEL2ARCSEC = 0.401 # pixel size in arcsec
ARCSEC2RADIANS = np.pi/(180.*3600.) # conversion factor from arcsec to radians
DISTANCE2CCD = 55.56 # distance between hologram and CCD in mm
DISTANCE2CCD_ERR = 0.17 # uncertainty on distance between hologram and CCD in mm

# Making of the holograms
LAMBDA_CONSTRUCTOR = 639e-6 # constructor wavelength to make holograms in mm
GROOVES_PER_MM = 350 # approximate effective number of lines per millimeter of the hologram
PLATE_CENTER_SHIFT_X = -6. # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y = -8. # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_X_ERR = 2. # estimate uncertainty on plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y_ERR = 2. # estimate uncertainty on plate center shift on x in mm in filter frame

# H-alpha filter
HALPHA_CENTER = 655.9e-6 # center of the filter in mm
HALPHA_WIDTH = 6.4e-6 # width of the filter in mm

# Main emission/absorption lines in nm
HALPHA = {'lambda':656.3,'atmospheric':False,'label':'$H\\alpha$','pos':[0.007,0.02]}
HBETA = {'lambda': 486.3,'atmospheric':False,'label':'$H\\beta$','pos':[0.007,0.02]} 
HGAMMA = {'lambda':434.0,'atmospheric':False,'label':'$H\\gamma$','pos':[0.007,0.02]} 
HDELTA = {'lambda': 410.2,'atmospheric':False,'label':'$H\\delta$','pos':[0.007,0.02]}
CII1 =  {'lambda': 723.5,'atmospheric':False,'label':'$C_{II}$','pos':[0.007,0.02]}
CII2 =  {'lambda': 711.0,'atmospheric':False,'label':'$C_{II}$','pos':[0.007,0.02]}
CIV =  {'lambda': 706.0,'atmospheric':False,'label':'$C_{IV}$','pos':[-0.02,0.92]}
CII3 =  {'lambda': 679.0,'atmospheric':False,'label':'$C_{II}$','pos':[0.007,0.02]}
CIII1 =  {'lambda': 673.0,'atmospheric':False,'label':'$C_{III}$','pos':[-0.02,0.92]}
CIII2 =  {'lambda': 570.0,'atmospheric':False,'label':'$C_{III}$','pos':[0.007,0.02]}
HEI =  {'lambda': 587.5,'atmospheric':False,'label':'$He_{I}$','pos':[0.007,0.02]}
HEII =  {'lambda': 468.6,'atmospheric':False,'label':'$He_{II}$','pos':[0.007,0.02]}
O2 = {'lambda': 762.1,'atmospheric':True,'label':'$O_2$','pos':[0.007,0.02]} # http://onlinelibrary.wiley.com/doi/10.1029/98JD02799/pdf
H2O = {'lambda': 960,'atmospheric':True,'label':'$H_2 O$','pos':[0.007,0.02]}
LINES = [HALPHA,HBETA,HGAMMA,HDELTA,O2,H2O,CII1,CII2,CIV,CII3,CIII1,CIII2,HEI,HEII]


DATA_DIR = "../../common_tools/data/"


def plot_atomic_lines(ax,redshift=0,atmospheric_lines=True,hydrogen_only=False,color_atomic='g',color_atmospheric='b',fontsize=16):
    xlim = ax.get_xlim()
    for line in LINES:
        if not atmospheric_lines and line['atmospheric']: continue
        if hydrogen_only and '$H\\' not in line['label'] : continue
        color = color_atomic
        l = line['lambda']*(1+redshift)
        if line['atmospheric']: color = color_atmospheric
        ax.axvline(l,lw=2,color=color)
        ax.annotate(line['label'],xy=((l-xlim[0])/(xlim[1]-xlim[0])+line['pos'][0],line['pos'][1]),rotation=90,ha='left',va='bottom',xycoords='axes fraction',color=color,fontsize=fontsize)

        
def neutral_lines(x_center,y_center,theta_tilt):
    xs = np.linspace(0,IMSIZE,20)
    line1 = np.tan(theta_tilt*np.pi/180)*(xs-x_center)+y_center
    line2 = np.tan((theta_tilt+90)*np.pi/180)*(xs-x_center)+y_center
    return(xs,line1,line2)

def order01_positions(x_center,y_center,theta_tilt,verbose=True):
    # refraction angle between order 0 and order 1 at fabrication
    alpha = np.arcsin(GROOVES_PER_MM*LAMBDA_CONSTRUCTOR) 
    # distance between order 0 and order 1 in pixels
    AB = np.tan(alpha)*DISTANCE2CCD/PIXEL2MM
    # position of order 1 in pixels
    order1_position = [ 0.5*AB*np.cos(theta_tilt*np.pi/180)+x_center, 0.5*AB*np.sin(theta_tilt*np.pi/180)+y_center] 
    # position of order 0 in pixels
    order0_position = [ -0.5*AB*np.cos(theta_tilt*np.pi/180)+x_center, -0.5*AB*np.sin(theta_tilt*np.pi/180)+y_center]
    if verbose :
        print 'Order  0 position at x0 = %.1f and y0 = %.1f' % (order0_position[0],order0_position[1])
        print 'Order +1 position at x0 = %.1f and y0 = %.1f' % (order1_position[0],order1_position[1])
        print 'Distance between the orders: %.2f pixels (%.2f mm)' % (AB,AB*PIXEL2MM)
    return(order0_position,order1_position,AB)
    


def build_hologram(x_center,y_center,theta_tilt,lambda_plot=256000):
    # wavelength in nm, hologram porduced at 639nm
    order0_position,order1_position,AB = order01_positions(x_center,y_center,theta_tilt,verbose=False)
    # spherical wave centered in 0,0,0
    U = lambda x,y,z : np.exp(2j*np.pi*np.sqrt(x*x + y*y + z*z)*1e6/lambda_plot)/np.sqrt(x*x + y*y + z*z) 
    # superposition of two spherical sources centered in order 0 and order 1 positions
    plot_center = 0.5*IMSIZE*PIXEL2MM
    xA = [order0_position[0]*PIXEL2MM,order0_position[1]*PIXEL2MM]
    xB = [order1_position[0]*PIXEL2MM,order1_position[1]*PIXEL2MM]
    A = lambda x,y : U(x-xA[0],y-xA[1],-DISTANCE2CCD)+U(x-xB[0],y-xB[1],-DISTANCE2CCD)
    intensity = lambda x,y : np.abs(A(x,y))**2

    xholo = np.linspace(0,IMSIZE*PIXEL2MM,IMSIZE)
    yholo = np.linspace(0,IMSIZE*PIXEL2MM,IMSIZE)
    xxholo, yyholo = np.meshgrid(xholo,yholo)
    holo = intensity(xxholo,yyholo)
    return(holo)


def build_ronchi(x_center,y_center,theta_tilt,grooves=400):
    intensity = lambda x,y : 2*np.sin(2*np.pi*(x-x_center*PIXEL2MM)*0.5*grooves)**2

    xronchi = np.linspace(0,IMSIZE*PIXEL2MM,IMSIZE)
    yronchi = np.linspace(0,IMSIZE*PIXEL2MM,IMSIZE)
    xxronchi, yyronchi = np.meshgrid(xronchi,yronchi)
    ronchi = (intensity(xxronchi,yyronchi)).astype(int)
    rotated_ronchi=ndimage.interpolation.rotate(ronchi,theta_tilt)
    return(ronchi)





def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gauss(x,y,guess=[10,1000,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(gauss,x,y,p0=guess,bounds=bounds)
    return popt, pcov

def EmissionLineFit(spectra,left_edge=1200,right_edge=1600,guess=[10,1400,200],bounds=(-np.inf,np.inf)):
    xs = np.arange(left_edge,right_edge,1)
    right_spectrum = spectra[left_edge:right_edge]
    popt, pcov = fit_gauss(xs,right_spectrum,guess=guess)
    return(popt, pcov)



class Grating():
    def __init__(self,N,label=""):
        self.N = N # lines per mm
        self.N_err = 1
        self.label = label
        self.load_files()

    def load_files(self):
        filename = DATA_DIR+self.label+"/N.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.N = a[0]
            self.N_err = a[1]
        filename = DATA_DIR+self.label+"/hologram_center.txt"
        if os.path.isfile(filename):
            lines = [line.rstrip('\n') for line in open(filename)]
            #self.holo_center = map(float,lines[1].split(' ')[:2])
            self.theta_tilt = float(lines[1].split(' ')[2])
        else :
            self.theta_tilt = 0
            return
        self.plate_center = [0.5*IMSIZE,0.5*IMSIZE]
        print 'Grating plate center at x0 = %.1f and y0 = %.1f with average tilt of %.1f degrees' % (self.plate_center[0],self.plate_center[1],self.theta_tilt)

    def refraction_angle(self,deltaX):
        # refraction angle in radians
        return( np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD) )
        
    def refraction_angle_lambda(self,l,theta0=0,order=1):
        # refraction angle in radians with lambda in mm and
        # tetha0 incident angle in radians (wrt normal)
        return( np.arcsin(order*l*self.N + np.sin(theta0) ) )
        
    def grating_pixel_to_lambda(self,deltaX,theta0=0,order=1):
        # wavelength in nm
        theta = self.refraction_angle(deltaX)
        l = (np.sin(theta)-np.sin(theta0))/(order*self.N)
        return(l*1e6)

    def grating_resolution(self,deltaX):
        # wavelength resolution in nm per pixel
        theta = self.refraction_angle(deltaX)
        res = (np.cos(theta)**3*PIXEL2MM*1e6)/(self.N*DISTANCE2CCD)
        return(res)


        
class Hologram(Grating):

    def __init__(self,label,lambda_plot=256000):
        Grating.__init__(self,GROOVES_PER_MM,label=label)
        self.holo_center = None # center of symmetry of the hologram interferences in pixels
        self.plate_center = None # center of the hologram plate
        self.rotation_angle_map = None # interpolated rotation angle map of the hologram from data in degrees
        self.rotation_angle_map_x = None # x coordinates for the interpolated rotation angle map 
        self.rotation_angle_map_y = None # y coordinates for the interpolated rotation angle map 
        self.load_specs()
        self.hologram_shape = build_hologram(self.holo_center[0],self.holo_center[1],self.theta_tilt,lambda_plot=lambda_plot)

    def load_specs(self):
        filename = DATA_DIR+self.label+"/N.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.N = a[0]
            self.N_err = a[1]
        print 'N = %.2f +/- %.2f grooves/mm' % (self.N, self.N_err)
        filename = DATA_DIR+self.label+"/hologram_center.txt"
        if os.path.isfile(filename):
            lines = [line.rstrip('\n') for line in open(filename)]
            self.holo_center = map(float,lines[1].split(' ')[:2])
            self.theta_tilt = float(lines[1].split(' ')[2])
        else :
            self.holo_center = [0.5*IMSIZE,0.5*IMSIZE]
            self.theta_tilt = 0
            return
        print 'Hologram center at x0 = %.1f and y0 = %.1f with average tilt of %.1f degrees' % (self.holo_center[0],self.holo_center[1],self.theta_tilt)
        self.plate_center = [0.5*IMSIZE+PLATE_CENTER_SHIFT_X/PIXEL2MM,0.5*IMSIZE+PLATE_CENTER_SHIFT_Y/PIXEL2MM] 
        print 'Plate center at x0 = %.1f and y0 = %.1f with average tilt of %.1f degrees' % (self.plate_center[0],self.plate_center[1],self.theta_tilt)
        self.x_lines, self.line1, self.line2 = neutral_lines(self.holo_center[0],self.holo_center[1],self.theta_tilt)
        self.order0_position, self.order1_position, self.AB = order01_positions(self.holo_center[0],self.holo_center[1],self.theta_tilt)
        filename = DATA_DIR+self.label+"/rotation_angle_map.txt"
        if os.path.isfile(filename):
            self.rotation_angle_map = np.loadtxt(filename)
            self.rotation_angle_map_x = np.loadtxt(DATA_DIR+self.label+"/rotation_angle_map_x.txt")
            self.rotation_angle_map_y = np.loadtxt(DATA_DIR+self.label+"/rotation_angle_map_y.txt")
            

                
