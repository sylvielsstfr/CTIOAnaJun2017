import numpy as np
import os, sys
from scipy import ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# CCD characteristics
IMSIZE = 2048 # size of the image in pixel
PIXEL2MM = 24e-3 # pixel size in mm
PIXEL2ARCSEC = 0.401 # pixel size in arcsec
ARCSEC2RADIANS = np.pi/(180.*3600.) # conversion factor from arcsec to radians
#DISTANCE2CCD = 55.56 # distance between hologram and CCD in mm
#DISTANCE2CCD_ERR = 0.17 # uncertainty on distance between hologram and CCD in mm
DISTANCE2CCD = 55.45 # distance between hologram and CCD in mm
DISTANCE2CCD_ERR = 0.19 # uncertainty on distance between hologram and CCD in mm

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





def CalibrateDistance2CCD_OneOrder(thecorrspectra,thex0,all_filt,left_edge=1200,right_edge=1600,guess=[10,1400,200],bounds=(-np.inf,np.inf)):

    f, axarr = plt.subplots(2,2,figsize=(25,10))
    count = 0
    D_range = 1 # in mm
    print 'Present distance to CCD : %.2f mm (to update if necessary)' % DISTANCE2CCD
    print '-------------------------------'    
    distances = []
    distances_err = []
    for index in range(len(thecorrspectra)):
        if "Ron400" not in all_filt[index] and "Thor300" not in all_filt[index] : continue
        if "Thor300" in all_filt[index] : N_theo = 300
        if "Ron400" in all_filt[index] : N_theo = 400  
        # dispersion axis analysis
        spectra = thecorrspectra[index]
        popt, pcov = EmissionLineFit(spectra,left_edge,right_edge,guess,bounds)
        x0 = popt[1]
        x0_err = np.sqrt(pcov[1][1]) 
        deltaX = np.abs(x0 - thex0[index])
        theta0 = (thex0[index] - IMSIZE/2)*PIXEL2ARCSEC*ARCSEC2RADIANS
        deltaX0 = np.tan(theta0)*DISTANCE2CCD/PIXEL2MM
        print all_filt[index]
        print 'Position of the H-alpha emission line : %.2f +/- %.2f pixels (%.2f percent)' % (deltaX,x0_err,x0_err/deltaX*100)
        Ds = np.linspace(DISTANCE2CCD-D_range,DISTANCE2CCD+D_range,100)
        Ns = []
        diffs = []
        optimal_D = DISTANCE2CCD
        optimal_D_inf = DISTANCE2CCD
        optimal_D_sup = DISTANCE2CCD
        test = 1e20
        test_sup = 1e20
        test_inf = 1e20
        for D in Ds :
            theta = np.arctan2(deltaX*PIXEL2MM,D)
            N = np.sin(theta)/(HALPHA_CENTER)
            Ns.append( N )
            diff = np.abs(N-N_theo)
            diff_sup = np.abs(N-N_theo+1)
            diff_inf = np.abs(N-N_theo-1)
            diffs.append(diff)
            if diff < test :
                test = diff
                optimal_D = D
            if diff_sup < test_sup :
                test_sup = diff_sup
                optimal_D_sup = D
            if diff_inf < test_inf :
                test_inf = diff_inf
                optimal_D_inf = D
        optimal_D_err  = 0.5*(optimal_D_sup-optimal_D_inf)
        distances.append(optimal_D)
        distances_err.append(optimal_D_err)
        print 'Deduced distance to CCD with %s : %.2f +/- %.2f mm (%.2f percent)' % (all_filt[index],optimal_D,optimal_D_err,100*optimal_D_err/optimal_D)
        # plot Ns vs Ds
        axarr[0,count].plot(Ds,Ns,'b-',lw=2)
        axarr[0,count].plot([np.min(Ds),np.max(Ds)],[N_theo,N_theo],'r-',lw=2)
        axarr[0,count].plot([optimal_D,optimal_D],[np.min(Ns),np.max(Ns)],'r-',lw=2)
        axarr[0,count].plot([np.min(Ds),np.max(Ds)],[N_theo+1,N_theo+1],'k--',lw=2)
        axarr[0,count].plot([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],'k--',lw=2)
        axarr[0,count].plot([optimal_D_inf,optimal_D_inf],[np.min(Ns),np.max(Ns)],'k--',lw=2)
        axarr[0,count].plot([optimal_D_sup,optimal_D_sup],[np.min(Ns),np.max(Ns)],'k--',lw=2)
        axarr[0,count].fill_between([optimal_D_inf,optimal_D_sup],[np.min(Ns),np.min(Ns)],[np.max(Ns),np.max(Ns)],color='red',alpha=0.2)
        axarr[0,count].plot([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],'k--',lw=2)
        axarr[0,count].fill_between([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],[N_theo+1,N_theo+1],color='red',alpha=0.2)
        axarr[0,count].scatter([optimal_D],[N_theo],s=200,color='r')
        axarr[0,count].set_xlim([np.min(Ds),np.max(Ds)])
        axarr[0,count].set_ylim([np.min(Ns),np.max(Ns)])
        axarr[0,count].grid(True)
        axarr[0,count].text(DISTANCE2CCD+0.6*D_range,N_theo+2.3,all_filt[index],verticalalignment='top', horizontalalignment='center',color='blue',fontweight='bold', fontsize=20)
        axarr[0,count].set_xlabel('Distance to CCD [mm]',fontsize=16)
        axarr[0,count].set_ylabel('Grooves per mm',fontsize=16)
        # plot diffs vs Ds
        axarr[1,count].plot(Ds,diffs,'b-',lw=2)
        axarr[1,count].plot([optimal_D,optimal_D],[np.min(diffs),np.max(diffs)],'r-',lw=2)
        axarr[1,count].plot([np.min(Ds),np.max(Ds)],[1,1],'k--',lw=2)
        #axarr[1,count].scatter([N_theo],[optimal_D],s=200,color='r')
        axarr[1,count].set_xlim([np.min(Ds),np.max(Ds)])
        axarr[1,count].set_ylim([np.min(diffs),np.max(diffs)])
        axarr[1,count].grid(True)
        axarr[1,count].set_xlabel('Distance to CCD [mm]',fontsize=16)
        axarr[1,count].set_ylabel('Difference to $N_{\mathrm{theo}}$ [grooves per mm]',fontsize=16)
        axarr[1,count].plot([optimal_D_inf,optimal_D_inf],[np.min(diffs),np.max(diffs)],'k--',lw=2)
        axarr[1,count].plot([optimal_D_sup,optimal_D_sup],[np.min(diffs),np.max(diffs)],'k--',lw=2)
        axarr[1,count].fill_between([optimal_D_inf,optimal_D_sup],[np.min(diffs),np.min(diffs)],[np.max(diffs),np.max(diffs)],color='red',alpha=0.2)
        axarr[1,count].text(DISTANCE2CCD+0.6*D_range,2.5,all_filt[index],verticalalignment='top', horizontalalignment='center',color='blue',fontweight='bold', fontsize=20)
        count += 1
        print '-------------------------------'  
    distances_mean = np.mean(distances)
    distances_mean_err = np.sqrt(np.mean(np.array(distances_err)**2))
    print 'Average distance to CCD : %.2f +/- %.2f mm (%.2f percent)' % (distances_mean,distances_mean_err,100*distances_mean_err/distances_mean)

    plt.show()
    return(distances_mean,distances_mean_err)


def CalibrateDistance2CCD_TwoOrder(thecorrspectra,all_filt,leftorder_edges=[100,400],rightorder_edges=[1200,1600],guess=[[10,200,100],[10,1400,200]],bounds=(-np.inf,np.inf)):

    f, axarr = plt.subplots(2,2,figsize=(25,10))
    count = 0
    D_range = 1 # in mm
    print 'Present distance to CCD : %.2f mm (to update if necessary)' % DISTANCE2CCD
    print '-------------------------------'    
    distances = []
    distances_err = []
    for index in range(len(thecorrspectra)):
        if "Ron400" not in all_filt[index] and "Thor300" not in all_filt[index] : continue
        if "Thor300" in all_filt[index] : N_theo = 300
        if "Ron400" in all_filt[index] : N_theo = 400  
        # dispersion axis analysis
        spectra = thecorrspectra[index]
        popt, pcov = EmissionLineFit(spectra,leftorder_edges[0],leftorder_edges[1],guess[0],bounds)
        x0_left = popt[1]
        x0_left_err = np.sqrt(pcov[1][1]) 
        popt, pcov = EmissionLineFit(spectra,rightorder_edges[0],rightorder_edges[1],guess[1],bounds)
        x0_right = popt[1]
        x0_right_err = np.sqrt(pcov[1][1]) 
        deltaX = 0.5*np.abs(x0_right - x0_left)
        x0_err = 0.5*np.sqrt(x0_right_err**2+x0_left_err**2)
        print all_filt[index]
        print 'Position of the H-alpha emission line : %.2f +/- %.2f pixels (%.2f percent)' % (deltaX,x0_err,x0_err/deltaX*100)
        Ds = np.linspace(DISTANCE2CCD-D_range,DISTANCE2CCD+D_range,100)
        Ns = []
        diffs = []
        optimal_D = DISTANCE2CCD
        optimal_D_inf = DISTANCE2CCD
        optimal_D_sup = DISTANCE2CCD
        test = 1e20
        test_sup = 1e20
        test_inf = 1e20
        for D in Ds :
            theta = np.arctan2(deltaX*PIXEL2MM,D)
            N = np.sin(theta)/HALPHA_CENTER
            Ns.append( N )
            diff = np.abs(N-N_theo)
            diff_sup = np.abs(N-N_theo+1)
            diff_inf = np.abs(N-N_theo-1)
            diffs.append(diff)
            if diff < test :
                test = diff
                optimal_D = D
            if diff_sup < test_sup :
                test_sup = diff_sup
                optimal_D_sup = D
            if diff_inf < test_inf :
                test_inf = diff_inf
                optimal_D_inf = D
        optimal_D_err  = 0.5*(optimal_D_sup-optimal_D_inf)
        distances.append(optimal_D)
        distances_err.append(optimal_D_err)
        print 'Deduced distance to CCD with %s : %.2f +/- %.2f mm (%.2f percent)' % (all_filt[index],optimal_D,optimal_D_err,100*optimal_D_err/optimal_D)
        # plot Ns vs Ds
        axarr[0,count].plot(Ds,Ns,'b-',lw=2)
        axarr[0,count].plot([np.min(Ds),np.max(Ds)],[N_theo,N_theo],'r-',lw=2)
        axarr[0,count].plot([optimal_D,optimal_D],[np.min(Ns),np.max(Ns)],'r-',lw=2)
        axarr[0,count].plot([np.min(Ds),np.max(Ds)],[N_theo+1,N_theo+1],'k--',lw=2)
        axarr[0,count].plot([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],'k--',lw=2)
        axarr[0,count].plot([optimal_D_inf,optimal_D_inf],[np.min(Ns),np.max(Ns)],'k--',lw=2)
        axarr[0,count].plot([optimal_D_sup,optimal_D_sup],[np.min(Ns),np.max(Ns)],'k--',lw=2)
        axarr[0,count].fill_between([optimal_D_inf,optimal_D_sup],[np.min(Ns),np.min(Ns)],[np.max(Ns),np.max(Ns)],color='red',alpha=0.2)
        axarr[0,count].plot([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],'k--',lw=2)
        axarr[0,count].fill_between([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],[N_theo+1,N_theo+1],color='red',alpha=0.2)
        axarr[0,count].scatter([optimal_D],[N_theo],s=200,color='r')
        axarr[0,count].set_xlim([np.min(Ds),np.max(Ds)])
        axarr[0,count].set_ylim([np.min(Ns),np.max(Ns)])
        axarr[0,count].grid(True)
        axarr[0,count].text(DISTANCE2CCD+0.6*D_range,N_theo+2.3,all_filt[index],verticalalignment='top', horizontalalignment='center',color='blue',fontweight='bold', fontsize=20)
        axarr[0,count].set_xlabel('Distance to CCD [mm]',fontsize=16)
        axarr[0,count].set_ylabel('Grooves per mm',fontsize=16)
        # plot diffs vs Ds
        axarr[1,count].plot(Ds,diffs,'b-',lw=2)
        axarr[1,count].plot([optimal_D,optimal_D],[np.min(diffs),np.max(diffs)],'r-',lw=2)
        axarr[1,count].plot([np.min(Ds),np.max(Ds)],[1,1],'k--',lw=2)
        #axarr[1,count].scatter([N_theo],[optimal_D],s=200,color='r')
        axarr[1,count].set_xlim([np.min(Ds),np.max(Ds)])
        axarr[1,count].set_ylim([np.min(diffs),np.max(diffs)])
        axarr[1,count].grid(True)
        axarr[1,count].set_xlabel('Distance to CCD [mm]',fontsize=16)
        axarr[1,count].set_ylabel('Difference to $N_{\mathrm{theo}}$ [grooves per mm]',fontsize=16)
        axarr[1,count].plot([optimal_D_inf,optimal_D_inf],[np.min(diffs),np.max(diffs)],'k--',lw=2)
        axarr[1,count].plot([optimal_D_sup,optimal_D_sup],[np.min(diffs),np.max(diffs)],'k--',lw=2)
        axarr[1,count].fill_between([optimal_D_inf,optimal_D_sup],[np.min(diffs),np.min(diffs)],[np.max(diffs),np.max(diffs)],color='red',alpha=0.2)
        axarr[1,count].text(DISTANCE2CCD+0.6*D_range,2,all_filt[index],verticalalignment='top', horizontalalignment='center',color='blue',fontweight='bold', fontsize=20)
        count += 1
        print '-------------------------------'  
    distances_mean = np.mean(distances)
    distances_mean_err = np.sqrt(np.mean(np.array(distances_err)**2))
    print 'Average distance to CCD : %.2f +/- %.2f mm (%.2f percent)' % (distances_mean,distances_mean_err,100*distances_mean_err/distances_mean)

    plt.show()
    return(distances_mean,distances_mean_err)


def GratingResolution_OneOrder(thecorrspectra,thex0,all_images,all_filt,left_edge=1200,right_edge=1600,guess=[10,1400,200],bounds=(-np.inf,np.inf)):
    print 'H-alpha filter center: %.1fnm ' % (HALPHA_CENTER*1e6)
    print 'H-alpha filter width: %.1fnm\n' % (HALPHA_WIDTH*1e6)

    Ns = []
    N_errs = []
    for index in range(len(thecorrspectra)):
        print all_filt[index]
        # dispersion axis analysis
        spectra = thecorrspectra[index]
        popt, pcov = EmissionLineFit(spectra,left_edge,right_edge,guess=guess,bounds=bounds)
        x0 = popt[1]
        # compute N
        deltaX = np.abs(x0 - thex0[index])
        theta = np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD)
        N = np.sin(theta)/HALPHA_CENTER
        Ns.append(N)
        # compute N uncertainty 
        theta = np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD+DISTANCE2CCD_ERR)
        N_up = np.sin(theta)/HALPHA_CENTER
        theta = np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD-DISTANCE2CCD_ERR)
        N_low = np.sin(theta)/HALPHA_CENTER
        N_err = 0.5*np.abs(N_up-N_low)
        N_errs.append(N_err)
        # look at finesse
        fwhm_line = np.abs(popt[2])*2.355
        g = Grating(N,label=all_filt[index])
        res = g.grating_resolution(deltaX)
        finesse = HALPHA_CENTER/(res*fwhm_line*1e-6-HALPHA_WIDTH)
        # transverse profile analysis
        yprofile=np.copy(all_images[index])[:,int(x0)]
        popt2, pcov2 = EmissionLineFit(yprofile,0,len(yprofile),[np.max(yprofile),0.5*len(yprofile),10])
        fwhm_profile = np.abs(popt2[2])*2.355
        finesse_profile = HALPHA_CENTER*1e6/(res*fwhm_profile)
    
        print 'N=%.1f +/- %.1f lines/mm\t H-alpha FWHM=%.1fpix with res=%.3fnm/pix : FWHM=%.1fnm\t ie finesse=%.1f' % (N,N_err,fwhm_line,res,res*fwhm_line,finesse)
        print 'Transverse profile FWHM=%.1fpix ' % (fwhm_profile)
        print '-------------------------------'
    return(Ns,N_errs)



def GratingResolution_TwoOrder(thecorrspectra,all_images,all_filt,leftorder_edges=[100,400],rightorder_edges=[1200,1600],guess=[[10,200,100],[10,1400,200]],bounds=(-np.inf,np.inf)):
    print 'H-alpha filter center: %.1fnm ' % (HALPHA_CENTER*1e6)
    print 'H-alpha filter width: %.1fnm\n' % (HALPHA_WIDTH*1e6)

    Ns = []
    N_errs = []
    for index in range(len(thecorrspectra)):
        # dispersion axis analysis
        spectra = thecorrspectra[index]
        popt, pcov = EmissionLineFit(spectra,leftorder_edges[0],leftorder_edges[1],guess[0],bounds)
        x0_left = popt[1]
        popt2, pcov = EmissionLineFit(spectra,rightorder_edges[0],rightorder_edges[1],guess[1],bounds)
        x0_right = popt2[1]
        deltaX = 0.5*np.abs(x0_left-x0_right)
        # compute N
        theta = np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD)
        N = np.sin(theta)/HALPHA_CENTER
        Ns.append(N)
        # compute N uncertainty 
        theta = np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD+DISTANCE2CCD_ERR)
        N_up = np.sin(theta)/HALPHA_CENTER
        theta = np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD-DISTANCE2CCD_ERR)
        N_low = np.sin(theta)/HALPHA_CENTER
        N_err = 0.5*np.abs(N_up-N_low)
        N_errs.append(N_err)
        # look at finesse
        g = Grating(N,label=all_filt[index])
        res = g.grating_resolution(deltaX)
        # right
        fwhm_line_right = np.abs(popt2[2])*2.355
        finesse_right = HALPHA_CENTER/(res*fwhm_line_right*1e-6-HALPHA_WIDTH)
        # left
        fwhm_line_left = np.abs(popt[2])*2.355
        finesse_left = HALPHA_CENTER/(res*fwhm_line_left*1e-6-HALPHA_WIDTH)
        # transverse profile analysis
        # right
        yprofile=np.copy(all_images[index])[:,int(x0_right)]
        popt2, pcov2 = EmissionLineFit(yprofile,0,len(yprofile),[np.max(yprofile),0.5*len(yprofile),10])
        fwhm_profile_right = np.abs(popt2[2])*2.355
        finesse_profile_right = HALPHA_CENTER*1e6/(res*fwhm_profile_right)
        # left
        yprofile=np.copy(all_images[index])[:,int(x0_left)]
        popt2, pcov2 = EmissionLineFit(yprofile,0,len(yprofile),[np.max(yprofile),0.5*len(yprofile),10])
        fwhm_profile_left = np.abs(popt2[2])*2.355
        finesse_profile_left = HALPHA_CENTER*1e6/(res*fwhm_profile_left)
    
        print all_filt[index]
        print 'N=%.1f +/- %.1f lines/mm' % (N,N_err)
        print 'Right order: H-alpha FWHM=%.1fpix with res=%.2gnm/pix : FWHM=%.1fnm\t ie finesse=%.1f' % (fwhm_line_right,res,res*fwhm_line_right,finesse_right)
        print 'Left  order: H-alpha FWHM=%.1fpix with res=%.2gnm/pix : FWHM=%.1fnm\t ie finesse=%.1f' % (fwhm_line_left,res,res*fwhm_line_left,finesse_left)
        print 'Transverse profile FWHM :  %.1fpix (right)  %.1fpix (left)' % (fwhm_profile_right, fwhm_profile_left)
        print '-------------------------------'
    return(Ns,N_errs)


def CalibrateSpectra(spectra,redshift,thex0,all_titles,object_name,all_filt,xlim=(1000,1800),target=None):
    """
    CalibrateSpectra show the right part of spectrum with identified lines
    =====================
    """
    NBSPEC=len(spectra)
    
    if target is not None :
        target.load_spectra()
    
    left_cut = xlim[0]
    right_cut = xlim[1]

    f, axarr = plt.subplots(NBSPEC,1,figsize=(20,7*NBSPEC))
    for index in np.arange(0,NBSPEC):

        spec = spectra[index][left_cut:right_cut]
        ######## convert pixels to wavelengths #########
        print all_filt[index]
        holo = Hologram(all_filt[index])
        #holo = Grating(Ns[index])
        print '-----------------------------------------------------'
        pixels = np.arange(left_cut,right_cut,1)-int(thex0[index])
        lambdas = holo.grating_pixel_to_lambda(pixels)
        
        axarr[index].plot(lambdas,spec,'r-',lw=2,label='Order +1 QSO spectrum')
    
        plot_atomic_lines(axarr[index],redshift=redshift,atmospheric_lines=False)
        if target is not None :
            for isp,sp in enumerate(target.spectra):
                if isp==0 or isp==2 : continue
                axarr[index].plot(target.wavelengths[isp],0.3*sp*spec.max()/np.max(sp),label='NED spectrum %d' % isp,lw=2)

        ######## set plot
        axarr[index].set_title(all_titles[index])
        axarr[index].annotate(all_filt[index],xy=(0.95,0.9),xytext=(0.95,0.9),verticalalignment='top', horizontalalignment='right',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
        axarr[index].legend(fontsize=16,loc='best')
        axarr[index].set_xlabel('Wavelength [nm]', fontsize=16)
        axarr[index].grid(True)
        axarr[index].set_ylim(0.,spec.max()*1.2)
        #axarr[index].set_xlim(xlim)
    plt.show()
