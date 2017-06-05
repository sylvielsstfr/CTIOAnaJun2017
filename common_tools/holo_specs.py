import numpy as np
import os, sys

# CCD characteristics
IMSIZE = 2048 # size of the image in pixel
PIXEL2MM = 24e-3 # pixel size in mm
DISTANCE2CCD = 58 # distance between hologram and CCD in mm

# Making of the holograms
LAMBDA_CONSTRUCTOR = 639e-6 # constructor wavelength to make holograms in mm
LINES_PER_MM = 355 # approximate effective number of lines per millimeter of the hologram
PLATE_CENTER_SHIFT_X = -6. # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y = -8. # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_X_ERR = 2. # estimate uncertainty on plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y_ERR = 2. # estimate uncertainty on plate center shift on x in mm in filter frame

DATA_DIR = "../../common_tools/data/"


def neutral_lines(x_center,y_center,theta_tilt):
    xs = np.linspace(0,IMSIZE,20)
    line1 = np.tan(theta_tilt*np.pi/180)*(xs-x_center)+y_center
    line2 = np.tan((theta_tilt+90)*np.pi/180)*(xs-x_center)+y_center
    return(xs,line1,line2)

def order01_positions(x_center,y_center,theta_tilt,verbose=True):
    # refraction angle between order 0 and order 1 at fabrication
    alpha = np.arcsin(LINES_PER_MM*LAMBDA_CONSTRUCTOR) 
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



class Hologram():

    def __init__(self,label,lambda_plot=256000):
        self.label = label
        self.lines_per_mm = LINES_PER_MM
        self.holo_center = None # center of symmetry of the hologram interferences in pixels
        self.plate_center = None # center of the hologram plate
        self.rotation_angle_map = None # interpolated rotation angle map of the hologram from data in degrees
        self.rotation_angle_map_x = None # x coordinates for the interpolated rotation angle map 
        self.rotation_angle_map_y = None # y coordinates for the interpolated rotation angle map 
        self.load_specs()
        self.hologram_shape = build_hologram(self.holo_center[0],self.holo_center[1],self.theta_tilt,lambda_plot=lambda_plot)

    def load_specs(self):
        filename = DATA_DIR+self.label+"/lines_per_mm.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            print a
            self.lines_per_mm = a
        filename = DATA_DIR+self.label+"/hologram_center.txt"
        if os.path.isfile(filename):
            lines = [line.rstrip('\n') for line in open(filename)]
            self.holo_center = map(float,lines[1].split(' ')[:2])
            self.theta_tilt = float(lines[1].split(' ')[2])
        else :
            self.holo_center = [0.5*IMSIZE,0.5*IMSIZE]
            self.theta_tilt = 0
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
            

                
