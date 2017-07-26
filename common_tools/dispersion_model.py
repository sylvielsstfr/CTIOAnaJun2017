from holo_specs import *
from sympy import *
from sympy.geometry import *
from sympy.plotting import *
from sympy.plotting.pygletplot import PygletPlot as Plot
#from sympy.plotting.pyglet import Plot
import matplotlib.pyplot as plt
import matplotlib.colors

TELESCOPE_DIAMETER = 0.9 # CTIO telescope diameter in m
TELESCOPE_FOCAL = 12.6 # CTIO focal in m
d = DISTANCE2CCD*TELESCOPE_DIAMETER/TELESCOPE_FOCAL
theta_telescope = 2*np.arctan2(0.5*d,DISTANCE2CCD)

def plot_point(Point,fmt='o',color='k'):
    plt.plot([Point.x],[Point.y],fmt,color=color)

def plot_line(Line,fmt='-',color='k'):
    plt.plot([Line.p1[0],Line.p2[0]],[Line.p1[1],Line.p2[1]],fmt,color=color)

def plot_beam(Line1,Line2,fmt='-',color='k',alpha=1):
    plt.plot([Line1.p1[0],Line1.p2[0]],[Line1.p1[1],Line1.p2[1]],fmt,color=color)
    plt.plot([Line2.p1[0],Line2.p2[0]],[Line2.p1[1],Line2.p2[1]],fmt,color=color)




def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A=0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)    

def plot_refracted_beam(wavelength,g,delta_ra=0,color=None,plot=False):
    """ wavelength in mm """
    """ delta_ra in radians """

    O = Point(0,0)
    X0 = Point(DISTANCE2CCD*np.tan(delta_ra),0)
    A = Point(0,DISTANCE2CCD)
    A_left = Point(-d/2,DISTANCE2CCD)
    A_right = Point(d/2,DISTANCE2CCD)
    grating_line = Line(Point(5*d,DISTANCE2CCD),Point(-5*d,DISTANCE2CCD))
    ccd_line = Line(Point(5*d,0),Point(-5*d,0))

    zero_order_line = Line(X0,A)
    zero_order_line_left = Line(A_left,X0)
    zero_order_line_right = Line(A_right,X0)

    theta_lambda = g.refraction_angle_lambda(wavelength,theta0=delta_ra)
    C = Point(DISTANCE2CCD*np.tan(theta_lambda),0)
    refraction_line = Line(A,C)


    theta0_left = 0.5*theta_telescope + delta_ra
    theta_lambda_left = g.refraction_angle_lambda(wavelength,theta0=theta0_left)
    C_left = Point(A_left.x+DISTANCE2CCD*np.tan(theta_lambda_left),0)
    refraction_line_left = Line(A_left,C_left)
    zero_order_line_left_up  = Line(A_left,Point(A_left.x-d*np.tan(theta0_left),A_left.y+d))

    theta0_right = -0.5*theta_telescope + delta_ra
    theta_lambda_right = g.refraction_angle_lambda(wavelength,theta0=theta0_right)
    C_right = Point(A_right.x+DISTANCE2CCD*np.tan(theta_lambda_right),0)
    refraction_line_right = Line(A_right,C_right)
    zero_order_line_right_up  = Line(A_right,Point(A_right.x-d*np.tan(theta0_right),A_right.y+d))

    B = intersection(refraction_line_left,refraction_line_right)[0]
    C_middle = Point(0.5*(C_left.x+C_right.x),0)

    if plot :
        plot_line(grating_line,fmt='k--')
        plot_line(ccd_line,fmt='k-')
        plot_line(zero_order_line,fmt='k-')
        plot_line(zero_order_line_left,fmt='k-')
        plot_line(zero_order_line_right,fmt='k-')
        plot_line(zero_order_line_left_up,fmt='k-')
        plot_line(zero_order_line_right_up,fmt='k-')

        if color is None :
            color = wavelength_to_rgb(wavelength*1e6, gamma=0.8)
        plot_line(refraction_line,fmt='-',color=color)
        plot_line(refraction_line_left,fmt='-',color=color)
        plot_line(refraction_line_right,fmt='-',color=color)

        plot_point(X0)
        plot_point(O)
        plot_point(A)
        plot_point(B,color=color)
        plot_point(C,color=color)
        plot_point(C_middle,'*',color=color)
    return(C_middle,C,C_left,C_right,X0)

def input_flux(wavelength):
    return(np.exp(-0.5*(wavelength-HALPHA_CENTER)**2/(0.5*HALPHA_WIDTH)**2))

#def input_flux(wavelength):
#    if wavelength > HALPHA_CENTER - 0.5*HALPHA_WIDTH and  wavelength < HALPHA_CENTER + 0.5*HALPHA_WIDTH:
#        return 1
#    else :
#        return 0




fig = plt.figure(1,figsize=(8,6))
ax1 = fig.add_subplot(111)

Cs = []
N = 300
g = Grating(N)
print 'Grating %d grooves/mm' % N



delta_ra = -200*PIXEL2ARCSEC*ARCSEC2RADIANS
wavelengths = 1e-6*np.arange(400,1000,100)
wavelengths = np.arange(HALPHA_CENTER-3*HALPHA_WIDTH,HALPHA_CENTER+3*HALPHA_WIDTH,0.5e-6)
for w in wavelengths:
    Cs.append(plot_refracted_beam(w,g,delta_ra,plot=False))

C_lambda = plot_refracted_beam(HALPHA_CENTER,g,delta_ra,color='k',plot=True)
x_shift = C_lambda[0].x-C_lambda[1].x
lambda_shift = ( g.grating_pixel_to_lambda(float(C_lambda[0].x-C_lambda[4].x)/PIXEL2MM,theta0=delta_ra) - g.grating_pixel_to_lambda(float(C_lambda[1].x-C_lambda[4].x)/PIXEL2MM,theta0=delta_ra) )
print 'With simple model:'
print 'Distance between beam center and true line direction at l = %.3g nm:' % HALPHA_CENTER
print '\t %.3f mm ie %.1f pixels ie %.1f nm' % (x_shift,x_shift/PIXEL2MM,lambda_shift)
ax1.set_xlabel('X [mm]')
ax1.set_ylabel('Y [mm]')




xs = np.arange(-5*d,5*d,PIXEL2MM)
xs = np.arange(0,IMSIZE)
output_flux = np.zeros_like(xs,dtype=np.float64)

for ic,c in enumerate(Cs):
    C_middle = c[0]
    C = c[1]
    C_left = c[2]
    C_right = c[3]
    beam_width = np.abs(C_left.x-C_right.x)
    flux = input_flux(wavelengths[ic])*d/beam_width
    for ix,x in enumerate(xs) :
        xpos = (x-IMSIZE/2)*PIXEL2MM
        if xpos<C_right.x : continue
        if xpos>C_left.x : break
        output_flux[ix] += flux

output_flux *= 0.5*DISTANCE2CCD/np.max(output_flux)
ixmax = np.where(output_flux==np.max(output_flux))[0][0]
popt, pcov = EmissionLineFit(output_flux,left_edge=0,right_edge=IMSIZE,guess=[0.5*DISTANCE2CCD,ixmax,1])
xmax = (popt[1]-IMSIZE/2)*PIXEL2MM        

x_shift = xmax-C_lambda[1].x
lambda_shift = ( g.grating_pixel_to_lambda(float(xmax-C_lambda[4].x)/PIXEL2MM,theta0=delta_ra) - g.grating_pixel_to_lambda(float(C_lambda[1].x-C_lambda[4].x)/PIXEL2MM,theta0=delta_ra) )

print 'With gaussian fitting:'
print 'Distance between HALPHA_CENTER position and output flux maximum:'
print '\t %.3f mm ie %.1f pixels ie %.1f nm' % (x_shift,x_shift/PIXEL2MM,lambda_shift) 

#fig = plt.figure(1,figsize=(8,6))
plt.plot((xs-IMSIZE/2)*PIXEL2MM,output_flux)
plt.plot((xs-IMSIZE/2)*PIXEL2MM,gauss(xs,*popt),'b-')
plt.plot([C_lambda[1].x,C_lambda[1].x],[0,0.5*DISTANCE2CCD])
plt.plot([xmax,xmax],[0,0.5*DISTANCE2CCD],'g-')
ax1.grid(True)

ax2 = ax1.twiny()
ax1Ticks = ax1.get_xticks()   
ax2Ticks = ax1Ticks
ax2.set_xticks(ax2Ticks)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(g.grating_pixel_to_lambda(ax2Ticks/PIXEL2MM))

ax2.set_xlabel('Wavelengths [nm]')

plt.show()
