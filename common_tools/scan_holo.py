from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hessian_matrix

from holo_specs import *


def dist2line(x,theta,x_line,y_line):
    t = np.interp(x,x_line,y_line)
    return(np.abs(t-theta))

def dist2lines(x,theta,lines):
    dist = 0
    for l in lines :
        x_line = l[0]
        y_line = l[1]
        dist += dist2line(x,theta,x_line,y_line)
    return(dist)
def RotationLines(all_theta,central_positions):
    proj_x_lines = []
    for i in range(5):
        x = []
        y = []
        for it,t in enumerate(all_theta):
            if it / 5 == i : 
                x.append(central_positions[it][1])
                y.append(t)
        proj_x_lines.append([x,y])
    proj_y_lines = []
    for i in reversed(range(5)):
        x = []
        y = []
        for it,t in enumerate(all_theta):
            if it % 5 == i : 
                x.append(central_positions[it][0])
                y.append(t)
        proj_y_lines.append([x,y])
    return(proj_x_lines,proj_y_lines)



def FindHoloCenter(all_angles,central_positions,proj_x_lines,proj_y_lines):
    """
    FindHoloCenter
    =============
    
    input:
    ------
    all_angles:
    
    
    output:
    ------
    x_center
    y_center
    theta_tilt
    
    """
    theta_max = np.max(all_angles)
    theta_min = np.min(all_angles)
    x_min = np.min(central_positions.T[1])
    x_max = np.max(central_positions.T[1])
    y_min = np.min(central_positions.T[0])
    y_max = np.max(central_positions.T[0])
    
    # Minimize in the x direction
    bounds=[[x_min,x_max],[theta_min,theta_max]]
    fun = lambda point : dist2lines(point[0],point[1],proj_x_lines)
    res = optimize.minimize(fun,(800,-1), method='SLSQP',bounds=bounds)
    x_center = res.x[0]
    theta_tilt = res.x[1]
    # Minimize in the y direction
    bounds=[[y_min,y_max],[theta_min,theta_max]]
    fun = lambda point : dist2lines(point[0],point[1],proj_y_lines)
    res = optimize.minimize(fun,(800,-1), method='SLSQP',bounds=bounds)
    y_center = res.x[0]
    theta_tilt = (theta_tilt+res.x[1])/2.
    print 'Hologram center at x0 = %.1f and y0 = %.1f with average tilt of %.1f degrees' % (x_center,y_center,theta_tilt)
    return(x_center,y_center,theta_tilt)




def hessian_analysis(z):
    # le parametre sigma permet de lisser le hessien
    Hxx, Hxy, Hyy = hessian_matrix(z, sigma=0.1, order = 'xy')
    dets = np.zeros_like(Hxx)
    for i in range(dets.shape[0]):
        for j in range(dets.shape[1]):
            h = np.array([[Hxx[i,j],Hxy[i,j]],[Hxy[i,j],Hyy[i,j]]])
            dets[i,j] = np.linalg.det(h)
    return(Hxx,Hyy,Hxy,dets)

def scatter_interpol(central_positions,values,margin=10):
    y, x = np.array(central_positions).T
    interp = interpolate.interp2d(x, y, values, kind='cubic')

    step = 1
    xx = np.arange(np.min(x)-margin,np.max(x)+margin,step)
    yy = np.arange(np.min(y)-margin,np.max(y)+margin,step)
    z = interp(xx,yy)
    return(xx,yy,z,interp)

def surface_gradient(z,margin=10):
    gx, gy = np.gradient(z)
    grad = np.sqrt(gy**2+gx**2)
    grad = grad[margin:-margin,margin:-margin]
    return(grad)

def saddle_point(central_positions,all_theta,margin=10):
    xx,yy,z,interp = scatter_interpol(central_positions,all_theta,margin=margin)
    grad = surface_gradient(z)
    null_grad = np.where(np.isclose(grad,np.min(grad)))
    x_center = xx[null_grad[1]]
    y_center = yy[null_grad[0]]
    theta_tilt = interp(x_center,y_center)
    min_grad = np.min(grad[margin:-margin,margin:-margin])
    return(x_center,y_center,null_grad,theta_tilt,min_grad)





