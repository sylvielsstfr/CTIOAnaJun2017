from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
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
                x.append(central_positions[it][0])
                y.append(t)
        proj_x_lines.append([x,y])
    proj_y_lines = []
    for i in reversed(range(5)):
        x = []
        y = []
        for it,t in enumerate(all_theta):
            if it % 5 == i : 
                x.append(central_positions[it][1])
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
    x_min = np.min(central_positions.T[0])
    x_max = np.max(central_positions.T[0])
    y_min = np.min(central_positions.T[1])
    y_max = np.max(central_positions.T[1])
    
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


def plot_rotation_lines(proj_x_lines,proj_y_lines,holo_name='',center=None):
    f, axarr = plt.subplots(1,2,figsize=(16,6))
    for i in range(5):
        x, y = proj_x_lines[i]
        axarr[0].plot(x,y)
    axarr[0].set_xlabel('Pixel x')
    axarr[0].set_ylabel('Rotation angle')
    axarr[0].annotate(holo_name,xy=(0.95,0.05),xytext=(0.95,0.05),fontsize=16,
                      fontweight='bold',color='blue',horizontalalignment='right',xycoords='axes fraction')
    for i in reversed(range(5)):
        x, y = proj_y_lines[i]
        axarr[1].plot(x,y)
    axarr[1].set_xlabel('Pixel y')
    axarr[1].set_ylabel('Rotation angle')
    if center is not None :
        x_center = center[0]
        y_center = center[1]
        theta0 = center[2]
        axarr[0].scatter(x_center,theta0,s=200,color='red')
        axarr[1].scatter(y_center,theta0,s=200,color='red')
    axarr[1].annotate(holo_name,xy=(0.95,0.05),xytext=(0.95,0.05),fontsize=16,
                      fontweight='bold',color='blue',horizontalalignment='right',xycoords='axes fraction')
    plt.show()

def plot_dispersion_axis(central_positions,all_theta,holo_name=''):
    fig=plt.figure(figsize=(6,6))
    for index in np.arange(len(all_theta)):
        y0=central_positions[index][1]
        x0=central_positions[index][0]
        x_new = np.linspace(0,IMSIZE, 50)
        y_new = y0 + (x_new-x0)*np.tan(all_theta[index]*np.pi/180.)
        if index==0 :
            plt.plot(x_new,y_new,'r-',label='dispersion axis')
            plt.plot(x0,y0,'bo',label='star centroids')
        else :
            plt.plot(x_new,y_new,'r-')
            plt.plot(x0,y0,'bo')
    plt.xlim([0,IMSIZE])
    plt.ylim([0,IMSIZE])
    plt.grid(color='black', ls='solid')
    plt.xlabel('Pixel x')
    plt.ylabel('Pixel y')
    plt.annotate(holo_name,xy=(0.05,0.9),xytext=(0.05,0.9),fontsize=16,fontweight='bold',color='blue',horizontalalignment='left',xycoords='axes fraction')
    plt.legend()
    plt.show()


def hessian_analysis(z):
    # le parametre sigma permet de lisser le hessien
    Hxx, Hxy, Hyy = hessian_matrix(z, sigma=0.1, order = 'xy')
    dets = np.zeros_like(Hxx)
    for i in range(dets.shape[0]):
        for j in range(dets.shape[1]):
            h = np.array([[Hxx[i,j],Hxy[i,j]],[Hxy[i,j],Hyy[i,j]]])
            dets[i,j] = np.linalg.det(h)
    return(Hxx,Hyy,Hxy,dets)

def scatter_interpol(central_positions,values,margin=10,kind='cubic',plot=False,holo_name='',zlabel=''):
    x, y = np.array(central_positions).T
    interp_2args = interpolate.interp2d(x, y, values, kind=kind)
    step = 1
    xx = np.arange(np.min(x)-margin,np.max(x)+margin,step)
    yy = np.arange(np.min(y)-margin,np.max(y)+margin,step)
    z = interp_2args(xx,yy)
    interp = lambda x: interp_2args(x[0],x[1])
    if plot :
        fig = plt.figure(figsize=(8,8))
        plt.scatter(x, y, c=values, s=100, cmap=cm.jet, marker='o', vmin=np.min(values), vmax=np.max(values),edgecolors='k')
        im = plt.imshow(z, extent=[np.min(xx)-margin, np.max(xx)+margin, np.min(yy)-margin, np.max(yy)+margin],origin='lower',aspect='auto',cmap=cm.jet)
        plt.xlabel('Pixel x')
        plt.ylabel('Pixel y')
        plt.annotate(holo_name,xy=(0.95,0.05),xytext=(0.95,0.05),fontsize=16,
                          fontweight='bold',color='blue',horizontalalignment='right',xycoords='axes fraction')
        cb = fig.colorbar(im,ax=plt.gca())#, cax=cbar_ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.set_label(zlabel) #,fontsize=16)
        plt.show()
    return(xx,yy,z,interp)

def surface_gradient(xx,yy,z,margin=10,plot=False,holo_name='',dir_top_images=None):
    gx, gy = np.gradient(z)
    grad = np.sqrt(gy**2+gx**2)
    grad = grad[margin:-margin,margin:-margin]
    min_grad = np.min(grad)
    null_grad = np.where(np.isclose(grad,min_grad))
    x_center = xx[null_grad[1]]
    y_center = yy[null_grad[0]]
    if plot :
        fig, ax = plt.subplots(1,2,figsize=(13.85,5), sharex=True,sharey=False)
        im = ax[0].imshow(z[margin:-margin,margin:-margin],origin='lower',cmap='rainbow', extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)])
        im2 = ax[1].imshow(grad,origin='lower',cmap='rainbow', extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)])
        ax[0].grid(color='white', ls='solid')
        ax[1].grid(color='white', ls='solid')
        ax[0].plot(x_center,y_center,'kx',markersize=20,label='Geometric center of hologram')
        ax[1].plot(x_center,y_center,'kx',markersize=20,label='Geometric center of hologram')
        ax[0].annotate(holo_name,xy=(0.5,1.05),xytext=(0.5,1.05),fontsize=16,
                          fontweight='bold',color='blue',horizontalalignment='center',xycoords='axes fraction')
        fig.subplots_adjust(wspace=0.235)
        #cbar_ax = fig.add_axes([0.45, 0.15, 0.03, 0.7])
        cb = fig.colorbar(im,ax=ax[0])#, cax=cbar_ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.set_label('Rotation angle $\\theta$') #,fontsize=16)
        #cbar_ax2 = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        cb2 = fig.colorbar(im2, ax=ax[1])#, cax=cbar_ax2)
        cb2.formatter.set_powerlimits((0, 0))
        cb2.update_ticks()
        cb2.set_label('Gradient $\\vec{\\nabla} \\theta$') #,fontsize=16)
        ax[0].set_xlabel('Pixel x')#,fontsize=14)
        ax[1].set_xlabel('Pixel x')#,fontsize=14)
        ax[0].set_ylabel('Pixel y')#,fontsize=14)
        ax[1].set_ylabel('Pixel y')#,fontsize=14)
        if dir_top_images is not None :
            figfilename=os.path.join(dir_top_images,'rotation_map.pdf')
            plt.savefig(figfilename)
        plt.show()
    return(grad)

def saddle_point(xx,yy,z,interp,margin=10,plot=True,verbose=True,holo_name='',vmin=-0.000025,vmax=0.000025,dir_top_images=None):
    grad = surface_gradient(xx,yy,z,margin=margin,plot=False)
    min_grad = np.min(grad)
    null_grad = np.where(np.isclose(grad,min_grad,rtol=1e-6))
    x_center = xx[null_grad[1]]
    y_center = yy[null_grad[0]]
    theta_tilt = interp([x_center,y_center])
    Hxx, Hxy, Hyy, hessians  = hessian_analysis(z)
    if verbose :
        print 'Minimum gradient of image:',min_grad
        print 'Hessian determinant at minimum:',hessians[null_grad][0],('(if negative: saddle point)')
        print 'Center at x0 = %.1f and y0 = %.1f with average tilt of %.2f degrees' % (x_center,y_center, theta_tilt)
    if plot :
        fig, ax = plt.subplots(2,2,figsize=(11,10), sharex=True,sharey=True)
        ax[0,0].imshow(Hxx[margin:-margin,margin:-margin],origin='lower',cmap='rainbow',aspect='auto',vmin=vmin,vmax=vmax, extent=[np.min(xx)+margin, np.max(xx)-margin, np.min(yy)+margin, np.max(yy)-margin])
        ax[0,0].grid(color='white', ls='solid')
        ax[0,0].text(800,800,'$H_{xx}=\\frac{\partial^2 \\theta}{\partial x^2}$',fontsize=18)
        ax[0,0].annotate(holo_name,xy=(0.5,1.05),xytext=(0.5,1.05),fontsize=16,
                          fontweight='bold',color='blue',horizontalalignment='center',xycoords='axes fraction')
        ax[0,0].plot(x_center,y_center,'kx',markersize=20,label='Geometric center of hologram')
        ax[1,0].imshow(Hyy[margin:-margin,margin:-margin],origin='lower',cmap='rainbow',aspect='auto',vmin=vmin,vmax=vmax, extent=[np.min(xx)+margin, np.max(xx)-margin, np.min(yy)+margin, np.max(yy)-margin])
        ax[1,0].grid(color='white', ls='solid')
        ax[1,0].text(800,800,'$H_{yy}=\\frac{\partial^2 \\theta}{\partial y^2}$',fontsize=18)
        ax[1,0].plot(x_center,y_center,'kx',markersize=20,label='Geometric center of hologram')
        im = ax[0,1].imshow(Hxy[margin:-margin,margin:-margin],origin='lower',cmap='rainbow',aspect='auto',vmin=vmin,vmax=vmax, extent=[np.min(xx)+margin, np.max(xx)-margin, np.min(yy)+margin, np.max(yy)-margin])
        ax[0,1].grid(color='white', ls='solid')
        ax[0,1].plot(x_center,y_center,'kx',markersize=20,label='Geometric center of hologram')
        im2 = ax[1,1].imshow(hessians[margin:-margin,margin:-margin],origin='lower',cmap='rainbow',aspect='auto',vmin=-vmin**2,vmax=vmax**2, extent=[np.min(xx)+margin, np.max(xx)-margin, np.min(yy)+margin, np.max(yy)-margin])
        ax[1,1].grid(color='white', ls='solid')
        ax[1,1].plot(x_center,y_center,'kx',markersize=20,label='Geometric center of hologram')
        ax[0,1].text(800,800,'$H_{xy}=\\frac{\partial^2 \\theta}{\partial x \partial y}$',fontsize=18)
        ax[1,1].text(800,800,'$det(H)$',fontsize=18)
        ax[1,0].set_xlabel('Pixel x')#,fontsize=14)
        ax[1,1].set_xlabel('Pixel x')#,fontsize=14)
        ax[0,0].set_ylabel('Pixel y')#,fontsize=14)
        ax[1,0].set_ylabel('Pixel y')#,fontsize=14)

        fig.subplots_adjust(hspace =0, wspace=0, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.55, 0.03, 0.3])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.set_label('Hessian values') #,fontsize=16)

        cbar_ax2 = fig.add_axes([0.85, 0.15, 0.03, 0.3])
        cb2 = fig.colorbar(im2, cax=cbar_ax2)
        cb2.formatter.set_powerlimits((0, 0))
        cb2.update_ticks()
        cb2.set_label('Hessian determinant') #,fontsize=16)

        #fig.subplots_adjust(0,0,1,1,0,0)
        if dir_top_images is not None :
            figfilename=os.path.join(dir_top_images,'rotation_hessian_maps.pdf')
            plt.savefig(figfilename)  
        plt.show()
    return(x_center,y_center,theta_tilt)


