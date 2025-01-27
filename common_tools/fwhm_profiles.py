import numpy as np
import matplotlib.pyplot as plt


def fwhm(ys,intensities,max_fwhm=50):
    imax = np.max(intensities)
    if np.isclose(imax,0.0):
        return np.nan
    half = 0.5*imax
    bounds = [y for y in np.arange(0,len(ys),0.1) if np.interp(y,ys,intensities)>half]
    if len(bounds)==0 : 
        return np.nan
    fwhm = bounds[-1]-bounds[0]
    if fwhm > max_fwhm : 
        return np.nan
    return fwhm


def FWHMProfiles(all_images,thex0,they0,all_expo,all_titles,object_name,all_filt,width_to_stack=10,central_star_cut = 100, width_cut = 20, left_edges = None, right_edges = None, max_fwhm=20):
    theprofiles = []
    fig = plt.figure(figsize=(20,8))
    if left_edges == None : left_edges = np.zeros(len(all_images)).astype(int)
    if right_edges == None : right_edges = 1900*np.ones(len(all_images)).astype(int)
    if not isinstance(left_edges, (list, tuple, np.ndarray)) :
        left_edges = [left_edges]*len(all_images)
    if not isinstance(right_edges, (list, tuple, np.ndarray)) :
        right_edges = [right_edges]*len(all_images)
        
    for index in range(len(all_images)):
        
        fwhms = []
        xs = []
        
        x_0=int(thex0[index])
        y_0=int(they0[index])

        full_image=np.copy(all_images[index])
        full_image[:,x_0-central_star_cut:x_0+central_star_cut]=0 ## TURN OFF CENTRAL STAR
        image=full_image[y_0-width_cut:y_0+width_cut,left_edges[index]:right_edges[index]]/all_expo[index]
        for x in range(0,image.shape[1],width_to_stack):
            # stack profile over a small width
            profile = np.median(image.T[x:x+width_to_stack],axis=0)
            fwhms.append( fwhm(range(image.shape[0]),profile,max_fwhm=max_fwhm) )
            xs.append(x+left_edges[index])
        theprofiles.append([xs,fwhms])
        plt.plot(xs,fwhms)
        plt.xlabel('Pixel x')
        plt.ylabel('FWHM')
        plt.ylim([0,20])
    return(theprofiles)
            
