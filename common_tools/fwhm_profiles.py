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


def FWHMProfiles(all_images,thex0,they0,all_expo,all_titles,object_name,all_filt,width_to_stack=10,central_star_cut = 100, width_cut = 20, left_edges = None, right_edges = None, max_fwhm=20, bgd_cut=30, bgd_width=20, bgd_deg=2, plot=False):
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
        bgd = full_image[y_0-bgd_cut-width_cut:y_0+bgd_cut+width_cut,left_edges[index]:right_edges[index]]/all_expo[index]
        bgd_pixels = np.array(list(np.arange(0, bgd_width))+list(bgd.shape[0]+np.arange(-bgd_width, 0)))
        for x in range(0,image.shape[1],width_to_stack):
            # stack profile over a small width
            xtest = x+left_edges[index]
            profile = np.median(image.T[x:x+width_to_stack],axis=0)
            bgd_profile = np.median(bgd.T[xtest:xtest+width_to_stack],axis=0)
            pval = np.polyfit(bgd_pixels, bgd_profile[bgd_pixels], bgd_deg)
            bgd_eval = np.polyval(pval, range(image.shape[0]))
            if np.max(profile-bgd_eval) < 3*np.sqrt(np.median(bgd_eval)):
                continue
            f = fwhm(range(image.shape[0]),profile-bgd_eval,max_fwhm=max_fwhm) 
            if not np.isnan(f):
                fwhms.append( f)
                xs.append(xtest)
                if plot:
                    fig=plt.figure(figsize=(20,6))
                    plt.imshow(image.T[x:x+width_to_stack],origin='lower',cmap='rainbow',vmin=0,vmax=10)
                    plt.plot(range(image.shape[0]),profile,'b-',lw=3)
                    plt.plot(range(image.shape[0]),image.T[x],'r--',lw=3)
                    plt.plot(range(image.shape[0]),bgd_eval,'g--', lw=3)
                    plt.show()
        theprofiles.append([xs,fwhms])
        plt.plot(xs,fwhms, "+")
        plt.xlabel('Pixel x')
        plt.ylabel('FWHM')
        plt.ylim([0,20])
    return(theprofiles)
            
