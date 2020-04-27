import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors 


#%%
def maskoverlay(img, mask, ahandle=None, outline=False):
    '''     
    maskoverlay(img, mask, ahandle=None)
    
    Takes grayscale img and boolean mask of equal size and plots the overlay. The img is treated as 
    normalized between 0 and 1. If not normalized, the img will be normalized by setting its maximum
    value to 1
    
    Optional inputs:
        * ahandles - An axis handels on which to plot
    '''
    if(ahandle is None): 
        ahandle = plt.gca()
        ahandle = plt.cla()
    
    if(np.max(img) > 1):
        img = img / np.max(img)
    
    rgb = np.zeros([img.shape[0],img.shape[1],3])
    rgb[:,:,1] = img
    rgb[:,:,0] = mask * img
    if(outline):
        rgb[:,:,2] = 1-mask  # --> Make background blue to inspect oversegmentation
    ahandle.imshow(rgb)


#%%
def maskcomparison(truemask, predictedmask, ahandle=None):
    '''     
    maskcomparison(truemask, predictedmask, ahandle=None)
    
    Takes two binary 2D arrays (masks) as input and shows an RGB image to visualize the comparison of
    both masks. For this, the first mask is treated as the truth and the second as a prediction. Color
    shadings visualize true positives (green), false positives (red) and false negatives (blue).
    
    Optional inputs:
        * ahandles - An axis handels on which to plot
    '''
    if(ahandle is None): 
        ahandle = plt.gca()
        ahandle = plt.cla()
        
    assert truemask.size == len((np.where(truemask==1))[0]) + len((np.where(truemask==0))[0])
    assert predictedmask.size == len((np.where(predictedmask==1))[0]) + len((np.where(predictedmask==0))[0])
    
    rgb = np.zeros([truemask.shape[0],truemask.shape[1],3])
    truepositives = np.zeros([truemask.shape[0],truemask.shape[1]])
    falsepositives = np.zeros([truemask.shape[0],truemask.shape[1]])
    falsenegatives = np.zeros([truemask.shape[0],truemask.shape[1]])
    truepositives[np.where(truemask+predictedmask==2)] = 1
    falsepositives[np.where(predictedmask-truemask==1)] = 1
    falsenegatives[np.where(truemask-predictedmask==1)] = 1
    rgb[:,:,0] = falsepositives
    rgb[:,:,1] = truepositives
    rgb[:,:,2] = falsenegatives
    ahandle.imshow(rgb)

#%%
def mask_pred_overlay(img, truemask, predictedmask, color=[0,1,0], opacity=1, ahandle=None):
    '''     
        
    Optional inputs:
        * opacity - a value between 0 and ca. 2-3 try something like 0.5 or 1.0 first
        * ahandles - An axis handels on which to plot
    '''
    if(ahandle is None): 
        ahandle = plt.gca()
        ahandle = plt.cla()
        
    assert truemask.size == len((np.where(truemask==1))[0]) + len((np.where(truemask==0))[0])
    assert predictedmask.size == len((np.where(predictedmask==1))[0]) + len((np.where(predictedmask==0))[0])
    
    if(np.max(img) > 1):
        img = img / np.max(img)
    
    thickness = int(np.ceil(np.min(img.shape)/50))
    outline = cv2.morphologyEx(truemask, cv2.MORPH_GRADIENT, kernel=np.ones((thickness,thickness)))
    
    rgb = np.zeros([img.shape[0],img.shape[1],3])
    # add semi-transparent shading
    for c in [0,1,2]:
        rgb[:,:,c] = img + color[c] * np.clip(0.5*predictedmask * (opacity+img),0,1)
    rgb = rgb / np.max(rgb)
    # add fully opaque outline
    for c in [0,1,2]:
        rgb[:,:,c] = np.clip(rgb[:,:,c] + color[c] * outline,0,1)
    
    ahandle.imshow(rgb)


#%% 
def intensity(array, ahandle=None, color='green', cap=None):
    '''
    intensity(array, ahandle=None, color='green', cap=None)
    
    Plots 2D numpy array based on intensity with monochromatic plot. If no signal cap is provided 
    and the maximum value is above 1, the image will be normalized by its maximum value.
    '''
    if(ahandle is None):
        ahandle = plt.gca()
        ahandle = plt.cla()
    if(cap is not None):
        array = np.clip(array,0,cap) / cap
    elif(np.max(array)>1):
        array = array / np.max(array)
    ahandle.imshow(array,cmap=cmap_intensity(color),vmin=0,vmax=1)


#%% 
def cmap_intensity(color):
    C = np.zeros((256,3))
    if(color=='red'   or color=='yellow'  or color=='magenta' or color=='white'): C[:,0] = np.linspace(0,255,num=256)
    if(color=='green' or color=='yellow'  or color=='cyan'    or color=='white'): C[:,1] = np.linspace(0,255,num=256)
    if(color=='blue'  or color=='magenta' or color=='cyan'    or color=='white'): C[:,2] = np.linspace(0,255,num=256)
    if(color=='black'): 
        for c in [0,1,2]:
            C[:,c] = np.linspace(255,0,num=256)
    return matplotlib.colors.ListedColormap(C/255.0)

