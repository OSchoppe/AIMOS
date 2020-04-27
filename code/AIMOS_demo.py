import AIMOS_pipeline
from utils import filehandling
from utils import plotting

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


# ===============================================================================
# === INFO ======================================================================
#
# Fully functional demonstration of AIMOS pipeline
#
# This demo loads a 3D native micro-CT scan of a mouse and segments the major
# organs using the AIMOS pipeline. The predicted segmentation is then visualized
# and compared to the manually created delineation in two plots.


# ===============================================================================
# === Step 1: Setup =============================================================

# Define parameters
dataset   = 'NACT'
scanname  = 'M03_004h'
modelname = 'NACTmodel.pt'
basepath = '../'


# Define paths & read in data
print("Loading data...")
path_CIDs = basepath + 'data/' + dataset + '/trainingCIDs'
path_scan = basepath + 'data/' + dataset + '/' + scanname + '/C00'
path_gt   = basepath + 'data/' + dataset + '/' + scanname + '/GT'
CIDs     = filehandling.pload(path_CIDs)
vol_scan = filehandling.readNifti(path_scan)
vol_gt   = filehandling.readNifti(path_gt)


# ===============================================================================
# === Step 2: Load pretrained model & predict organ segmentation ================

# Configure
config = {'dataset': dataset, 'trainingCIDs': CIDs}
config = AIMOS_pipeline.complete_config_with_default(config)

# Load pretrained model
print("Loading model...")
path_model = basepath + 'trainedmodels/' + dataset + 'model.pt'
model = AIMOS_pipeline.load_demo_model(config,path_model)

# Predict segmentation
vol_pred = AIMOS_pipeline.segmentStudy(model, scanname, config)
print("Prediction completed.")


# ===============================================================================
# === Step 3: Define a few helper functions for plotting ========================

colors = {}
colors['Bone']   = np.asarray([152, 218, 220]) / 255
colors['Brain']  = np.asarray([ 47,  85, 151]) / 255
colors['Heart']  = np.asarray([192,   0,   0]) / 255
colors['Lung']   = np.asarray([172,  96,   0]) / 255
colors['Liver']  = np.asarray([ 89,  65,  13]) / 255
colors['Kidney'] = np.asarray([ 84, 130,  53]) / 255
colors['Spleen'] = np.asarray([112,  48, 160]) / 255

def get_representative_z(CID,vol_gt):
    '''
    z = get_representative_z(CID,vol_gt)
    
    This function returns the z-index of a representative coronal slice
    through the mouse body. CID is the class ID of the organ of interest.
    Set it to None if you prefer a slice through the entire body rather
    than a specific organ. vol_gt is the ground truth annotation volume.
    '''
    count = 0
    z_selected = None
    for z in range(0,vol_gt.shape[2]):
        if(CID is not None):
            current_count = np.sum(np.where(vol_gt[:,:,z] == CID))
        else:
            current_count = np.sum(np.where(vol_gt[:,:,z] > 0))
        count = np.max([count, current_count])
        if(current_count == count):
            z_selected = z
    return z_selected


def get_slice_visualization(vol_scan,z,HUmax=1000):
    '''
    img = get_slice_visualization(vol_scan,z)
    
    This function extracts a coronal slice (defined by index z) and
    normalizes it between 0 and 1000 Hounsfield Units
    '''
    img = copy.deepcopy(vol_scan[:,:,z])
    img = np.clip(img,0,HUmax)
    img = img / (np.max(img) + 1e-7)
    return img


def get_organ_bb(mask_gt,mask_pred,pad=10):
    '''
    y0, y1, x0, x1 = get_organ_bb(mask_gt,mask_pred,pad=10)
    
    This function returns the bounding box coordinates around an organ
    based on the predicted and ground truth binary masks. Use this to
    visualize the prediction accuracy for a given organ.
    '''
    try:
        y0, x0 = np.min([np.min(np.where(mask_gt),1),np.min(np.where(mask_pred),1)],0)
        y1, x1 = np.max([np.max(np.where(mask_gt),1),np.max(np.where(mask_pred),1)],0)
    except: # prediction is empty in this slice
        y0, x0 = np.min(np.where(mask_gt),1)
        y1, x1 = np.max(np.where(mask_gt),1)
    y0 = np.max([y0-pad,0])
    x0 = np.max([x0-pad,0])
    y1 = np.min([y1+pad,mask_gt.shape[0]])
    x1 = np.min([x1+pad,mask_gt.shape[1]])
    return y0, y1, x0, x1


def get_dice(gt, seg):
    '''
    dice = get_dice(gt, seg)
    
    This function computes the Soerensen-Dice score for a given binary
    predicted mask and a binary ground truth mask
    '''
    eps = 0.0001
    gt = gt.astype(np.bool)
    seg = seg.astype(np.bool)
    intersection = np.logical_and(gt, seg)
    dice = 2 * (intersection.sum() + eps) / (gt.sum() + seg.sum() + eps)
    return dice    



# ===============================================================================
# === Step 4: Plot prediction results ===========================================
    
plt.figure(num=1)
plt.clf()
print("Plotting visualizations of prediction result...")

# Plot whole-body visualizations (1 of 2)
plt.subplot(1,2,1)
rgbs = np.zeros([vol_scan.shape[0],vol_scan.shape[1],3])
for z in range(0,vol_scan.shape[2]):
    img = get_slice_visualization(vol_scan,z)
    rgb = np.zeros([vol_scan.shape[0],vol_scan.shape[1],3])
    for c in [0,1,2]: rgb[:,:,c] = img
    GT_slice       = copy.deepcopy(vol_gt[:,:,z])
    pred_seg_slice = copy.deepcopy(vol_pred[:,:,z])
    for organ in CIDs:
        CID   = CIDs[organ]
        color = colors[organ]
        truemask = np.zeros(GT_slice.shape)
        truemask[np.where(GT_slice == CID)] = True
        predmask = np.zeros(pred_seg_slice.shape)
        predmask[np.where(pred_seg_slice == CID)] = True
        for c in [0,1,2]:
            rgb[:,:,c] = rgb[:,:,c] + color[c] * np.clip(0.5*predmask * (3+rgb[:,:,c]),0,1)
    rgbs += rgb
rgbs = rgbs / np.max(rgbs)
plt.imshow(rgbs)
plt.title('Mean-intensity projection of entire body')

# Plot whole-body visualizations (2 of 2)
plt.subplot(1,2,2)
z = get_representative_z(None,vol_gt)
rgb = np.zeros([vol_scan.shape[0],vol_scan.shape[1],3])
img = get_slice_visualization(vol_scan,z)
for c in [0,1,2]: rgb[:,:,c] = img
GT_slice       = copy.deepcopy(vol_gt[:,:,z])
pred_seg_slice = copy.deepcopy(vol_pred[:,:,z])
for organ in CIDs:
    CID   = CIDs[organ]
    color = colors[organ]
    truemask = np.zeros(GT_slice.shape)
    truemask[np.where(GT_slice == CID)] = True
    predmask = np.zeros(pred_seg_slice.shape)
    predmask[np.where(pred_seg_slice == CID)] = True
    outline = cv2.morphologyEx(truemask, cv2.MORPH_GRADIENT, kernel=np.ones((2,2)))
    for c in [0,1,2]:
        rgb[:,:,c] = rgb[:,:,c] + color[c] * np.clip(0.5*predmask * (1+rgb[:,:,c]) + outline,0,1)
rgb = np.clip(rgb,0,1)
plt.imshow(rgb)
plt.title('Representative coronal slice')
plt.suptitle('Predicted segmentation vs. Ground Truth\n(Whole body view)')


# Plot raw scan & segmentation for each organ in detail
plt.figure(num=2)
plt.clf()
for o,organ in enumerate(CIDs):
    CID = CIDs[organ] 
    z = get_representative_z(CID,vol_gt)
    img = get_slice_visualization(vol_scan,z,HUmax=500)
    mask_gt   = (vol_gt[:,:,z]   ==  CID).astype(np.uint16)
    mask_pred = (vol_pred[:,:,z] ==  CID).astype(np.uint16)
    y0, y1, x0, x1 = get_organ_bb(mask_gt,mask_pred)
    ax = plt.subplot(2,len(CIDs),o+1)
    plotting.intensity(img[y0:y1,x0:x1], color='white', ahandle = ax)
    plt.title(organ)
    ax = plt.subplot(2,len(CIDs),o+1+len(CIDs))
    plotting.mask_pred_overlay(img[y0:y1,x0:x1], mask_gt[y0:y1,x0:x1], mask_pred[y0:y1,x0:x1], color=colors[organ], ahandle = ax)
    plt.title(organ + ': ' + str(int(100*get_dice(mask_gt, mask_pred)))+'%')
plt.suptitle('Predicted segmentation vs. Ground Truth\n(Individual organ view)')


print("Demonstration complete.")
