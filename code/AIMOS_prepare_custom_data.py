from utils import filehandling

import os
import cv2
import numpy as np
import shutil


# ===============================================================================
# === INFO ======================================================================
#
# Prepare your own data to train & run AIMOS
#
# In the /data/ folder, please place a new folder with your data. The folder name
# is treated as the name for this dataset. It must contain a number of X 
# subfolders, named 'mouse_1' to 'mouse_X', that all contain a gray-value scan 
# (of any imaging modality) and a segmentation. Scan and annotation should be 3D
# Nifti volumes of the same size. In the segmentation volume, each voxel should be
# an integer that encodes the segmentation class (organ or anatomical structure).
#
# This script checks for completeness and consistency and then turns the volumes
# into coronal slices, saved as TIFFs, that can be individually loaded by AIMOS.
#
# AIMOS was designed for mouse organ segmentation. However, it can in theory be
# applied to any kind of volumetric segmentation data - regardless of imaging 
# modality, species, or the nature of the segmentation task to be performed. We
# expect it to work well on any kind of anatomical segmentation. Scans can be of
# arbitrary resolution; we recommend volumes around (256 pixel)³ for a start. The
# length, width, and depth of the scan do not need to match.


# ===============================================================================
# === Step 1: Setup =============================================================

# Define parameters
basepath  = '../'

dataset   = 'LSFM' # name of data set

class_IDs = {} # please note that '0' is reserved for background
class_IDs['Brain']  = 1
class_IDs['Heart']  = 2
class_IDs['Lung']   = 3
class_IDs['Liver']  = 4
class_IDs['Kidney'] = 5
class_IDs['Spleen'] = 6

filenames = {}
filenames['C00'] = 'scan_native.nii.gz' # name of scan volume
filenames['GT']  = 'GT.nii.gz' # name of segmentation file

# Check available scans & consistency
path_data = basepath + 'data/' + dataset + '/'
if(os.path.isdir(path_data) is False or os.path.isdir(path_data+'mouse_1') is False):
    raise Exception("Please load data into right directory")
mice = filehandling.listfolders(path_data,searchstring='mouse_')
for mouse in mice:
    scan_exists = os.path.isfile(path_data+mouse+'/'+filenames['C00'])
    gt_exists   = os.path.isfile(path_data+mouse+'/'+filenames['GT'])
    if(scan_exists is False or gt_exists is False):
        raise Exception("Data incomplete for " + mouse)
print("At total of "+str(len(mice))+" annotated mice were found.")


# ===============================================================================
# === Step 2: Prepare data for training =========================================

# Save class ID for future reference
filehandling.psave(path_data + 'trainingCIDs',class_IDs)

# Turn volumes into coronal sclices for faster processing
for mouse in mice:
    print("Generating training data for "+mouse)
    mpath = path_data+mouse+'/'
    
    # Delete old data
    for channel in filenames:
        try: 
            shutil.rmtree(mpath + '/' + channel + '/')
        except: 
            pass

    # Create coronal slices
    for channel in filenames:
        cpath = mpath + '/' + channel + '/'
        os.mkdir(cpath)
        vol = filehandling.readNifti(mpath + '/' + filenames[channel])
        # Normalize data to non-negative integers in 16 bit depth
        if(channel != 'GT'):
            vol = vol.astype(np.float)
            vol = vol - np.min(vol)
            vol = vol / np.max(vol)
            vol = (vol * (2**16-1))
        vol = vol.astype(np.uint16)
        # Save coronal slices
        n_z = vol.shape[2]
        for z in range(0,n_z):
            z_slice = vol[:,:,z]
            z_slice_name = "Z{:04.0f}.tif".format(z)
            cv2.imwrite(cpath + z_slice_name,z_slice)
        print(" Saved "+str(n_z)+" TIFF files for "+channel)

print("Custom dataset prepared for AIMOS.")













