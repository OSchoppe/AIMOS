import os 
import numpy as np
import pickle
import nibabel as nib


#%%
def psave(path, variable):
    '''
    psave(path, variable)
    
    Takes a variable (given as string with its name) and saves it to a file as specified in the path.
    The path must at least contain the filename (no file ending needed), and can also include a 
    relative or an absolute folderpath, if the file is not to be saved to the current working directory.
    
    # ToDo: save several variables (e.g. take X args, store them to special DICT, and save to file)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    path = path.replace('\\','/')
    cwd = os.getcwd().replace('\\','/')
    if(path[0:2] != cwd[0:2] and path[0:5] != '/mnt/'):
        path = os.path.abspath(cwd + '/' + path) # If relatice path was given, turn into absolute path
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    file = open(path, 'wb')
    pickle.dump(variable,file,protocol=4)


#%%
def pload(path):
    '''
    variable = pload(path)
    
    Loads a variable from a file that was specified in the path. The path must at least contain the 
    filename (no file ending needed), and can also include a relative or an absolute folderpath, if 
    the file is not to located in the current working directory.
    
    # ToDo: load several variables (e.g. load special DICT from file and return matching entries)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    path = path.replace('\\','/')
    cwd = os.getcwd().replace('\\','/')
    if(path[0:2] != cwd[0:2] and path[0:5] != '/mnt/'):
        path = os.path.abspath(cwd + '/' + path) # If relatice path was given, turn into absolute path
    file = open(path, 'rb')
    return pickle.load(file)


#%%
def writeNifti(path,volume,compress=False):
    '''
    writeNifti(path,volume)
    
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. 
    '''
    if(path.find('.nii')==-1 and compress==False):
        path = path + '.nii'
    if(path.find('.nii.gz')==-1 and compress==True):
        path = path + '.nii.gz'
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    # Save volume with adjusted orientation
    # --> Swap X and Y axis to go from (y,x,z) to (x,y,z)
    # --> Show in RAI orientation (x: right-to-left, y: anterior-to-posterior, z: inferior-to-superior)
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume,0,1), affine=affmat)
    nib.save(NiftiObject,path)


def readNifti(path,reorient=None):
    '''
    volume = readNifti(path)
    
    Reads in the NiftiObject saved under path and returns a Numpy volume.
    This function can also read in .img files (ANALYZE format).
    '''
    if(path.find('.nii')==-1 and path.find('.img')==-1):
        path = path + '.nii'
    if(os.path.isfile(path)):    
        NiftiObject = nib.load(path)
    elif(os.path.isfile(path + '.gz')):
        NiftiObject = nib.load(path + '.gz')
    else:
        raise Exception("No file found at: "+path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    if(reorient=='uCT_Rosenhain' and path.find('.img')):
        # Only perform this when reading in raw .img files
        # from the Rosenhain et al. (2018) dataset
        #    y = from back to belly
        #    x = from left to right
        #    z = from toe to head
        volume = np.swapaxes(volume,0,2) # swap y with z
        volume = np.flip(volume,0) # head  should by at y=0
        volume = np.flip(volume,2) # belly should by at x=0
    return volume


#%%

def listfolders(path, searchstring=''):
    if(path[-1] != '/'):
        path = path + '/'
    folders = []
    for element in os.listdir(path):
        cond1 =  os.path.isdir(path + element) is True
        cond2 =  searchstring in element
        if(cond1 and cond2):
            folders.append(element)
    folders = sorted(folders)
    return folders


def listfiles(path, searchstring=''):
    if(path[-1] != '/'):
        path = path + '/'
    folders = []
    for element in os.listdir(path):
        cond1 =  os.path.isdir(path + element) is False
        cond2 =  searchstring in element
        if(cond1 and cond2):
            folders.append(element)
    folders = sorted(folders)
    return folders

















