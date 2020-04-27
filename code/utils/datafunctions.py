from utils import filehandling
from utils import tools

import os
import random
import numpy as np
import cv2
import copy
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

basepath = '../'

# ===============================================================================
# === PyTorch datasets used for AIMOS ===========================================

class pairedSlices(Dataset):
    """ 
    Class used load the images for training, performing augmentations and data changes
    based on the configuration. It loads individual slices instead of normalized volumes 
    """

    def __init__(self, studies, config, mode):
        self.config   = config
        self.mode     = mode
        self.imgPaths = []
        self.gtPaths  = []
        for study in studies: 
            imgFolder = os.path.join(basepath,'data',config['dataset'],study,config['modality'])
            gtFolder  = os.path.join(basepath,'data',config['dataset'],study,'GT')
            self.imgPaths += [imgFolder+'/'+filename for filename in sorted(os.listdir(imgFolder))]
            self.gtPaths  += [gtFolder +'/'+filename for filename in sorted(os.listdir(gtFolder))]
        if((mode == 'train' or mode == 'val') and config['emptySlices'] == 'ignore'):
            self.imgPaths, self.gtPaths = self.ignore_empty_slices(self.imgPaths, self.gtPaths)

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        imgPath = self.imgPaths[index]
        img     = cv2.imread(imgPath, 2).astype(np.float32)
        gtPath  = self.gtPaths[index]
        gtImg   = cv2.imread(gtPath, 2).astype(np.float32)
        img, gtImg = self.pairedTransformations(img, gtImg, self.config, self.mode)
        return img, gtImg

    def pairedTransformations(self, img, gtImg, config, mode):
        # Resize images
        img   = TF.to_pil_image(img)
        gtImg = TF.to_pil_image(gtImg)
        img   = TF.resize(img, size=(config["imgSize"], config["imgSize"]), interpolation=2)
        gtImg = TF.resize(gtImg, size=(config["imgSize"], config["imgSize"]), interpolation=0)
        if(self.mode == 'train'):
            # Rotate images
            if(random.choice([True, False]) and 'Rotate' in config['augmentations']):
                angle = random.randint(-10, -10)
                img   = TF.rotate(img, angle)
                gtImg = TF.rotate(gtImg, angle)
            # Randomly crop images
            if(random.choice([True, False]) and 'Crop' in config['augmentations']):
                i, j, h, w = transforms.RandomResizedCrop.get_params(gtImg, scale=(0.8, 1), ratio=(0.75, 1))
                img   = TF.resized_crop(img, i, j, h, w, size=(config["imgSize"], config["imgSize"]), interpolation=2)
                gtImg = TF.resized_crop(gtImg, i, j, h, w, size=(config["imgSize"], config["imgSize"]), interpolation=0)
       # Standardize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[img.mean()], std=[img.std()])
        gtImg = torch.from_numpy(np.expand_dims(np.array(gtImg), 0))
        return img, gtImg
    
    def ignore_empty_slices(self, imgSlicePaths, gtSlicePaths): 
        """ 
        Filters to slices with actual content in the annotations 
        """
        slicePaths = []
        gtPaths = []
        for imgIdx in range(0, len(gtSlicePaths)):
            gtImg = cv2.imread(gtSlicePaths[imgIdx], 2).astype(np.float32)
            if(gtImg.max() > 0):
                slicePaths.append(imgSlicePaths[imgIdx])
                gtPaths.append(gtSlicePaths[imgIdx])
        return slicePaths, gtPaths


class testSlices(Dataset):
    """ 
    Class used to set perform inference on one single testStudy during k-fold Cross-Validation 
    """

    def __init__(self, testStudy, config):
        self.config = config
        self.path = os.path.join(basepath, 'data', config['dataset'], testStudy, config['modality'])
        self.file_names = sorted(os.listdir(self.path))
        first_img = cv2.imread(os.path.join(self.path, self.file_names[0]), 2)
        self.original_shape = [first_img.shape[0], first_img.shape[1], len(self.file_names)] # heplful for reconstruction

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img = cv2.imread(os.path.join(self.path, file_name), 2).astype(np.float32)
        img = self.Transformations(img, self.config)
        return img, file_name

    def __len__(self):
        return len(self.file_names)

    def Transformations(self, img, config):
        # Resize images
        img = TF.to_pil_image(img)
        img = TF.resize(img, size=(config["imgSize"], config["imgSize"]), interpolation=2)
        # Standardize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[img.mean()], std=[img.std()])
        return img


# ===============================================================================
# === General data handling functions for AIMOS =================================

def save_model(model, config, epoch, trainLoss, valLoss):
    path = config["path_for_results"] + '/model.pt'
    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'trainLoss': trainLoss, 'valLoss': valLoss}
    torch.save(checkpoint, path)


def load_model(model, path, mode=None):
    """ Loads a model given a path """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if(mode == "eval"):
        model.eval()
    elif(mode == "train"):
        model.train()
    return model


def load_pretrained_model(config):
    # 1 load untrained model of desired architecture
    model = tools.choose_architecture(config)
    
    # 2 load pretrained model (with different last layer)
    config_pretrained = copy.deepcopy(config)
    config_pretrained['dataset'] = config_pretrained['pretrain_path'].split('data/')[1].split('model')[0]
    config_pretrained["trainingCIDs"] = filehandling.pload(basepath + 'data/' + config_pretrained["dataset"] + '/trainingCIDs')
    model_pretrained = tools.choose_architecture(config_pretrained)
    checkpoint = torch.load(config_pretrained['pretrain_path'])
    model_pretrained.load_state_dict(checkpoint['model_state_dict'])
    
    # 3 replace last layer of pretrained model with desired architecture
    model_pretrained._modules['classify'] = copy.deepcopy(model._modules['classify'])

    return model_pretrained



def save_config(config):
    path = config["path_for_results"] + '/config'
    filehandling.psave(path,config)


def save_test_metrics(metrics, config, scanname):
    path = os.path.join(config["path_for_results"], scanname, 'metrics')
    filehandling.psave(path,metrics)


def save_prediction(prediction, config, scanname, stage):
    if(stage == 'logits'):       filename = 'predicted_logits' # this will be a 4D input (CIDs,y,x,z)
    if(stage == 'probs'):        filename = 'predicted_probs'  # this will be a 4D input (CIDs,y,x,z)
    if(stage == 'segmentation'): filename = 'predicted_segmentation' # this will be a 3D input (y,x,z)
    path = os.path.join(config["path_for_results"],scanname,filename)
    filehandling.writeNifti(path,prediction,compress=True)


def load_GT_volume(dataset, study):
    gt_path = os.path.join(basepath, 'data', dataset, study,'GT.nii.gz')
    vol_gt = filehandling.readNifti(gt_path)
    return vol_gt



