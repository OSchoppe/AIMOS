import numpy as np
import torch
from torch.nn import functional as F

from utils import architecture

#%%

def choose_architecture(config):
    if(  config['architecture'] == 'Unet64'):
        model = architecture.UNet64(num_classes = len(config["trainingCIDs"]) + 1)
    elif(config['architecture'] == 'Unet128'):
        model = architecture.UNet128(num_classes = len(config["trainingCIDs"]) + 1)
    elif(config['architecture'] == 'Unet256'):
        model = architecture.UNet256(num_classes = len(config["trainingCIDs"]) + 1)
    elif(config['architecture'] == 'Unet512'):
        model = architecture.UNet512(num_classes = len(config["trainingCIDs"]) + 1)
    elif(config['architecture'] == 'Unet768'):
        model = architecture.UNet768(num_classes = len(config["trainingCIDs"]) + 1)
    elif(config['architecture'] == 'Unet1024'):
        model = architecture.UNet1024(num_classes = len(config["trainingCIDs"]) + 1)
    else:
        raise ValueError("Model not implemented")
    return model


class RunningAverage():
    """
    A simple class that maintains the running average of a quantity
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
        self.avg   = None
    
    def update(self, val):
        self.total += val
        self.steps += 1
        self.avg = self.total/float(self.steps)
    
    def __call__(self):
        return self.avg    


def get_metrics(vol_gt, vol_segmented, config):

    num_classes = len(config['trainingCIDs']) + 1 # add +1 for BG class (CID = 0)
    oneHotGT  = np.zeros((num_classes,vol_gt.shape[0],vol_gt.shape[1],vol_gt.shape[2]))
    oneHotSeg = np.zeros((num_classes,vol_gt.shape[0],vol_gt.shape[1],vol_gt.shape[2]))

    metrics = {}
    for classname in config['trainingCIDs'].keys():
        CID = config['trainingCIDs'][classname]
        oneHotGT[CID,:,:,:][np.where(vol_gt==CID)] = 1
        oneHotSeg[CID,:,:,:][np.where(vol_segmented==CID)] = 1
        metrics[classname] = {}
        metrics[classname]["DICE"] = dice(oneHotGT[CID,:,:,:], oneHotSeg[CID,:,:,:])
    return metrics


def dice(gt, seg):
    """
        compute dice score
    """
    eps = 0.0001
    gt = gt.astype(np.bool)
    seg = seg.astype(np.bool)
    intersection = np.logical_and(gt, seg)
    dice = 2 * (intersection.sum() + eps) / (gt.sum() + seg.sum() + eps)
    return dice    


def dice_loss(label, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize 
    the dice loss so we return the negated dice loss.

    Args:
        label: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.

    Taken from: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py#L78     
    """
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes)[label.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())

    dims = (0,) + tuple(range(2, label.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps))
    
    return [(1 - dice_loss.mean()), dice_loss.detach().cpu().numpy()]    


def sigmoid(x):
    '''
    Exact Numpy equivalent for torch.sigmoid()
    '''
    y = 1/(1+np.exp(-x))
    return y


def sortAbyB(listA, listB):
    '''
    sorted_listA = sortAbyB(listA, ListB)
    
    Sorts list A by values of list B (alphanumerically)
    '''
    if(listB == sorted(listB)):
        return listA # a) no sorting needed; b) also avoids error when all elements of A are identical
    else:
        return [a for _,a in sorted(zip(listB,listA))]


