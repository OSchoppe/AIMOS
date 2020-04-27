from utils import datafunctions
from utils import tools

import torch
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import time


# ===============================================================================
# === Run a single session for a given train, val, and test set =================

def run_session(config):
    """ 
    Performs a sesion of training and validation using the loaded datasets 
    """

    # Initialize model and optimizer
    model = tools.choose_architecture(config) # Set the model architecture
    if('pretrain_path' in config.keys()):
        tqdm.write(" Loading pretrained model...")
        time.sleep(0.5) # just for tqdm
        model = datafunctions.load_pretrained_model(config)
    model = model.cuda() 
    optimizer = torch.optim.Adam(model.parameters(), lr=config["initialLR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, verbose=True)
    
    # Load training & validation data
    trainSet = datafunctions.pairedSlices(config["trainingStudies"],   config, mode='train')
    valSet   = datafunctions.pairedSlices(config["validationStudies"], config, mode='val')
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=config["batchSize"], shuffle=True, num_workers=4)
    valLoader   = torch.utils.data.DataLoader(valSet,   batch_size=config["batchSize"], shuffle=True, num_workers=4)

    # Train model
    for epoch in range(config['numEpochs']):
        tqdm.write(" Epoch {}/{}".format(epoch+1,config['numEpochs']))
        time.sleep(0.5) # just for tqdm
        trainLoss, diceTrain = train_epoch(model, optimizer, trainLoader)
        valLoss,   diceVal   = validate_epoch(model, valLoader)
        scheduler.step(valLoss, epoch)
    
    # Save trained model, if desired
    if(config["saveModel"]):
        datafunctions.save_model(model, config, epoch+1, trainLoss, valLoss)
        print("Saved model to file.")

    # Perform inference on all testStudies (individually) and save to file
    for testStudy in config['testStudies']:
        metrics = test_model(model, testStudy, config)
        datafunctions.save_test_metrics(metrics, config, testStudy)
        tqdm.write(" Test Dice scores for " + testStudy + ":")
        for classname in config["trainingCIDs"].keys():
            padding = ":" + (7 - len(classname))*" "
            tqdm.write(" "+classname+padding+" {:4.1f}%".format(100*metrics[classname]['DICE']))
    
    del model, optimizer, scheduler
    torch.cuda.empty_cache()


# ===============================================================================
# === Core model functionality for training and prediction ======================

def train_epoch(model, optimizer, dataLoader):
    """ 
    training_loss, training_dice = train_epoch(model, optimizer, dataLoader)
    
    Performs a training step for one epoch (one full pass over the training set) 
    """

    model.train()
    lossValue = tools.RunningAverage()
    diceValue = tools.RunningAverage()
    with tqdm(total=len(dataLoader), position=0, leave=True) as (t):
        t.set_description('  Training  ')
        for i, (trainBatch, labelBatch) in enumerate(dataLoader):

            trainBatch, labelBatch = trainBatch.cuda(non_blocking=True), labelBatch.cuda(non_blocking=True)
            trainBatch, labelBatch = torch.autograd.Variable(trainBatch), torch.autograd.Variable(labelBatch)
            
            outputBatch = model(trainBatch)
            diceLoss, dice = tools.dice_loss(labelBatch.long(), outputBatch)
            loss = diceLoss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lossValue.update(loss.item())
            diceValue.update(dice)
            t.set_postfix(tloss=('{:04.1f}').format(100*lossValue()))
            t.update()
    return lossValue(), diceValue()


def validate_epoch(model, dataLoader):
    """ 
   validation_loss, validation_dice = validate_epoch(model, dataLoader)
    
    Computes a validation step for one epoch (one full pass over the validation set) 
    """

    model.eval()
    lossValue = tools.RunningAverage()
    diceValue = tools.RunningAverage()
    with tqdm(total=len(dataLoader), position=0, leave=True) as (t):
        t.set_description('  Validation')
        with torch.no_grad():
            for i, (trainBatch, labelBatch) in enumerate(dataLoader):
                
                trainBatch, labelBatch = trainBatch.cuda(non_blocking=True), labelBatch.cuda(non_blocking=True)
                trainBatch, labelBatch = torch.autograd.Variable(trainBatch, requires_grad=False), torch.autograd.Variable(labelBatch, requires_grad=False)
                
                outputBatch = model(trainBatch)
                loss, dice = tools.dice_loss(labelBatch.long(), outputBatch)
    
                lossValue.update(loss.item())
                diceValue.update(dice)
                t.set_postfix(vloss=('{:04.1f}').format(100*lossValue()))
                t.update()
    return lossValue(), diceValue()


def segmentStudy(model, scanname, config, OrderByFileName=True):
    """ 
    segmentation = segmentStudy(model, scanname, config)
    
    Performs inference on slices of test study and returns reconstructed volume of predicted segmentation
    """
    testSet = datafunctions.testSlices(scanname, config)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=config["batchSize"], shuffle=False, num_workers=4)
    logits_slice_list = []
    file_name_list   = []
    model.eval()
    with tqdm(total=len(testLoader), position=0, leave=True) as (t):
        t.set_description(' AIMOS prediction:')
        with torch.no_grad():
            for i, (imgs, file_names) in enumerate(testLoader):
                imgs = imgs.cuda(non_blocking=True)
                imgs = torch.autograd.Variable(imgs, requires_grad=False)
                logits = model(imgs)
                logits_of_batch = logits.detach().cpu().numpy()
                for b in range(0, logits_of_batch.shape[0]):
                    logits_slice = logits_of_batch[b,:,:,:] # batchsize, n_classes, height, width
                    logits_slice_list += [logits_slice]
                    file_name = file_names[b]
                    file_name_list += [file_name]
                t.update()
    # Re-order by filenames
    if(OrderByFileName):
        logits_slice_list = tools.sortAbyB(logits_slice_list, file_name_list)
    # Turn into segmentation volume
    logits_vol = np.asarray(logits_slice_list) # z-slices, n_classes, height, width
    logits_vol = np.moveaxis(logits_vol,0,-1)  # n_classes, height, width, z-slices
    probs_vol = tools.sigmoid(logits_vol)
    segmentation_vol = np.argmax(probs_vol, axis=0) # height, width, z-slices
    # Resample segmentation volume to original dimensions
    zoomFactors = np.asarray(testSet.original_shape) / np.asarray(segmentation_vol.shape)
    segmentation_resampled = scipy.ndimage.zoom(segmentation_vol, zoomFactors, order=0)
    return segmentation_resampled


def test_model(model, scanname, config):
    """ 
    metrics = test_model(model, scanname, config)
    
    Gets DICE scores for all organs of given test study 
    """
    tqdm.write("Testing model on " + scanname)
    time.sleep(0.5) # just for tqdm
    vol_gt = datafunctions.load_GT_volume(config['dataset'], scanname)
    vol_segmented = segmentStudy(model, scanname, config)
    metrics = tools.get_metrics(vol_gt, vol_segmented, config)
    if(config["saveSegs"]):
        datafunctions.save_prediction(vol_segmented.astype(vol_gt.dtype), config, scanname, stage='segmentation')
    return metrics



# ===============================================================================
# === Basic helper function =====================================================


def complete_config_with_default(config):
    default_config = {}
    default_config["description"]   = "Description of experiment"
    default_config["runName"]       = 'Default'
    default_config["dataset"]       = None # there is not default data set
    default_config["architecture"]  = 'Unet768'
    default_config["initialLR"]     = 1e-3
    default_config["batchSize"]     = 32
    default_config["numEpochs"]     = 30
    default_config["imgSize"]       = 256
    default_config["modality"]      = 'C00'
    default_config['emptySlices']   = 'ignore' 
    default_config['augmentations'] = 'RotateCrop' 
    default_config["saveModel"]     = False
    default_config["saveLogits"]    = False
    default_config["saveProbs"]     = False
    default_config["saveSegs"]      = False
    # determine runName based on configuration
    ignore_list = ['runName','description','dataset','saveModel','saveLogits','saveProbs','saveSegs']
    for key in default_config.keys():
            if(key not in config.keys()):
                config[key] = default_config[key]
            if(config[key] != default_config[key] and key not in ignore_list):
                config["runName"] += '_'+key+str(config[key])
                
    config["path_for_results"] += config["runName"]  +'/'
    return config


def load_demo_model(config,path_model):
    model = tools.choose_architecture(config)
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    return model