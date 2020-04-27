import AIMOS_pipeline
from utils import filehandling

import numpy as np


# ===============================================================================
# === INFO ======================================================================
#
# Train AIMOS on your own data
#
# Notes: 
#  - Please first prepare your data with AIMOS_prepare_data.py
#  - We recommend at least 8-12 annotated whole-body scans and 30 epochs
#  - Consider using a pre-trained model instead of training from scratch


# ===============================================================================
# === Step 1: Define parameters =================================================
basepath = '../'
config = {}
config["runName"]          = 'MyFirstTestRun'
config["dataset"]          = "LSFM" 
config["numEpochs"]        = 30
config["initialLR"]        = 1e-3
config["path_for_results"] = basepath + 'results/'
#config["pretrain_path"]    = basepath + 'path/to/pretrainedmodel.pt' 


# ===============================================================================
# === Step 3: Define training, validation, and test data ========================

path_data = basepath + 'data/' + config["dataset"] + '/'
mice = filehandling.listfolders(path_data,searchstring='mouse_')

config["trainingCIDs"] = filehandling.pload(path_data + 'trainingCIDs')
config["trainingStudies"] = mice

rand_idx = np.random.randint(0,len(config["trainingStudies"]))
config["validationStudies"] = [config["trainingStudies"].pop(rand_idx)]
rand_idx = np.random.randint(0,len(config["trainingStudies"]))
config["testStudies"] = [config["trainingStudies"].pop(rand_idx)]

print("At total of "+str(len(mice))+" annotated mice were found.")
print("We will train AIMOS on "+str(len(config["trainingStudies"]))+" scans and use the others for validation and testing.")


# ===============================================================================
# === Step 3: Train AIMOS and assess performance on test set ====================
config = AIMOS_pipeline.complete_config_with_default(config)
AIMOS_pipeline.run_session(config)
print("Finished training and testing.")