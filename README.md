# AIMOS
<b>AI-enabled Mouse Organ Segmentation</b>
![AIMOS - AI-based Mouse Organ Segmentation ](https://www.tum.de/fileadmin/_processed_/a/a/csm_201123_Oliver_Schoppe_AE_183_2100_c1de664141.jpg)
This repository contains the code to apply or adapt AIMOS, a deep learning processing pipeline for the segmentation of organs in volumetric mouse scans. The AIMOS pipeline is written in Python. The deep learning backbone is based on a Unet-like architecture and implemented in PyTorch. The code provided here comprises the architecture, the full inference pipeline, as well as a training procedure. Furthermore, we provide demonstration files guiding users through the steps of retraining AIMOS on custom datasets.

This code saved as the basis for the following research article:

O Schoppe, C Pan, J Coronel, H Mai, Z Rong, M Todorov, A Müskes, F Navarro, H Li, A Ertürk & B Menze. Deep learning-enabled multi-organ segmentation in whole-body mouse scans. Nature Communications 2020 (https://www.nature.com/articles/s41467-020-19449-7)



This code goes along with two examplary datasets:

*Native and contrast-enhanced micro-CT* 

Rosenhain, S., Magnuska, Z., Yamoah, G. et al. A preclinical micro-computed tomography database including 3D whole body organ segmentations. Sci Data 5, 180294 (2018). https://doi.org/10.1038/sdata.2018.294

*Light-sheet fluorescent microscopy*

Schoppe, Oliver, 2020, "AIMOS - light-sheet microscopy dataset", https://doi.org/10.7910/DVN/LL3C1R, Harvard Dataverse, V1

Pretrained models for these datasets can be found here:  
Schoppe, Oliver, 2020, "AIMOS - pre-trained models", https://doi.org/10.7910/DVN/G6VLZN, Harvard Dataverse, V1 

Please note that we also provide a fully functional live online demonstration on CodeOcean. Please refer to the manuscript for a link to the CodeOcean demonstration.
