
﻿# Centroid Based Concept Learning for RGB-D Indoor Scene Classification
Keras code for the paper: [Centroid Based Concept Learning for RGB-D Inddor Scene Classification](https://arxiv.org/abs/1911.00155) 
## Applied on CIFAR-100, Caltech-101 and CUBS-200-2011 

### Requirements
* torch (Current working with 1.3.1)
* Scipy (Currently working with 1.2.1)
* Scikit Learn (Currently working with 0.21.2)
* Get the models pretrained on Places365 dataset from from https://github.com/CSAILVision/places365
* Download the SUNRGBD and NYU datasets in */data directory
## Usage
* First run ```depth_training.py``` to train VGG16 from scratch on depth data. You can change the dataset from SUN to NYU in the file to train on NYU Depth V2 dataset.
* Run ```get_features.py``` twice to get the VGG16 features for all the RGB and depth images in the dataset. Change the model name in the file to get the correct features. 
* Features are extracted using an updated version of the ```img2vec.py``` file from https://github.com/christiansafka/img2vec repo. Check that repo if you want more details.
* After feature extraction, simply run ```main_file.py``` to get the results. 
* You can also get the silhouette analysis results from main.py. The helper functions are called from the Functions.py file.
* If you want to run the experiment with updated labels. Change category labels based on the merged categories provided in supplementary file of the paper.
## If you consider citing us
```
@InProceedings{Ayub_2020_BMVC,  
author = {Ayub, Ali and Wagner, Alan R.},  
title = {Centroid Based Concept Learning for RGB-D Indoor Scene Classification},  
booktitle = {British Machine Vision Conference (BMVC)},  
year = {2020}  
}
```
