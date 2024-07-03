# Graduation Thesis: Applying Deep Learning for Multi-Class Segmentation in Drone Imagery

## Introduction

This repository contains the code used for the thesis titled "Applying Deep Learning for Multi-Class Segmentation in Drone Imagery." 
The project involves data preprocessing, model implementation, training, and evaluation for multi-class segmentation using deep learning techniques on drone-captured images.

## Preprocess Data
First download the required dataset from an external source. All the code regarding the data preprocessing process is in the folder `preprocess_data`

For converting the json label to mask image, please refer to this link: 
https://drive.google.com/drive/folders/1e9lCYpOoDrmp_LlHsnXr3kbrFKIW4734?usp=drive_link


## Training Models
The model implementation is in folder `SegFormer` and `Upernet`

Run the files with `train` prefix to train the corresponding model

For example:
```
python train_upernet.py

```

The loss and mean IoU changes during the training process are in `result` folder

The weight of the models is saved in the `weights` folder (The content in this folder is ignored for reduced size) 

## Evaluation
The result analysis process could be inspected via:
- `eval` file for evaluating the mean IoU on the test set
- `plot_prediction` file to inspect predictions on test patches
-  `inference` file for inference on a large image and calculate inference time. 

The inference result and prediction on test patches could be found in `infer_image` folder
