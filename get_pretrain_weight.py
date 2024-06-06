import os

# Set environment variable for segmentation models
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import pandas as pd


import sys
# Add the parent directory of the 'models' directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'SegFormer-tf')))

import random
import glob
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from models import SegFormer_B3
import segmentation_models as sm
import albumentations as A

# retval = os.getcwd()
# print( "Thu muc dang lam viec hien tai la %s" % retval)

# Paths to the dataset
save_train_image_dataset_path = './save_data_path/train_image_dataset.npy'
save_val_image_dataset_path = './save_data_path/val_image_dataset.npy'
save_train_mask_image_dataset_path = './save_data_path/mask_train_image_dataset.npy'
save_val_mask_image_dataset_path = './save_data_path/mask_val_image_dataset.npy'

# Load dataset shapes
train_image_shape = np.load(save_train_image_dataset_path, mmap_mode='r').shape
val_image_shape = np.load(save_val_image_dataset_path, mmap_mode='r').shape
train_mask_shape = np.load(save_train_mask_image_dataset_path, mmap_mode='r').shape
val_mask_shape = np.load(save_val_mask_image_dataset_path, mmap_mode='r').shape

# Define generator functions with dtype conversion
def train_generator():
    X_train = np.load(save_train_image_dataset_path, mmap_mode='r')
    y_train = np.load(save_train_mask_image_dataset_path, mmap_mode='r')
    for i in range(train_image_shape[0]):
        yield X_train[i].astype(np.float32), y_train[i].astype(np.float32)

def val_generator():
    X_val = np.load(save_val_image_dataset_path, mmap_mode='r')
    y_val = np.load(save_val_mask_image_dataset_path, mmap_mode='r')
    for i in range(val_image_shape[0]):
        yield X_val[i].astype(np.float32), y_val[i].astype(np.float32)

# Define the output shapes and types
output_signature = (
    tf.TensorSpec(shape=train_image_shape[1:], dtype=tf.float32),
    tf.TensorSpec(shape=train_mask_shape[1:], dtype=tf.float32)
)

# Create TensorFlow datasets from the generators
train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=output_signature
)

val_dataset = tf.data.Dataset.from_generator(
    val_generator,
    output_signature=output_signature
)

# Optional: Apply any additional dataset operations (e.g., batching, shuffling)
batch_size = 32
train_dataset = train_dataset.repeat().batch(batch_size)
val_dataset = val_dataset.repeat().batch(batch_size)

# Path to save model checkpoint
checkpoint_path = "./pretrain_weights/segformer_B3/cp.weights.h5"

# Initialize and compile the model

dice_loss = sm.losses.DiceLoss() 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (2 * focal_loss)

model = SegFormer_B3(input_shape=(256, 256, 3), num_classes=5)
model.compile('Adam', loss=total_loss, metrics=[sm.metrics.iou_score])

# Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=1)
]

steps_per_epoch = np.floor(train_image_shape[0] / batch_size)
validation_steps = np.floor(val_image_shape[0] / batch_size)

# model.load_weights(checkpoint_path)

# Train the model
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=200,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks,
)

# Load the best weights
# model.load_weights(checkpoint_path)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('./pretrain_result/augmented_segformer_B3/loss.png')
plt.clf()  # Clear the current figure

acc = history.history['iou_score']
val_acc = history.history['val_iou_score']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('./pretrain_result/augmented_segformer_B3/mean_iou.png')
