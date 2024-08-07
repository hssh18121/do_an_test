import os

# Set environment variable for segmentation models
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import pandas as pd


import sys

import random
import glob
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from Upernet.models import upernet_convnext_tiny_org
from SegFormer.models import SegFormer_B0
import segmentation_models as sm
import albumentations as A

# Enable mixed precision
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Paths to dataset files
save_train_image_dataset_path = './bk-isut-dataset/train_image_dataset.npy'
save_val_image_dataset_path = './bk-isut-dataset/val_image_dataset.npy'
save_test_image_dataset_path = './bk-isut-dataset/test_image_dataset.npy'
save_train_mask_image_dataset_path = './bk-isut-dataset/mask_train_image_dataset.npy'
save_val_mask_image_dataset_path = './bk-isut-dataset/mask_val_image_dataset.npy'
save_test_mask_image_dataset_path = './bk-isut-dataset/mask_test_image_dataset.npy'

# Load datasets
X_train = np.load(save_train_image_dataset_path, mmap_mode='c')
X_val = np.load(save_val_image_dataset_path, mmap_mode='c')
X_test = np.load(save_test_image_dataset_path, mmap_mode='c')
y_train = np.load(save_train_mask_image_dataset_path, mmap_mode='c')
y_val = np.load(save_val_mask_image_dataset_path, mmap_mode='c')
y_test = np.load(save_test_mask_image_dataset_path, mmap_mode='c')


# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# Define the augmentation function
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation(height, width):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.6, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0, value=0),
        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0, value=0),
        A.RandomCrop(height=height, width=width, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

# Function to apply augmentation
def augment_data(image, mask):
    aug = get_training_augmentation(256, 256)(image=image, mask=mask)
    aug_image = aug['image'].astype(np.uint8)
    aug_mask = aug['mask'].astype(np.uint8)
    return aug_image, aug_mask

# Wrap the augmentation function for TensorFlow
def tf_augment_data(image, mask):
    aug_img, aug_mask = tf.numpy_function(func=augment_data, inp=[image, mask], Tout=[tf.uint8, tf.uint8])
    return aug_img, aug_mask

def _normalize(X_batch, y_batch):
    X_batch = tf.cast(X_batch, tf.float32)
    y_batch = tf.cast(y_batch, tf.float32)
    return X_batch, y_batch


augmented_train_dataset = train_dataset.map(tf_augment_data)
combined_train_dataset = train_dataset.concatenate(augmented_train_dataset)

train_dataset = combined_train_dataset.map(_normalize).batch(16)

val_dataset = val_dataset.map(_normalize).batch(16)

import logging

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

teacher_checkpoint_path = "./weights/best_model/cp.weights.h5"
student_pretrain_path = "./weights/offline_distill/cp.segformer_student_converge_with_best_model.h5"

# Build two student models
input_shape = (256, 256, 3)
num_classes = 5
teacher = upernet_convnext_tiny_org.UPerNet(input_shape=input_shape, num_classes=num_classes)
student =  SegFormer_B0(input_shape = (256,256,3), num_classes = 5)

teacher.load_weights(teacher_checkpoint_path)
student.load_weights(student_pretrain_path)

class KLDivergenceLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name="kl_divergence_loss"):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        y_true = tf.keras.backend.softmax(y_true / self.temperature)
        y_pred = tf.keras.backend.softmax(y_pred / self.temperature)
        return tf.keras.backend.sum(y_true * tf.keras.backend.log(y_true / y_pred), axis=-1)

dice_loss = sm.losses.DiceLoss() 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (2 * focal_loss)
kl_loss = KLDivergenceLoss()

# Custom combined loss function
def custom_combined_loss(y_true, student_pred, teacher_pred):
    
    kl_loss_value = kl_loss(teacher_pred, student_pred)
    return dice_loss(y_true, student_pred) + (2 * focal_loss(y_true, student_pred)) + kl_loss_value


# Optimizers
optimizer = tf.keras.optimizers.Adam()
lr_patience = 0
total_patience = 0


# Define a custom training loop
epochs = 200
best_val_loss = np.inf
best_weights_student = None

for epoch in range(epochs):
    student_epoch_loss = []
    student_epoch_iou = []

    for step, (images_orig, labels_orig) in enumerate(train_dataset):
        # sys.stdout.write(f"step {step} ")
        with tf.GradientTape(persistent=True) as tape:

            # Forward pass through student with original dataset
            teacher_pred = teacher(images_orig, training=False)
            student_pred = student(images_orig, training=True)

            # Compute losses
            loss = custom_combined_loss(labels_orig, student_pred, teacher_pred)

            # sys.stdout.write(f"Loss: {loss} \n")


        # Compute gradients and update weights for both models
        grads = tape.gradient(loss, student.trainable_variables)

        optimizer.apply_gradients(zip(grads, student.trainable_variables))

        student_epoch_loss.append(loss.numpy())

    # Calculate and print metrics for the epoch
    student_mean_loss = np.mean(student_epoch_loss)

    print(f"Epoch {epoch+1}/{epochs}, Student Loss: {student_mean_loss}")
    logging.info(f"Epoch {epoch+1}/{epochs}, Student Distill Loss: {student_mean_loss}")

    # Validation step
    val_losses = []
    val_iou_scores = []

    for images, labels in val_dataset:
        preds = student(images, training=False)

        # combined_preds = (preds1 + preds2) / 2.0
        val_loss = dice_loss(labels, preds) + (2 * focal_loss(labels, preds))
        val_losses.append(val_loss.numpy())

        val_iou_scores.append(sm.metrics.iou_score(labels, preds).numpy())

    val_mean_loss = np.mean(val_losses)

    val_iou = np.mean(val_iou_scores)

    print(f"Epoch {epoch+1}/{epochs}, Student. Validation Loss: {val_mean_loss}, Validation IoU: {val_iou}")
    logging.info(f"Epoch {epoch+1}/{epochs}, Student. Validation Loss: {val_mean_loss}, Validation IoU: {val_iou}")


    # Check for the lowest validation loss and save the weights
    if val_mean_loss < best_val_loss:
        lr_patience = 0
        total_patience = 0
        best_val_loss = val_mean_loss
        best_weights_student = student.get_weights()
        print(f"New best validation loss of student: {val_mean_loss}. Saving the weights")
        logging.info(f"New best validation loss: {val_mean_loss}. Saving the weights")
        # Save the best weights
        student.set_weights(best_weights_student)
        student.save_weights('./weights/offline_distill/cp.segformer_student_converge_with_best_model.h5')
    else: 
        lr_patience += 1
        total_patience += 1
        if lr_patience == 8:
           new_lr = optimizer.learning_rate / 2
           optimizer.learning_rate.assign(new_lr)
           print(f"Reduce lr by 2. new lr: {new_lr}")

    if total_patience == 25:
        print("Result not improving after 25 epochs. Abort training")
        break
