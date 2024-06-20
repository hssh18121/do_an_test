import os

# Set environment variable for segmentation models
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import pandas as pd


import sys
# Add the parent directory of the 'models' directory to the sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Upernet')))

import random
import glob
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from Upernet.models import upernet_convnext_improved
from SegFormer.models import SegFormer_B3
import segmentation_models as sm
import albumentations as A

# retval = os.getcwd()
# print( "Thu muc dang lam viec hien tai la %s" % retval)

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
    aug_img.set_shape((256, 256, 3))
    aug_mask.set_shape((256, 256, 1))
    return aug_img, aug_mask

def _normalize(X_batch, y_batch):
    X_batch = tf.cast(X_batch, tf.float32)
    y_batch = tf.cast(y_batch, tf.float32)
    return X_batch, y_batch

augmented_train_dataset = train_dataset.map(tf_augment_data)
combined_train_dataset = train_dataset.concatenate(augmented_train_dataset)

# Determine the size of the combined dataset
dataset_size = sum(1 for _ in combined_train_dataset)

# Calculate the midpoint of the dataset
midpoint = dataset_size // 2

print(midpoint)

# Split the dataset into two parts
first_half_dataset = combined_train_dataset.take(midpoint)
second_half_dataset = combined_train_dataset.skip(midpoint)

# Batch and normalize the datasets
augmented_train_dataset = second_half_dataset.batch(4).map(_normalize)
train_dataset = first_half_dataset.batch(4).map(_normalize)
val_dataset = val_dataset.batch(4).map(_normalize)

import logging

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

pretrain_upernet_checkpoint_path = "./weights/imgnet_augmented_upernet_convnext_improved/cp.weights.h5"
pretrain_segformer_checkpoint_path = "./weights/augmented_segformer_B3_with_pretrain/cp.weights.h5"

# Build two student models
input_shape = (256, 256, 3)
num_classes = 5
student1 = upernet_convnext_improved.UPerNet(input_shape=input_shape, num_classes=num_classes, name_prefix="convnext1")
student2 = SegFormer_B3(input_shape=input_shape, num_classes=num_classes)

student1.load_weights(pretrain_upernet_checkpoint_path)
student2.load_weights(pretrain_segformer_checkpoint_path)
# Assuming UPerNetConvnext and UPerNet are defined somewhere

class KLDivergenceLoss(tf.keras.losses.Loss):
    def __init__(self, name="kl_divergence_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.keras.backend.clip(y_true, 1e-10, 1)
        y_pred = tf.keras.backend.clip(y_pred, 1e-10, 1)
        return tf.keras.backend.sum(y_true * tf.keras.backend.log(y_true / y_pred), axis=-1)

dice_loss = sm.losses.DiceLoss() 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (2 * focal_loss)
kl_loss = KLDivergenceLoss()

# Define trainable alphas
alphas = tf.Variable([0.5, 0.5], dtype=tf.float32, trainable=True)

# Custom combined loss function
# Custom combined loss function
def custom_combined_loss(y_true, y_pred, preds1, preds2, alphas):
    # Calculate the weighted sum of preds1 and preds2
    combined_preds = alphas[0] * preds1 + alphas[1] * preds2
    # Ensure the weights sum to 1 and are non-negative
    combined_preds = tf.clip_by_value(combined_preds, 1e-10, 1.0)
    combined_preds = combined_preds / tf.reduce_sum(combined_preds, axis=-1, keepdims=True)
    
    kl_loss_value = kl_loss(combined_preds, y_pred)
    return dice_loss(y_true, y_pred) + (2 * focal_loss(y_true, y_pred)) + kl_loss_value


# Optimizers
optimizer1 = tf.keras.optimizers.Adam()
optimizer2 = tf.keras.optimizers.Adam()

alpha_optimizer = tf.keras.optimizers.Adam()

# Define a custom training loop
epochs = 30
best_val_loss1 = np.inf
best_val_loss2 = np.inf
best_weights_student1 = None
best_weights_student2 = None

for epoch in range(epochs):
    student1_epoch_loss = []
    student2_epoch_loss = []
    student1_epoch_iou = []
    student2_epoch_iou = []

    for step, (images_orig, labels_orig) in enumerate(train_dataset):
        print(f"step {step}")
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(alphas)  # Ensure alphas is watched

            # Forward pass through student1 with original dataset
            preds1 = student1(images_orig, training=True)
            # Forward pass through student2 with augmented dataset
            preds2 = student2(images_orig, training=True)

            # Compute losses
            loss1 = custom_combined_loss(labels_orig, preds1, preds1, preds2, alphas)
            loss2 = custom_combined_loss(labels_orig, preds2, preds1, preds2, alphas)

            print(f"Loss1: {loss1}")
            print(f"Loss2: {loss2}")

            total_loss = loss1 + loss2


        # Compute gradients and update weights for both models
        grads1 = tape.gradient(loss1, student1.trainable_variables)
        grads2 = tape.gradient(loss2, student2.trainable_variables)

        alpha_grads = tape.gradient(total_loss, [alphas])
        # print(alpha_grads)

        optimizer1.apply_gradients(zip(grads1, student1.trainable_variables))
        optimizer2.apply_gradients(zip(grads2, student2.trainable_variables))

        if alpha_grads[0] is not None:
            alpha_optimizer.apply_gradients(zip(alpha_grads, [alphas]))

        student1_epoch_loss.append(loss1.numpy())
        student2_epoch_loss.append(loss2.numpy())

    # Calculate and print metrics for the epoch
    student1_mean_loss = np.mean(student1_epoch_loss)
    student2_mean_loss = np.mean(student2_epoch_loss)

    print(f"Epoch {epoch+1}/{epochs}, Student1 Loss: {student1_mean_loss}")
    print(f"Epoch {epoch+1}/{epochs}, Student2 Loss: {student2_mean_loss}")
    print(f"Epoch {epoch+1}/{epochs}, Alpha vals: {alphas[0]}, {alphas[1]}")

    logging.info(f"Epoch {epoch+1}/{epochs}, Student1 Distill Loss: {student1_mean_loss}")
    logging.info(f"Epoch {epoch+1}/{epochs}, Student2 Distill Loss: {student2_mean_loss}")

    # Validation step
    val_losses1 = []
    val_losses2 = []
    val_iou_scores1 = []
    val_iou_scores2 = []

    for images, labels in val_dataset:
        preds1 = student1(images, training=False)
        preds2 = student2(images, training=False)
        # combined_preds = (preds1 + preds2) / 2.0
        val_loss1 = dice_loss(labels, preds1, per_image=True) + (2 * focal_loss(labels, preds1, per_image=True))
        val_losses1.append(val_loss1.numpy())

        val_loss2 = dice_loss(labels, preds2, per_image=True) + (2 * focal_loss(labels, preds2, per_image=True))
        val_losses2.append(val_loss2.numpy())

        val_iou_scores1.append(sm.metrics.iou_score(labels, preds1).numpy())
        val_iou_scores2.append(sm.metrics.iou_score(labels, preds2).numpy())

    val_mean_loss1 = np.mean(val_losses1)
    val_mean_loss2 = np.mean(val_losses2)

    val_iou1 = np.mean(val_iou_scores1)
    val_iou2 = np.mean(val_iou_scores2)

    print(f"Epoch {epoch+1}/{epochs}, Student 1. Validation Loss: {val_mean_loss1}, Validation IoU: {val_iou1}")
    print(f"Epoch {epoch+1}/{epochs}, Student 2. Validation Loss: {val_mean_loss2}, Validation IoU: {val_iou2}")
    logging.info(f"Epoch {epoch+1}/{epochs}, Student 1. Validation Loss: {val_mean_loss1}, Validation IoU: {val_iou1}")
    logging.info(f"Epoch {epoch+1}/{epochs}, Student 2. Validation Loss: {val_mean_loss2}, Validation IoU: {val_iou2}")

    # Check for the lowest validation loss and save the weights
    if val_mean_loss1 < best_val_loss1:
        best_val_loss1 = val_mean_loss1
        best_weights_student1 = student1.get_weights()
        print(f"New best validation loss of student1: {val_mean_loss1}. Saving the weights")
        logging.info(f"New best validation loss: {val_mean_loss1}. Saving the weights")
        # Save the best weights
        student1.set_weights(best_weights_student1)
        student1.save_weights('./weights/online_distill/student1_best_weights.h5')
        
    if val_mean_loss2 < best_val_loss2:
        best_val_loss2 = val_mean_loss2
        best_weights_student2 = student2.get_weights()
        print(f"New best validation loss: {val_mean_loss2}. Saving the weights")
        logging.info(f"New best validation loss: {val_mean_loss2}. Saving the weights")
        # Save the best weights
        student2.set_weights(best_weights_student2)
        student2.save_weights('./weights/online_distill/student2_best_weights.h5')

        

