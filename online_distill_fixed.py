import os

# Set environment variable for segmentation models
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import pandas as pd


import sys
# Add the parent directory of the 'models' directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Upernet')))

import os
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import albumentations as A
from models import upernet_convnext_improved
import logging

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

# Split the dataset into two parts
first_half_dataset = combined_train_dataset.take(midpoint)
second_half_dataset = combined_train_dataset.skip(midpoint)

# Batch and normalize the datasets
augmented_train_dataset = second_half_dataset.batch(4).map(_normalize)
train_dataset = first_half_dataset.batch(4).map(_normalize)
val_dataset = val_dataset.batch(4).map(_normalize)

# Setup logging
logging.basicConfig(filename='./result/online_distill/training_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

pretrain_checkpoint_path = "./weights/imgnet_augmented_upernet_convnext_improved/cp.weights.h5"

# Build two student models
input_shape = (256, 256, 3)
num_classes = 5
student1 = upernet_convnext_improved.UPerNet(input_shape=input_shape, num_classes=num_classes, name_prefix="convnext1")
student2 = upernet_convnext_improved.UPerNet(input_shape=input_shape, num_classes=num_classes, name_prefix="convnext2")

student1.load_weights(pretrain_checkpoint_path)
student2.load_weights(pretrain_checkpoint_path)

class KLDivergenceLoss(tf.keras.losses.Loss):
    def __init__(self, name="kl_divergence_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.keras.backend.clip(y_true, 1e-10, 1)
        y_pred = tf.keras.backend.clip(y_pred, 1e-10, 1)
        return tf.keras.backend.sum(y_true * tf.keras.backend.log(y_true / y_pred), axis=-1)

dice_loss = sm.losses.DiceLoss() 
focal_loss = sm.losses.CategoricalFocalLoss()
kl_loss = KLDivergenceLoss()

# Attention aggregation module
class AttentionAggregation(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionAggregation, self).__init__()
        self.shared_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, inputs):
        F1, F2, F3 = inputs
        F1_weight = self.shared_mlp(F1)
        F2_weight = self.shared_mlp(F2)
        F3_weight = self.shared_mlp(F3)
        weights = tf.concat([F1_weight, F2_weight, F3_weight], axis=-1)
        return tf.nn.softmax(weights, axis=-1)

# Custom combined loss function
def custom_combined_loss(y_true, y_pred, preds1, preds2, attention_weights):
    combined_preds = (attention_weights[0] * preds1 + attention_weights[1] * preds2)
    kl_loss1 = kl_loss(y_true, preds1)
    kl_loss2 = kl_loss(y_true, preds2)
    return dice_loss(y_true, combined_preds) + (2 * focal_loss(y_true, combined_preds)) + kl_loss1 + kl_loss2

# Optimizers
optimizer1 = tf.keras.optimizers.Adam()
optimizer2 = tf.keras.optimizers.Adam()

# Define the attention aggregation layer
attention_aggregation = AttentionAggregation()

# Define a custom training loop
epochs = 10
best_val_loss = np.inf
best_weights_student1 = None
best_weights_student2 = None

for epoch in range(epochs):
    student1_epoch_loss = []
    student2_epoch_loss = []
    student1_epoch_iou = []
    student2_epoch_iou = []

    for step, ((images_orig, labels_orig), (images_aug, labels_aug)) in enumerate(zip(train_dataset, augmented_train_dataset)):
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass through student1 with original dataset
            preds1 = student1(images_orig, training=True)
            # Forward pass through student2 with augmented dataset
            preds2 = student2(images_aug, training=True)

            # Compute attention weights
            F1 = tf.reduce_max(preds1 * labels_orig, axis=[1, 2])
            F2 = tf.reduce_mean(tf.nn.relu(-(preds1 - tf.reduce_max(preds1 * labels_orig, axis=-1, keepdims=True))), axis=[1, 2])
            F3 = tf.reduce_mean(tf.nn.relu(-(preds1 - tf.reduce_max(preds1 * labels_orig, axis=-1, keepdims=True))) / (-preds1 + 1e-6), axis=[1, 2])
            attention_weights = attention_aggregation([F1, F2, F3])

            # Compute losses
            loss1 = custom_combined_loss(labels_orig, preds1, preds1, preds2, attention_weights)
            loss2 = custom_combined_loss(labels_aug, preds2, preds1, preds2, attention_weights)
            print(f"Loss1: {loss1}")
            print(f"Loss2: {loss2}")
            total_loss = loss1 + loss2

        # Compute gradients and update weights
        grads1 = tape.gradient(total_loss, student1.trainable_variables)
        grads2 = tape.gradient(total_loss, student2.trainable_variables)
        optimizer1.apply_gradients(zip(grads1, student1.trainable_variables))
        optimizer2.apply_gradients(zip(grads2, student2.trainable_variables))
        
        student1_epoch_loss.append(loss1.numpy())
        student2_epoch_loss.append(loss2.numpy())

    # Validation step
    val_losses = []
    for images, labels in val_dataset:
        preds1 = student1(images, training=False)
        preds2 = student2(images, training=False)

        F1 = tf.reduce_max(preds1 * labels, axis=[1, 2])
        F2 = tf.reduce_mean(tf.nn.relu(-(preds1 - tf.reduce_max(preds1 * labels, axis=-1, keepdims=True))), axis=[1, 2])
        F3 = tf.reduce_mean(tf.nn.relu(-(preds1 - tf.reduce_max(preds1 * labels, axis=-1, keepdims=True))) / (-preds1 + 1e-6), axis=[1, 2])
        attention_weights = attention_aggregation([F1, F2, F3])

        val_loss1 = custom_combined_loss(labels, preds1, preds1, preds2, attention_weights)
        val_loss2 = custom_combined_loss(labels, preds2, preds1, preds2, attention_weights)
        val_losses.append(val_loss1.numpy() + val_loss2.numpy())

    val_mean_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_mean_loss}")
    logging.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_mean_loss}")
    
    if val_mean_loss < best_val_loss:
        best_val_loss = val_mean_loss
        student1.save_weights('./weights/student1_best_weights.h5')
        student2.save_weights('./weights/student2_best_weights.h5')
        print("New best validation loss, saving weights.")

# Continue with the rest of your code...
