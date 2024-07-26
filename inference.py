import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm

from sklearn.preprocessing import MinMaxScaler

img = cv2.imread("./infer_image/test_rgb_img.png", 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

from Upernet.models import upernet_convnext_tiny_org

# size of patches
patch_size = 256

# Number of classes 
n_classes = 5

checkpoint_path = "./weights/augmented_upernet_tiny_org_with_pretrain_converge/cp.weights.h5"
model = upernet_convnext_tiny_org.UPerNet(input_shape = (256, 256,3), num_classes = 5)
model.compile('Adam', loss=sm.losses.dice_loss, metrics=[sm.metrics.iou_score],)
model.load_weights(checkpoint_path)

large_img = Image.fromarray(img)
large_img = np.array(large_img)     

import time

# Start the timer
start_time = time.time()

# Predict patch by patch with no smooth blending
SIZE_X = (img.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
SIZE_Y = (img.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size

large_img = Image.fromarray(img)
large_img = large_img.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
large_img = np.array(large_img)
print(large_img.shape)

patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap
patches_img = patches_img[:, :, 0, :, :, :]
print(patches_img.shape)

patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, :, :, :]
        
        # Use MinMaxScaler instead of just dividing by 255.
        # single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0, :, :]
        
        patched_prediction.append(pred)

patched_prediction = np.array(patched_prediction)
print(patched_prediction.shape)  # Should be (number_of_patches, patch_height, patch_width)

# Reshape to 4D array
patched_prediction = patched_prediction.reshape((patches_img.shape[0], patches_img.shape[1], patch_size, patch_size))

# Unpatchify to get the large image
unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

# End the timer
end_time = time.time()

# Calculate and print the total time taken
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


# Assuming unpatched_prediction is already defined
# Define a color map for the pixel values
color_map = {
    0: [0, 0, 0],         # Black for background
    1: [0, 255, 0],       # Green for tree
    2: [0, 0, 255],       # Blue for water
    3: [255, 255, 0],     # Yellow for road
    4: [255, 0, 0]        # Red for building
}

# Get the shape of the unpatched_prediction
height, width = unpatched_prediction.shape

# Create an empty array for the colored image
colored_image = np.zeros((height, width, 3), dtype=np.uint8)

# Apply the color map
for value, color in color_map.items():
    colored_image[unpatched_prediction == value] = color

# Convert the numpy array to a PIL Image
image = Image.fromarray(colored_image)

# Specify the folder and filename
folder = 'infer_image'
filename = 'colored_prediction_2.jpg'
filepath = os.path.join(folder, filename)

# Ensure the directory exists
os.makedirs(folder, exist_ok=True)

# Save the image
image.save(filepath)

print(f'Image saved to {filepath}')

