import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import convnext

def build_ppm(input_tensor, bin_sizes):
    input_shape = input_tensor.shape
    h, w = input_shape[1], input_shape[2]
    ppm_outputs = [input_tensor]

    for bin_size in bin_sizes:
        pool_size = (h // bin_size, w // bin_size)
        x = layers.AveragePooling2D(pool_size=pool_size, strides=pool_size)(input_tensor)
        x = layers.Conv2D(512, kernel_size=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D(size=pool_size, interpolation='bilinear')(x)
        ppm_outputs.append(x)

    return layers.Concatenate()(ppm_outputs)

# Helper function to create the Feature Pyramid Network (FPN)
def build_fpn(features):
    fpn_outs = []
    for i in range(len(features)):
        x = layers.Conv2D(256, kernel_size=1, padding='same')(features[i])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        fpn_outs.append(x)

    for i in range(1, len(fpn_outs)):
        fpn_outs[i] = layers.UpSampling2D(size=(2**i, 2**i), interpolation='bilinear')(fpn_outs[i])

    return layers.Concatenate()(fpn_outs)

# Define the UPerNet model
def UPerNet(input_shape=(512, 512, 3), num_classes=21):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Backbone network (ConvNeXt Tiny)
    backbone = convnext.ConvNeXtTiny(include_top=False, input_tensor=inputs, weights='imagenet')
    # Select appropriate layers for FPN
    C2 = backbone.get_layer("convnext_tiny_stage_0_block_2_layernorm").output
    C3 = backbone.get_layer("convnext_tiny_stage_1_block_2_layernorm").output
    C4 = backbone.get_layer("convnext_tiny_stage_2_block_8_layernorm").output
    C5 = backbone.get_layer("convnext_tiny_stage_3_block_2_layernorm").output

    # Pyramid Pooling Module (PPM) on the last feature map (C5)
    ppm = build_ppm(C5, bin_sizes=[1, 2, 4, 6])
    
    # Feature Pyramid Network (FPN) on the intermediate feature maps (C3, C4, and PPM output)
    fpn = build_fpn([C2, C3, C4, ppm])
    
    # Final segmentation head
    x = layers.Conv2D(256, kernel_size=3, padding='same')(fpn)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_classes, kernel_size=1)(x)
    x_shape = x.shape
    x = layers.UpSampling2D(size=(input_shape[0] // x_shape[1], input_shape[1] // x_shape[2]), interpolation='bilinear')(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs=inputs, outputs=x)
    
    return model
