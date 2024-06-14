import tensorflow as tf
from tensorflow.keras import layers, models, initializers

def ConvBlock(filters, kernel_size, strides, use_bias=False):
    return models.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias, 
                      kernel_initializer=initializers.HeNormal()),
        layers.BatchNormalization(),
        layers.ReLU()
    ])

def ResidualBlock(filters, strides=1):
    def block(x):
        shortcut = x
        x = ConvBlock(filters, 3, strides)(x)
        x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False, 
                          kernel_initializer=initializers.HeNormal())(x)
        x = layers.BatchNormalization()(x)
        
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = ConvBlock(filters, 1, strides)(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x
    return block

def EfficientResNet(input_shape, num_blocks_list, filters_list):
    inputs = layers.Input(shape=input_shape)
    x = ConvBlock(filters_list[0], 7, 2)(inputs)
    x = layers.MaxPooling2D(3, 2, padding='same')(x)
    
    for num_blocks, filters in zip(num_blocks_list, filters_list):
        for i in range(num_blocks):
            strides = 1 if i == 0 else 1
            x = ResidualBlock(filters, strides)(x)
    
    model = models.Model(inputs, x)
    return model

def AttentionBoostingGate(x):
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, x.shape[-1]))(avg_pool)
    avg_pool = layers.Conv2D(x.shape[-1], 1, activation='sigmoid')(avg_pool)
    return layers.Multiply()([x, avg_pool])

def AttentionFusionNetwork(encoder_output):
    attention = AttentionBoostingGate(encoder_output)
    fusion = layers.Concatenate()([encoder_output, attention])
    return fusion

def SERNetFormer(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    efficient_resnet = EfficientResNet(input_shape, num_blocks_list=[3, 4, 6, 3], filters_list=[64, 128, 256, 512])
    encoder_output = efficient_resnet(inputs)
    
    # Attention Fusion Network
    fusion_output = AttentionFusionNetwork(encoder_output)
    
    # Decoder
    x = ConvBlock(256, 3, 1)(fusion_output)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = ConvBlock(128, 3, 1)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = ConvBlock(64, 3, 1)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = ConvBlock(32, 3, 1)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# Example usage
input_shape = (256, 256, 3)
num_classes = 5
model = SERNetFormer(input_shape, num_classes)
model.summary()
