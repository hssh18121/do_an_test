import tensorflow as tf
from tensorflow.keras import layers, models, initializers, regularizers
import numpy as np

class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if self.drop_prob == 0. or not training:
            return x
        keep_prob = 1.0 - self.drop_prob
        input_shape = tf.shape(x)
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(input_shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(x, keep_prob) * binary_tensor
        return output

class Block(tf.keras.layers.Layer):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        self.dwconv = layers.DepthwiseConv2D(kernel_size=7, padding='same')
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pwconv1 = layers.Dense(4 * dim)
        self.act = layers.Activation('gelu')
        self.pwconv2 = layers.Dense(dim)
        self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim)), trainable=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity

    def call(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + self.drop_path(x)
        return x

class ConvNeXt(tf.keras.Model):
    def __init__(self, in_chans=3, num_classes=1000, to_cls=False, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super(ConvNeXt, self).__init__()
        self.dims = dims
        self.to_cls = to_cls

        self.downsample_layers = [models.Sequential([
            layers.Conv2D(dims[0], kernel_size=4, strides=4),
            layers.LayerNormalization(epsilon=1e-6)
        ])]
        for i in range(3):
            downsample_layer = models.Sequential([
                layers.LayerNormalization(epsilon=1e-6),
                layers.Conv2D(dims[i + 1], kernel_size=2, strides=2)
            ])
            self.downsample_layers.append(downsample_layer)

        self.stages = []
        dp_rates = np.linspace(0, drop_path_rate, sum(depths))
        cur = 0
        for i in range(4):
            stage = models.Sequential([
                Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(num_classes, kernel_initializer=initializers.TruncatedNormal(stddev=0.02))
        self.head.build((None, dims[-1]))
        self.head.bias.assign(tf.zeros_like(self.head.bias) * head_init_scale)
        self.head.kernel.assign(tf.random.truncated_normal(shape=self.head.kernel.shape, stddev=0.02) * head_init_scale)

    def call(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        if not self.to_cls:
            return features
        else:
            x = self.norm(tf.reduce_mean(x, axis=[1, 2]))
            x = self.head(x)
            return x

def get_convnext(model_name='convnext_tiny', pretrained=True, in_chans=3, scale=4, to_cls=False, **kwargs):
    model_params = {
        'convnext_tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
        'convnext_small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768]},
        'convnext_base_1k': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
        'convnext_base_22k': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
        'convnext_large_1k': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
        'convnext_large_22k': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
        'convnext_xlarge_22k': {'depths': [3, 3, 27, 3], 'dims': [256, 512, 1024, 2048]}
    }

    params = model_params[model_name]
    model = ConvNeXt(in_chans=in_chans, to_cls=to_cls, **params, **kwargs)

    if pretrained:
        # Load pretrained weights if available (this part needs to be adapted to load actual weights)
        pass

    if in_chans != 3:
        stem = models.Sequential([
            layers.Conv2D(params['dims'][0], kernel_size=scale, strides=scale),
            layers.LayerNormalization(epsilon=1e-6)
        ])
        model.downsample_layers[0] = stem
    if to_cls:
        model.head = layers.Dense(1, kernel_initializer=initializers.TruncatedNormal(stddev=0.02))

    return model

if __name__ == '__main__':
    image_size = 224
    in_chans = 3
    model = get_convnext(model_name='convnext_tiny', pretrained=False, in_chans=in_chans)
    print(model.summary())
    img = tf.random.normal([1, image_size, image_size, in_chans])
    features = model(img)
    print([feature.shape for feature in features], model.dims)
