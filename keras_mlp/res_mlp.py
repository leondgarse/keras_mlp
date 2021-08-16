from tensorflow import keras

class ChannelAffine(keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, **kwargs):
        super(ChannelAffine, self).__init__(**kwargs)
        self.use_bias, self.weight_init_value = use_bias, weight_init_value
        self.ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        self.ww = self.add_weight(name="weight", shape=(input_shape[-1]), initializer=self.ww_init, trainable=True)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=(input_shape[-1]), initializer=self.bb_init, trainable=True)
        super(ChannelAffine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.use_bias:
            return inputs * self.ww + self.bb
        else:
            return inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ChannelAffine, self).get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# NOT using
def channel_affine(inputs, use_bias=True, weight_init_value=1, name=""):
    ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
    nn = keras.backend.expand_dims(inputs, 1)
    nn = keras.layers.DepthwiseConv2D(1, depthwise_initializer=ww_init, use_bias=use_bias, name=name)(nn)
    return keras.backend.squeeze(nn, 1)


def mlp_block(inputs, mlp_dim, activation="gelu", name=None):
    affine_inputs = ChannelAffine(use_bias=True, name=name + "affine_1")(inputs)
    nn = keras.layers.Permute((2, 1), name=name + "permute_1")(affine_inputs)
    nn = keras.layers.Dense(nn.shape[-1], name=name + "dense_1")(nn)
    nn = keras.layers.Permute((2, 1), name=name + "permute_2")(nn)
    nn = ChannelAffine(use_bias=False, name=name + "gamma_1")(nn)
    skip_conn = keras.layers.Add(name=name + "add_1")([nn, affine_inputs])

    affine_skip = ChannelAffine(use_bias=True, name=name + "affine_2")(skip_conn)
    nn = keras.layers.Dense(mlp_dim, name=name + "dense_2_1")(affine_skip)
    nn = keras.layers.Activation(activation, name=name + "gelu")(nn)
    nn = keras.layers.Dense(inputs.shape[-1], name=name + "dense_2_2")(nn)
    nn = ChannelAffine(use_bias=False, name=name + "gamma_2")(nn)
    nn = keras.layers.Add(name=name + "add_2")([nn, affine_skip])
    return nn


def ResMLP(
    input_shape,
    num_blocks,
    patch_size,
    hidden_dim,
    mlp_dim,
    num_classes=0,
    dropout=0,
    classifier_activation="softmax",
    model_name="MlpMixer",
):
    inputs = keras.Input(input_shape)
    nn = keras.layers.Conv2D(hidden_dim, kernel_size=patch_size, strides=patch_size, padding="same", name="projector")(inputs)
    nn = keras.layers.Reshape([-1, hidden_dim])(nn)

    for ii in range(num_blocks):
        name = "_".join(["mlp_block", str(ii + 1), ""])
        nn = mlp_block(nn, mlp_dim=mlp_dim, name=name)

    if num_classes > 0:
        nn = ChannelAffine(name="post")(nn)
        nn = keras.layers.GlobalAveragePooling1D()(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)
    return keras.Model(inputs, nn, name=model_name)


def ResMLP12(input_shape=(224, 224, 3), hidden_dim=384, num_classes=0, dropout=0, classifier_activation="softmax", **kwargs):
    return ResMLP(
        input_shape, 12, 16, hidden_dim, hidden_dim * 4, num_classes, dropout, classifier_activation, "ResMLP12", **kwargs
    )


def ResMLP24(input_shape=(224, 224, 3), hidden_dim=384, num_classes=0, dropout=0, classifier_activation="softmax", **kwargs):
    return ResMLP(
        input_shape, 24, 16, hidden_dim, hidden_dim * 4, num_classes, dropout, classifier_activation, "ResMLP24", **kwargs
    )


def ResMLP36(input_shape=(224, 224, 3), hidden_dim=384, num_classes=0, dropout=0, classifier_activation="softmax", **kwargs):
    return ResMLP(
        input_shape, 36, 16, hidden_dim, hidden_dim * 4, num_classes, dropout, classifier_activation, "ResMLP36", **kwargs
    )
