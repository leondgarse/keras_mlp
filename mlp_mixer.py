from tensorflow import keras


def mlp_block(inputs, hidden_dim, activation="gelu", name=None):
    nn = keras.layers.Dense(hidden_dim, name=name + "dense_1")(inputs)
    nn = keras.layers.Activation(activation, name=name + "gelu")(nn)
    nn = keras.layers.Dense(inputs.shape[-1], name=name + "dense_2")(nn)
    return nn


def mixer_block(inputs, tokens_mlp_dim, channels_mlp_dim=None, activation=None, name=None):
    nn = keras.layers.LayerNormalization(axis=1, name=name + "layer_norm_1")(inputs)
    nn = keras.layers.Permute((2, 1), name=name + "permute_1")(nn)
    nn = mlp_block(nn, tokens_mlp_dim, activation, name=name + "mlp_1_")
    nn = keras.layers.Permute((2, 1), name=name + "permute_2")(nn)
    skip_conn = keras.layers.Add(name=name + "add_1")([nn, inputs])

    nn = keras.layers.LayerNormalization(axis=1, name=name + "layer_norm_2")(skip_conn)
    nn = mlp_block(nn, channels_mlp_dim, activation, name=name + "mlp_2_")
    return keras.layers.Add(name=name + "add_2")([nn, skip_conn])


def MlpMixerModel(
    input_shape,
    num_blocks,
    patch_size,
    hidden_dim,
    tokens_mlp_dim,
    channels_mlp_dim,
    num_classes=0,
    classifier_activation="softmax",
    model_name="MlpMixer",
):
    inputs = keras.Input(input_shape)
    nn = keras.layers.Conv2D(hidden_dim, kernel_size=patch_size, strides=patch_size, padding="same", name="projector")(inputs)
    nn = keras.layers.Reshape([-1, hidden_dim])(nn)

    for ii in range(num_blocks):
        name = "_".join(["mixer_block", str(ii + 1), ""])
        nn = mixer_block(nn, tokens_mlp_dim=tokens_mlp_dim, channels_mlp_dim=channels_mlp_dim, name=name)

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling1D()(nn)  # tf.reduce_mean(nn, axis=1)
        nn = keras.layers.LayerNormalization(name="pre_head_layer_norm")(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)
    return keras.Model(inputs, nn, name=model_name)


def MlpMixerModel_S32(input_shape=(224, 224, 3), num_classes=0, classifier_activation="softmax", **kwargs):
    return MlpMixerModel(input_shape, 8, 32, 512, 256, 2048, num_classes, classifier_activation, "MlpMixerModel_S32", **kwargs)


def MlpMixerModel_S16(input_shape=(224, 224, 3), num_classes=0, classifier_activation="softmax", **kwargs):
    return MlpMixerModel(input_shape, 8, 16, 512, 256, 2048, num_classes, classifier_activation, "MlpMixerModel_S16", **kwargs)


def MlpMixerModel_B32(input_shape=(224, 224, 3), num_classes=0, classifier_activation="softmax", **kwargs):
    return MlpMixerModel(input_shape, 12, 32, 768, 384, 3072, num_classes, classifier_activation, "MlpMixerModel_B32", **kwargs)


def MlpMixerModel_B16(input_shape=(224, 224, 3), num_classes=0, classifier_activation="softmax", **kwargs):
    return MlpMixerModel(input_shape, 12, 16, 768, 384, 3072, num_classes, classifier_activation, "MlpMixerModel_B16", **kwargs)


def MlpMixerModel_L32(input_shape=(224, 224, 3), num_classes=0, classifier_activation="softmax", **kwargs):
    return MlpMixerModel(
        input_shape, 24, 32, 1024, 512, 4096, num_classes, classifier_activation, "MlpMixerModel_L32", **kwargs
    )


def MlpMixerModel_L16(input_shape=(224, 224, 3), num_classes=0, classifier_activation="softmax", **kwargs):
    return MlpMixerModel(
        input_shape, 24, 16, 1024, 512, 4096, num_classes, classifier_activation, "MlpMixerModel_L16", **kwargs
    )


def MlpMixerModel_H14(input_shape=(224, 224, 3), num_classes=0, classifier_activation="softmax", **kwargs):
    return MlpMixerModel(
        input_shape, 32, 14, 1280, 640, 5120, num_classes, classifier_activation, "MlpMixerModel_H14", **kwargs
    )
