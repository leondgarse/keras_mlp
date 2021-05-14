from tensorflow import keras


def channel_affine(inputs, use_bias=True, weight_init_value=1, name=""):
    ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
    nn = keras.backend.expand_dims(inputs, 1)
    nn = keras.layers.DepthwiseConv2D(1, depthwise_initializer=ww_init, use_bias=use_bias, name=name + "affine")(nn)
    return keras.backend.squeeze(nn, 1)


def mlp_block(inputs, mlp_dim, activation="gelu", name=None):
    affine_inputs = channel_affine(inputs, use_bias=True, name=name + "1_")
    nn = keras.layers.Permute((2, 1), name=name + "permute_1")(affine_inputs)
    nn = keras.layers.Dense(nn.shape[-1], name=name + "dense_1")(nn)
    nn = keras.layers.Permute((2, 1), name=name + "permute_2")(nn)
    nn = channel_affine(nn, use_bias=False, name=name + "1_gamma_")
    skip_conn = keras.layers.Add(name=name + "add_1")([nn, affine_inputs])

    affine_skip = channel_affine(skip_conn, use_bias=True, name=name + "2_")
    nn = keras.layers.Dense(mlp_dim, name=name + "dense_2_1")(affine_skip)
    nn = keras.layers.Activation(activation, name=name + "gelu")(nn)
    nn = keras.layers.Dense(inputs.shape[-1], name=name + "dense_2_2")(nn)
    nn = channel_affine(nn, use_bias=False, name=name + "2_gamma_")
    nn = keras.layers.Add(name=name + "add_2")([nn, affine_skip])
    return nn


def ResMLP(
    input_shape,
    num_blocks,
    patch_size,
    hidden_dim,
    mlp_dim,
    num_classes=0,
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
        nn = channel_affine(nn, name="post")
        nn = keras.layers.GlobalAveragePooling1D()(nn)  # tf.reduce_mean(nn, axis=1)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)
    return keras.Model(inputs, nn, name=model_name)


def ResMLP12(input_shape=(224, 224, 3), hidden_dim=384, num_classes=0, classifier_activation="softmax", **kwargs):
    return ResMLP(
        input_shape, 12, 16, hidden_dim, hidden_dim * 4, num_classes, classifier_activation, "ResMLP12", **kwargs
    )


def ResMLP24(input_shape=(224, 224, 3), hidden_dim=384, num_classes=0, classifier_activation="softmax", **kwargs):
    return ResMLP(
        input_shape, 24, 16, hidden_dim, hidden_dim * 4, num_classes, classifier_activation, "ResMLP24", **kwargs
    )


def ResMLP36(input_shape=(224, 224, 3), hidden_dim=384, num_classes=0, classifier_activation="softmax", **kwargs):
    return ResMLP(
        input_shape, 36, 16, hidden_dim, hidden_dim * 4, num_classes, classifier_activation, "ResMLP36", **kwargs
    )
