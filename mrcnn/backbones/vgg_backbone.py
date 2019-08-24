import keras.layers as KL


def vgg_graph(input_image, architecture, include_top=True, pooling=None, classes=1000):
    assert architecture in VGG_STRUCTURES
    assert pooling in ["avg", "max", None]

    x = input_image
    stages = VGG_STRUCTURES["VGG11"]
    stage_checkpoints = []
    for bl, stage in enumerate(stages):
        *k, f = stage["filters"]
        layers = stage["layers"]
        for cv, l in enumerate(range(layers)):
            cv_name = "block{}_pool{}".format(bl, cv)
            x = KL.Conv2D(f, tuple(k), activation='relu', padding='same', name=cv_name)(x)
        mp_name = "block{}_pool".format(bl)
        x = KL.MaxPooling2D((2, 2), strides=(2, 2), name=mp_name)(x)
        stage_checkpoints.append(x)

    if include_top:
        # Classification block
        x = KL.Flatten(name='flatten')(x)
        x = KL.Dense(4096, activation='relu', name='fc1')(x)
        x = KL.Dense(4096, activation='relu', name='fc2')(x)
        x = KL.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = KL.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = KL.GlobalMaxPooling2D()(x)
    stage_checkpoints.append(x)
    return stage_checkpoints


VGG_STRUCTURES = {
    "VGG11": [
                {
                    "layers": 1,
                    "filters": [(3, 3, 64)]
                },
                {
                    "layers": 1,
                    "filters": [(3, 3, 128)]
                },
                {
                    "layers": 2,
                    "filters": [(3, 3, 256)]
                },
                {
                    "layers": 2,
                    "filters": [(3, 3, 512)]
                },
                {
                    "layers": 2,
                    "filters": [(3, 3, 512)]
                }
        ],
    "VGG16": [
                {
                    "layers": 2,
                    "filters": [(3, 3, 64)]
                },
                {
                    "layers": 2,
                    "filters": [(3, 3, 128)]
                },
                {
                    "layers": 3,
                    "filters": [(3, 3, 256)]
                },
                {
                    "layers": 3,
                    "filters": [(3, 3, 512)]
                },
                {
                    "layers": 3,
                    "filters": [(3, 3, 512)]
                }
        ],
    "VGG19": [
                {
                    "layers": 2,
                    "filters": [(3, 3, 64)]
                },
                {
                    "layers": 2,
                    "filters": [(3, 3, 128)]
                },
                {
                    "layers": 4,
                    "filters": [(3, 3, 256)]
                },
                {
                    "layers": 4,
                    "filters": [(3, 3, 512)]
                },
                {
                    "layers": 4,
                    "filters": [(3, 3, 512)]
                }
        ]
}
