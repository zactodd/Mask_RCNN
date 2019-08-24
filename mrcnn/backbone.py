import keras.layers as KL
from mrcnn.model import BatchNorm
import math
import itertools


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet34", ]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    stage_checkpoints = [x]

    # Stages 2 to 5
    stages = RESNET_STRUCTURES[architecture] if stage5 else RESNET_STRUCTURES[architecture][:-1]
    for i, stage in enumerate(stages):
        filters = stage["filters"]
        layers = stage["layers"]

        first, *names = block_names(layers)
        x = conv_block(x, filters, stage=i, block=first, strides=(1, 1), train_bn=train_bn)
        for name in names:
            x = conv_block(x, filters, stage=i, block=name, train_bn=train_bn)
        stage_checkpoints.append(x)

    return stage_checkpoints if stage5 else stage_checkpoints + [None]


def conv_block(input_tensor, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = None
    block_suffixes = block_names(len(filters))
    for *k, f, n in zip(filters, block_suffixes):
        kernel = tuple(k)
        cn_name = "{}2{}".format(conv_name_base, n)
        bn_name = "{}2{}".format(bn_name_base, n)

        x = KL.Conv2D(f, kernel, padding='same', strides=strides, name=cn_name, use_bias=use_bias)(input_tensor)
        x = BatchNorm(name=bn_name)(x, training=train_bn)
        x = KL.Activation('relu')(x)

    *k, f = filters[-1]
    shortcut = KL.Conv2D(f, tuple(k), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def identity_block(input_tensor, filters, stage, block, use_bias=True, train_bn=True):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = None
    block_suffixes = block_names(len(filters))
    for *k, f, n in zip(filters, block_suffixes):
        kernel = tuple(k)
        cn_name = "{}2{}".format(conv_name_base, n)
        bn_name = "{}2{}".format(bn_name_base, n)

        x = KL.Conv2D(f, kernel, padding='same', name=cn_name, use_bias=use_bias)(input_tensor)
        x = BatchNorm(name=bn_name)(x, training=train_bn)
        x = KL.Activation('relu')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


# TODO improve naming
def block_names(n):
    letters = math.ceil(math.log(n, 26))
    names = []
    for i, nums in enumerate(itertools.product(range(26), repeat=letters)):
        if i > n - 1:
            break
        else:
            names.append("".join(chr(num + 97) for num in nums))
    return names


RESNET_STRUCTURES = {
    "resent18":
        [
            {
                "layers": 2,
                "filters": [(3, 3, 64), (3, 3, 64)]
            },
            {
                "layers": 2,
                "filters": [(3, 3, 128), (3, 3, 128)]
            },
            {
                "layers": 2,
                "filters": [(3, 3, 256), (3, 3, 256)]
            },
            {
                "layers": 2,
                "filters": [(3, 3, 512), (3, 3, 512)]
            }
        ]
}