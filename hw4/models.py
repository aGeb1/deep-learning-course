# These are the models described in the articles themselves
from typing import List

import tensorflow as tf

from layers import *


class WideResNet(tf.Module):
    """arXiv:1605.07146v4"""

    def __init__(self, N, k, number_of_categories, dropout_rate):
        self.layers = []
        depths = [16, 16 * k, 32 * k, 64 * k]

        self.layers.append(GroupNormalization(4))
        self.layers.append(Conv2d(3, 3, 3, 16))

        for i in range(1, 4):
            for j in range(N):
                if j == 0:
                    input_channels = depths[i - 1]
                    if i == 1:
                        h = IdentityMapping(
                            depths[i - 1], depths[i], pooling=False
                        )
                        stride = 1
                    else:
                        h = IdentityMapping(depths[i - 1], depths[i])
                        stride = 2
                else:
                    input_channels = depths[i]
                    h = tf.identity
                    stride = 1

                self.layers.append(
                    Residual(
                        [
                            GroupNormalization(4),
                            tf.nn.relu,
                            Conv2d(3, 3, input_channels, depths[i]),
                            lambda x: tf.nn.dropout(
                                x, rate=dropout_rate, seed=472
                            ),
                            GroupNormalization(4),
                            tf.nn.relu,
                            Conv2d(3, 3, depths[i], depths[i], stride),
                        ],
                        h,
                    )
                )

        self.layers.append(GroupNormalization(4))
        self.layers.append(tf.nn.relu)

        self.layers.append(lambda x: tf.nn.avg_pool2d(x, 8, 1, "VALID"))
        self.layers.append(lambda x: tf.reshape(x, [tf.shape(x)[0], -1]))

        self.layers.append(Linear(depths[3], number_of_categories))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameter_count(self):
        """Copied from Teams"""
        return tf.math.add_n(
            [tf.math.reduce_prod(var.shape) for var in self.trainable_variables]
        ).numpy()
