# These are the components to make larger models

from typing import Tuple, Union

import tensorflow as tf

from linear import Linear


class Conv2d(tf.Module):
    def __init__(
        self,
        filter_height: int,
        filter_width: int,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: str = "SAME",
        activation=tf.identity,
        bias=True,
    ):
        rng = tf.random.get_global_generator()

        self.stride = stride
        self.padding = padding
        self.activation = activation

        n_hat = filter_height * filter_width * out_channels
        stddev = tf.sqrt(2 / n_hat)

        self.kernel = tf.Variable(
            rng.normal(
                shape=[filter_height, filter_width, in_channels, out_channels],
                stddev=stddev,
            ),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(shape=[out_channels]),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        y = tf.nn.conv2d(x, self.kernel, self.stride, self.padding)

        if self.bias:
            y = tf.nn.bias_add(y, self.b)

        return self.activation(y)


class FullyConnected(tf.Module):
    def __init__(self, num_inputs, num_outputs, activation=tf.identity):
        self.linear = Linear(num_inputs, num_outputs)
        self.activation = activation

    def __call__(self, x):
        y = self.linear(x)
        return self.activation(y)
