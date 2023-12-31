from typing import Tuple, Union

import tensorflow as tf


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z


class Conv2d(tf.Module):
    def __init__(
        self,
        filter_height: int,
        filter_width: int,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: str = "SAME",
        activation=tf.nn.relu,
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
            name="Conv2d/kernel",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(shape=[out_channels]),
                trainable=True,
                name="Conv2d/b",
            )

    def __call__(self, x):
        y = tf.nn.conv2d(x, self.kernel, self.stride, self.padding)

        if self.bias:
            y = tf.nn.bias_add(y, self.b)

        return self.activation(y)


class Residual(tf.Module):
    def __init__(self, F, h=tf.identity):
        self.F = F
        self.h = h

    def __call__(self, x):
        y = x

        for layer in self.F:
            y = layer(y)

        y += self.h(x)
        return y


class IdentityMapping(tf.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        self.conv2d = Conv2d(
            1,
            1,
            in_channels,
            out_channels,
            activation=tf.identity,
            bias=False,
        )
        self.pooling = pooling

    def __call__(self, x):
        y = x
        if self.pooling:
            y = tf.nn.max_pool2d(x, 2, 2, "VALID")
        y = self.conv2d(y)

        return y


class GroupNormalization(tf.Module):
    def __init__(self, G, eps=1e-5):
        self.beta = tf.Variable(0.0, True, name="GroupNorm/beta")
        self.gamma = tf.Variable(1.0, True, name="GroupNorm/gamma")
        self.G = G
        self.eps = eps

    def __call__(self, x):
        """Code copied from arXiv:1803.08494v3"""
        N, C, H, W = x.shape
        x = tf.reshape(x, [N, self.G, C // self.G, H, W])

        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)

        x = tf.reshape(x, [N, C, H, W])

        return x * self.gamma + self.beta
