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


class SirenLayer(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        omega_0=30.0,
        c=6.0,
        first=False,
        bias=True,
    ):
        rng = tf.random.get_global_generator()

        if first:
            radius = 1.0 / num_inputs
        else:
            radius = tf.math.sqrt(c / num_inputs) / omega_0

        self.w = tf.Variable(
            rng.uniform(
                shape=[num_inputs, num_outputs], minval=-radius, maxval=radius
            ),
            trainable=True,
            name="SirenLayer/w",
        )

        self.omega_0 = omega_0
        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="SirenLayer/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return tf.sin(self.omega_0 * z)


class Siren(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        output_activation=tf.identity,
    ):
        self.num_hidden_layers = num_hidden_layers + 1
        self.output_activation = output_activation

        if num_hidden_layers == 0:
            self.layers = [SirenLayer(num_inputs, num_outputs, first=True)]
        else:
            self.layers = []
            self.layers.append(
                SirenLayer(num_inputs, hidden_layer_width, first=True)
            )
            for _ in range(num_hidden_layers):
                self.layers.append(
                    SirenLayer(hidden_layer_width, hidden_layer_width)
                )
            self.layers.append(Linear(hidden_layer_width, num_outputs))

    def __call__(self, x):
        for i in range(self.num_hidden_layers):
            x = self.layers[i](x)
        return self.output_activation(self.layers[self.num_hidden_layers](x))
