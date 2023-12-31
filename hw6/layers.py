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


class FFN_ReLU(tf.Module):
    def __init__(self, d_model, d_ff):
        self.W1 = Linear(d_model, d_ff, bias=False)
        self.W2 = Linear(d_ff, d_model, bias=False)

    def __call__(self, x):
        return self.W2(tf.nn.relu(self.W1(x)))


class SwiGLU(tf.Module):
    """SwiGLU Based FFN (https://arxiv.org/pdf/2002.05202.pdf)"""

    def __init__(self, d_model, d_ff):
        self.W = Linear(d_model, d_ff, bias=False)
        self.V = Linear(d_model, d_ff, bias=False)
        self.W2 = Linear(d_ff, d_model, bias=False)

    def __call__(self, x):
        return self.W2(tf.nn.silu(self.W(x)) * self.V(x))


class LayerNormalization(tf.Module):
    def __init__(self, eps=1e-5):
        self.beta = tf.Variable(0.0, True, name="GroupNorm/beta")
        self.gamma = tf.Variable(1.0, True, name="GroupNorm/gamma")
        self.eps = eps

    def __call__(self, x):
        mean, var = tf.nn.moments(x, [2], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        return x * self.gamma + self.beta
