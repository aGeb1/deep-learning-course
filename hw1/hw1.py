import tensorflow as tf


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.b = tf.Variable(
            tf.zeros(shape=[1, num_outputs]),
            trainable=True,
            name="Linear/b",
        )

    def __call__(self, x):
        return x @ self.w + self.b


class BasisExpansion(tf.Module):
    def __init__(self, M):
        self.mu = tf.Variable(
            tf.ones(shape=[1, M]) / 2, trainable=True, name="Linear/mu"
        )

        self.sigma = tf.Variable(
            tf.ones(shape=[1, M]) / M, trainable=True, name="Linear/sigma"
        )

    def __call__(self, x):
        return tf.math.exp(-tf.square((x - self.mu) / self.sigma))


class Model(tf.Module):
    def __init__(self, M):
        self.linear = Linear(M, 1)
        self.basisexpansion = BasisExpansion(M)

    def __call__(self, x):
        phi = self.basisexpansion(x)
        y = self.linear(phi)
        return y


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import math

    import matplotlib.pyplot as plt
    from tqdm import trange

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    M = 6
    N = 50
    stddev = 0.1

    x = rng.uniform(shape=[N, 1])
    y = rng.normal(
        shape=[N, 1],
        mean=tf.math.sin(2 * math.pi * x),
        stddev=stddev,
    )

    model = Model(M)

    num_iters = 100
    step_size = 5e-2
    decay_rate = 0.999

    for i in trange(num_iters):
        with tf.GradientTape() as tape:
            loss = tf.math.reduce_mean(tf.square(y - model(x)))

        grads = tape.gradient(loss, model.trainable_variables)
        grad_update(step_size, model.trainable_variables, grads)

        step_size *= decay_rate

    fig, axs = plt.subplots(ncols=2)

    axs[0].plot(x, y, "o")
    a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), N)[:, tf.newaxis]
    axs[0].plot(a, tf.math.sin(2 * math.pi * a))
    axs[0].plot(a, model(a), "--")

    axs[0].set_title("Model Output")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    phi = model.basisexpansion(a)
    for i in range(M):
        axs[1].plot(a, phi[:, i])

    axs[1].set_title("Basis for Model")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    fig.savefig("plot.pdf")
