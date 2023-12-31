import numpy as np
import tensorflow as tf

from linear import Linear, grad_update


class MLP(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        if num_hidden_layers == 0:
            self.layers = [Linear(num_inputs, num_outputs)]
        else:
            self.layers = []
            self.layers.append(Linear(num_inputs, hidden_layer_width))
            for _ in range(num_hidden_layers - 1):
                self.layers.append(
                    Linear(hidden_layer_width, hidden_layer_width)
                )
            self.layers.append(Linear(hidden_layer_width, num_outputs))

    def __call__(self, x):
        for i in range(self.num_hidden_layers):
            x = self.hidden_activation(self.layers[i](x))
        return self.output_activation(self.layers[self.num_hidden_layers](x))


def archimedean_spiral(
    num_samples, noise_stddev, starting_angle, angle_range, rng
):
    a = rng.uniform(0, angle_range, [num_samples, 1])
    a_noise = rng.normal(a, noise_stddev)
    return np.hstack(
        (
            -a * np.cos(a_noise + starting_angle),
            a * np.sin(a_noise + starting_angle),
        )
    )


def l2_regularization(variables):
    complexity = 0
    for var in variables:
        complexity = complexity + tf.nn.l2_loss(var)
    return complexity


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml
    from sklearn.inspection import DecisionBoundaryDisplay
    from tqdm import trange

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument(
        "-c", "--config", type=Path, default=Path("config.yaml")
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = np.random.default_rng(seed=472)

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(472)

    num_samples = config["data"]["num_samples"]
    noise_stddev = config["data"]["noise_stddev"]

    spiral1 = archimedean_spiral(num_samples, noise_stddev, 0, 3.5 * np.pi, rng)
    spiral2 = archimedean_spiral(
        num_samples, noise_stddev, np.pi, 3.5 * np.pi, rng
    )

    x = np.vstack((spiral1, spiral2))
    y = np.vstack((np.ones((num_samples, 1)), np.zeros((num_samples, 1))))

    num_hidden_layers = config["model"]["num_hidden_layers"]
    hidden_layer_width = config["model"]["hidden_layer_width"]
    hidden_activation = eval(config["model"]["hidden_activation"])
    output_activation = eval(config["model"]["output_activation"])

    mlp = MLP(
        2,
        1,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation,
        output_activation,
    )

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    regularization_rate = config["learning"]["regularization_rate"]

    for i in trange(num_iters):
        with tf.GradientTape() as tape:
            y_hat = mlp(x)
            loss = tf.math.reduce_mean(tf.square(y - y_hat))
            complexity = l2_regularization(mlp.trainable_variables)
            l2_loss = loss + regularization_rate * complexity

        grads = tape.gradient(l2_loss, mlp.trainable_variables)
        grad_update(step_size, mlp.trainable_variables, grads)

        step_size *= decay_rate

    feature_1, feature_2 = np.meshgrid(
        np.linspace(x[:, 0].min(), x[:, 0].max()),
        np.linspace(x[:, 1].min(), x[:, 1].max()),
    )

    grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

    display = DecisionBoundaryDisplay(
        xx0=feature_1,
        xx1=feature_2,
        response=np.reshape(mlp(grid), feature_1.shape),
    )

    display.plot(levels=[0, 0.5, 1])

    display.ax_.scatter(x[:, 0], x[:, 1], c=y, edgecolor="black")

    plt.savefig("plot.pdf")
