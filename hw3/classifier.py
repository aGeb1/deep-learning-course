import tensorflow as tf
from tqdm import trange

from adam import Adam
from layers import *
from mnist_data import *


class Classifier(tf.Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            # breakpoint()
        return x


rng = tf.random.get_global_generator()
rng.reset_from_seed(472)

dropout_rate = 0.2
model = Classifier(
    [
        Conv2d(3, 3, 1, 16, activation=tf.nn.relu),
        lambda x: tf.nn.max_pool2d(x, 2, 2, "VALID"),
        lambda x: tf.nn.dropout(x, rate=dropout_rate, seed=472),
        Conv2d(3, 3, 16, 24, activation=tf.nn.relu),
        lambda x: tf.nn.max_pool2d(x, 2, 2, "VALID"),
        lambda x: tf.nn.dropout(x, rate=dropout_rate, seed=472),
        Conv2d(3, 3, 24, 32, 2, "VALID", activation=tf.nn.relu),
        lambda x: tf.reshape(x, [tf.shape(x)[0], -1]),
        FullyConnected(288, 10, tf.identity),
    ]
)


if __name__ == "__main__":
    num_iters = 1000
    batch_size = 50
    num_samples = len(training_labels)

    adam = Adam()

    for i in trange(num_iters):
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            training_batch = tf.gather(training_data, batch_indices)
            label_batch = tf.gather(training_labels, batch_indices)

            output = model(training_batch)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                label_batch, output
            )

        grads = tape.gradient(loss, model.trainable_variables)
        adam.apply_gradients(grads, model.trainable_variables)

    test_output = tf.math.argmax(
        model(test_data), axis=1, output_type=tf.dtypes.int32
    )
    bool_array = test_output == test_labels

    acc = 0
    for b in bool_array:
        if b:
            acc += 1

    print(acc / len(test_data))
