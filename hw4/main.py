import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import trange

from adam import Adam
from layers import *
from load_cifar import *
from models import WideResNet

rng = tf.random.get_global_generator()
rng.reset_from_seed(472)


def train_model(m, name=None):
    global training_data

    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]

    num_iters = np.int32((num_epochs * len(training_data)) / batch_size)

    adam = Adam(
        config["training"]["adam"]["learning_rate"],
        config["training"]["adam"]["beta_1"],
        config["training"]["adam"]["beta_2"],
        config["training"]["adam"]["ep"],
        config["training"]["adam"]["lambda"],
    )

    validation_rate = config["training"]["validation_rate"]
    validation_batch_size = config["training"]["validation_batch_size"]

    training_loss = np.array([])
    validation_loss = np.array([])

    for i in trange(num_iters):
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=len(training_data), dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            training_batch = augment(tf.gather(training_data, batch_indices))
            label_batch = tf.gather(training_labels, batch_indices)

            output = m(training_batch)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                label_batch, output
            )

        grads = tape.gradient(loss, m.trainable_variables)
        adam.apply_gradients(grads, m.trainable_variables)

        if i == int((12 * len(training_data)) // batch_size):
            adam.reduce_learning_rate(
                config["training"]["adam"]["learning_rate_drop"]
            )

        if i % validation_rate == 0:
            training_loss_value = tf.reduce_mean(loss)
            training_loss = np.append(training_loss, training_loss_value)

            validation_batch_indices = rng.uniform(
                shape=[validation_batch_size],
                maxval=len(validation_data),
                dtype=tf.int32,
            )
            validation_batch = tf.gather(
                validation_data, validation_batch_indices
            )
            validation_label_batch = tf.gather(
                validation_labels, validation_batch_indices
            )

            validation_output = m(validation_batch)
            validation_loss_value = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    validation_label_batch, validation_output
                )
            )
            validation_loss = np.append(
                validation_loss,
                validation_loss_value,
            )

            print("Training Loss: ", training_loss_value.numpy())
            print("Validation Loss: ", validation_loss_value.numpy())

    if name:
        x = np.linspace(0, num_epochs, training_loss.size)

        plt.plot(x, training_loss, label="Test Loss")
        plt.plot(x, validation_loss, label="Validation Loss")

        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")

        plt.legend()

        plt.savefig(name + ".pdf")


def augment(x):
    batch_size = config["training"]["batch_size"]

    x = tf.image.resize_with_pad(x, 40, 40)
    x = tf.image.random_flip_left_right(x, 472)
    x = tf.image.random_crop(x, (batch_size, 32, 32, 3), 472)

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Wide Residual Network",
        description="Generates a wide residual network based on a configuration file",
    )

    parser.add_argument(
        "-c", "--config", type=Path, default=Path("config.yaml")
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    dataset = config["general"]["dataset"]

    if dataset == "CIFAR-10":
        training_data = cifar_10_training_data
        training_labels = cifar_10_training_labels
        validation_data = cifar_10_validation_data
        validation_labels = cifar_10_validation_labels
        test_data = cifar_10_test_data
        test_labels = cifar_10_test_labels
        number_of_categories = 10
    elif dataset == "CIFAR-100":
        training_data = cifar_100_training_data
        training_labels = cifar_100_training_labels
        validation_data = cifar_100_validation_data
        validation_labels = cifar_100_validation_labels
        test_data = cifar_100_test_data
        test_labels = cifar_100_test_labels
        number_of_categories = 100

    model = WideResNet(
        config["resnet"]["N"],
        config["resnet"]["k"],
        number_of_categories,
        config["resnet"]["dropout_rate"],
    )

    print("Parameter count: ", model.parameter_count())

    train_model(model, "test")

    test_data = tf.split(test_data, 100)
    test_labels = tf.split(test_labels, 100)

    bool_array_top1 = []
    bool_array_top5 = []

    for i in range(100):
        data_partition = test_data[i]
        label_partition = test_labels[i]

        model_output = model(data_partition)
        top1_buffer = tf.math.in_top_k(label_partition, model_output, 1)
        top5_buffer = tf.math.in_top_k(label_partition, model_output, 5)

        bool_array_top1 += top1_buffer.numpy().tolist()
        bool_array_top5 += top5_buffer.numpy().tolist()

    acc_top1 = 0
    for b in bool_array_top1:
        if b:
            acc_top1 += 1

    acc_top5 = 0
    for b in bool_array_top5:
        if b:
            acc_top5 += 1

    print(acc_top1 / len(test_data))
    print(acc_top5 / len(test_data))
