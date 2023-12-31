# Help on loading the data came from:
# https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html

import pickle

import numpy as np
import tensorflow as tf


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")

    data = dict[b"data"]
    data = np.reshape(data, (-1, 3, 32, 32))
    data = np.transpose(data, (0, 2, 3, 1))
    data = tf.cast(data, np.float32)
    data /= 255

    try:
        labels = dict[b"labels"]
    except:
        labels = dict[b"fine_labels"]

    labels = tf.cast(labels, np.int32)

    return data, labels


cifar_10_training_data = [None, None, None, None]
cifar_10_training_labels = [None, None, None, None]

cifar_10_training_data[0], cifar_10_training_labels[0] = unpickle(
    "cifar10/data_batch_1"
)
cifar_10_training_data[1], cifar_10_training_labels[1] = unpickle(
    "cifar10/data_batch_2"
)
cifar_10_training_data[2], cifar_10_training_labels[2] = unpickle(
    "cifar10/data_batch_3"
)
cifar_10_training_data[3], cifar_10_training_labels[3] = unpickle(
    "cifar10/data_batch_4"
)

cifar_10_training_data = tf.concat(
    [
        cifar_10_training_data[0],
        cifar_10_training_data[1],
        cifar_10_training_data[2],
        cifar_10_training_data[3],
    ],
    0,
)

cifar_10_training_labels = tf.concat(
    [
        cifar_10_training_labels[0],
        cifar_10_training_labels[1],
        cifar_10_training_labels[2],
        cifar_10_training_labels[3],
    ],
    0,
)

cifar_10_validation_data, cifar_10_validation_labels = unpickle(
    "cifar10/data_batch_5"
)
cifar_10_test_data, cifar_10_test_labels = unpickle("cifar10/test_batch")


cifar_100_training_data, cifar_100_training_labels = unpickle("cifar100/train")

cifar_100_validation_data = cifar_100_training_data[-5000:]
cifar_100_validation_labels = cifar_100_training_labels[-5000:]

cifar_100_training_data = cifar_100_training_data[:-5000]
cifar_100_training_labels = cifar_100_training_labels[:-5000]

cifar_100_test_data, cifar_100_test_labels = unpickle("cifar100/test")
