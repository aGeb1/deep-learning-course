import idx2numpy as idx
import numpy as np
import tensorflow as tf


def load_data(file):
    data = idx.convert_from_file(file)
    data = tf.reshape(data, np.append(tf.shape(data), 1))
    data /= 255

    return data


def load_labels(file):
    data = idx.convert_from_file(file)
    data = tf.cast(data, tf.int32)

    return data


test_data = load_data("data/t10k-images.idx3-ubyte")
test_labels = load_labels("data/t10k-labels.idx1-ubyte")

training_data = load_data("data/train-images.idx3-ubyte")
training_labels = load_labels("data/train-labels.idx1-ubyte")
