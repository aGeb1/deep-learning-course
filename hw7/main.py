import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from tqdm import trange

from adamw import AdamW
from layers import *

parser = argparse.ArgumentParser(
    prog="SIREN Model",
    description="Represents an image using the SIREN model.",
)

parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
args = parser.parse_args()

config = yaml.safe_load(args.config.read_text())

input_name = config["input_name"]
output_name = config["output_name"]


def meshgrid(height, width):
    a = tf.linspace(-1, 1, height)
    b = tf.linspace(-1, 1, width)
    m = tf.meshgrid(a, b)
    m = tf.cast(m, tf.float32)
    m = tf.reshape(tf.transpose(m, [2, 1, 0]), [height * width, 2])

    return m


filename = input_name
pngstring = tf.io.read_file(filename)
image = tf.io.decode_png(pngstring)  # uint8

[height, width, _] = tf.shape(image)
x = meshgrid(height, width)
y = tf.reshape(image, [height * width, 3])
y = tf.cast(y, tf.float32) / 256


num_inputs = config['siren']['num_inputs']
num_outputs = config['siren']['num_outputs']
num_hidden_layers = config['siren']['num_hidden_layers']
hidden_layer_width = config['siren']['hidden_layer_width']

adam = AdamW()
model = Siren(2, 3, 3, 256)
num_iters = config['num_iters']

for i in trange(num_iters):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.math.reduce_mean(tf.square(y - y_pred))

    grads = tape.gradient(loss, model.trainable_variables)
    adam.apply_gradients(grads, model.trainable_variables)

    if i == num_iters - 1:
        print(tf.reduce_mean(loss).numpy())


upscale = config["upscale"]
x = meshgrid(height * upscale, width * upscale)
approx = model(x)
approx = tf.reshape(approx, [height * upscale, width * upscale, 3])
approx *= 256
approx = tf.cast(approx, tf.uint8)
approx = tf.io.encode_png(approx)
tf.io.write_file(output_name, approx)
