# Simple tests for trainability, dimensionality, and basic functionality

import pytest
import tensorflow as tf

from attention import DecoderOnlyModel, MultiHeadAttention, TransformerBlock
from layers import FFN_ReLU, SwiGLU


def test_ffn_relu_trainable():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    ffn = FFN_ReLU(128, 1024)

    x = rng.normal(shape=[10, 10, 128])

    with tf.GradientTape() as tape:
        y = ffn(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, ffn.trainable_variables)

    for grad, var in zip(grads, ffn.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(ffn.trainable_variables)


def test_ffn_dimensionality():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    ffn = FFN_ReLU(128, 1024)

    x = rng.normal(shape=[10, 20, 128])
    y = ffn(x)

    tf.assert_equal(tf.shape(x), tf.shape(y))


def test_swiglu_trainable():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    ffn = SwiGLU(128, 1024)

    x = rng.normal(shape=[10, 20, 128])

    with tf.GradientTape() as tape:
        y = ffn(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, ffn.trainable_variables)

    for grad, var in zip(grads, ffn.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)


def test_swiglu_dimensionality():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    ffn = SwiGLU(128, 1024)

    x = rng.normal(shape=[10, 10, 128])
    y = ffn(x)

    tf.assert_equal(tf.shape(x), tf.shape(y))


@pytest.mark.parametrize("masking", [True, False])
def test_mha_trainable(masking):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    mha = MultiHeadAttention(8, 128, 32, masking)

    x = rng.normal(shape=[10, 20, 128])

    with tf.GradientTape() as tape:
        y = mha(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, mha.trainable_variables)

    for grad, var in zip(grads, mha.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)


@pytest.mark.parametrize("masking", [True, False])
def test_mha_dimensionality(masking):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    mha = MultiHeadAttention(8, 128, 32, masking)

    x = rng.normal(shape=[10, 20, 128])
    y = mha(x)

    tf.assert_equal(tf.shape(x), tf.shape(y))


@pytest.mark.parametrize("masking", [True, False])
def test_transformer_block_trainable(masking):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    transformer_block = TransformerBlock(8, 128, 32, 256, masking)

    x = rng.normal(shape=[10, 20, 128])

    with tf.GradientTape() as tape:
        y = transformer_block(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, transformer_block.trainable_variables)

    for grad, var in zip(grads, transformer_block.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)


@pytest.mark.parametrize("masking", [True, False])
def test_transformer_block_dimensionality(masking):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    transformer_block = TransformerBlock(8, 128, 32, 256, masking)

    x = rng.normal(shape=[10, 20, 128])
    y = transformer_block(x)

    tf.assert_equal(tf.shape(x), tf.shape(y))


def test_decoder_only_model_trainable():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    decoder = DecoderOnlyModel(8, 32, 16, 64, 128, 2)

    x = rng.normal(shape=[10, 20, 32])

    with tf.GradientTape() as tape:
        y = decoder(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, decoder.trainable_variables)

    for grad, var in zip(grads, decoder.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)


def test_decoder_only_model_dimensionality():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    decoder = DecoderOnlyModel(8, 32, 16, 64, 128, 2)

    x = rng.normal(shape=[10, 20, 32])
    y = decoder(x)

    tf.assert_equal(tf.shape(y), [10, 20, 128])


@pytest.mark.parametrize("length", [8, 16, 32, 64, 128])
def test_causality(length):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    mha = MultiHeadAttention(1, 1, 1, True)

    with tf.GradientTape() as tape:
        x = rng.normal(shape=[1, length, 1])
        tape.watch(x)
        y = mha(x)

    J = tape.jacobian(y, x)
    J = tf.where(tf.equal(J, 0), tf.zeros_like(J), tf.ones_like(J))
    J = tf.reshape(J, [length, length])

    lower_triangle = tf.linalg.LinearOperatorLowerTriangular(
        tf.ones([length, length])
    ).to_dense()

    tf.assert_equal(J, lower_triangle)
