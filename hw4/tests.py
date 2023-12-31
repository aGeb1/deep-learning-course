import pytest
import tensorflow as tf


def test_residual_trainable():
    from layers import Residual, Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    residual = Residual([Linear(50, 50)])

    x = rng.normal(shape=[10, 50])

    with tf.GradientTape() as tape:
        y = residual(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, residual.trainable_variables)

    for grad, var in zip(grads, residual.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(residual.trainable_variables)


@pytest.mark.parametrize("width", [1, 16, 128])
def test_residual_dimensionality(width):
    from layers import Residual, Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    residual = Residual([Linear(width, width)], tf.identity)

    x = rng.normal(shape=[15, width])
    y = residual(x)

    tf.assert_equal(tf.shape(x), tf.shape(y))


def test_residual_linear():
    from layers import Residual, Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    width = 50
    num_test_cases = 100

    residual = Residual([Linear(width, width)])

    a = rng.normal(shape=[1, width])
    b = rng.normal(shape=[1, width])
    c = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(
        residual(a + b), residual(a) + residual(b), summarize=2
    )
    tf.debugging.assert_near(
        residual(a * c) / 2, residual(a) * c / 2, summarize=2
    )


def test_group_normalization_trainable():
    from layers import GroupNormalization

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    group_norm = GroupNormalization(4)

    x = rng.normal(shape=[15, 32, 32, 16])

    with tf.GradientTape() as tape:
        y = group_norm(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, group_norm.trainable_variables)

    for grad, var in zip(grads, group_norm.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(group_norm.trainable_variables)


def test_group_normalization_initialization():
    from layers import GroupNormalization

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    group_norm = GroupNormalization(4)

    x = rng.normal(shape=[15, 32, 32, 16])
    y = group_norm(x)

    tf.debugging.assert_less(
        tf.math.abs(tf.math.reduce_mean(y)), 1e-4, summarize=2
    )
    tf.debugging.assert_less(
        tf.math.abs(tf.math.reduce_std(y) - 1), 1e-4, summarize=2
    )


def test_wrn_trainable():
    from models import WideResNet

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    wrn = WideResNet(2, 8, 10, 0.3)
    x = rng.normal(shape=[15, 32, 32, 3])

    with tf.GradientTape() as tape:
        y = wrn(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, wrn.trainable_variables)

    for grad, var in zip(grads, wrn.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")

    assert len(grads) == len(wrn.trainable_variables)


@pytest.mark.parametrize("categories", [1, 16, 128])
def test_wrn_dimensionality(categories):
    from models import WideResNet

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    wrn = WideResNet(2, 8, categories, 0.3)
    x = rng.normal(shape=[15, 32, 32, 3])
    y = wrn(x)

    tf.assert_equal(tf.shape(y), (15, categories), summarize=2)
