import pytest


def test_linear_additivity():
    import tensorflow as tf

    from hw1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(linear(a + b), linear(a) + linear(b))


def test_linear_homogeneity():
    import tensorflow as tf

    from hw1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(linear(a * 10), linear(a) * 10)


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_linear_dimensionality(num_outputs):
    import tensorflow as tf

    from hw1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])

    tf.assert_equal(tf.shape(linear(a))[-1], num_outputs)


def test_linear_trainable():
    import tensorflow as tf

    from hw1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs)

    x = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        loss = tf.math.reduce_mean(-linear(x))

    grads = tape.gradient(loss, linear.trainable_variables)
    for grad, var in zip(grads, linear.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(linear.trainable_variables)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [([1000, 1000], [100, 100]), ([1000, 100], [100, 100]), ([100, 1000], [100, 100])],
)
def test_linear_init_properties(a_shape, b_shape):
    import tensorflow as tf

    from hw1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs_a, num_outputs_a = a_shape
    num_inputs_b, num_outputs_b = b_shape

    linear_a = Linear(num_inputs_a, num_outputs_a)
    linear_b = Linear(num_inputs_b, num_outputs_b)

    std_a = tf.math.reduce_std(linear_a.w)
    std_b = tf.math.reduce_std(linear_b.w)

    tf.debugging.assert_less(std_a, std_b)


@pytest.mark.parametrize("M", [1, 16, 128])
def test_basis_expansion_dimensionality(M):
    import tensorflow as tf

    from hw1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    N = 10

    basisexpansion = BasisExpansion(M)

    x = rng.uniform(shape=[N, 1])

    tf.assert_equal(tf.shape(basisexpansion(x)), [N, M])


def test_basis_expansion_trainable():
    import tensorflow as tf

    from hw1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    M = 10
    N = 50

    basisexpansion = BasisExpansion(M)

    x = rng.normal(shape=[N, 1])
    with tf.GradientTape() as tape:
        loss = tf.math.reduce_mean(-basisexpansion(x))

    grads = tape.gradient(loss, basisexpansion.trainable_variables)
    for grad, var in zip(grads, basisexpansion.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(basisexpansion.trainable_variables)


def test_model_trainable():
    import tensorflow as tf

    from hw1 import Model

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    M = 10
    N = 50

    model = Model(M)

    x = rng.normal(shape=[N, 1])
    with tf.GradientTape() as tape:
        loss = tf.math.reduce_mean(-model(x))

    grads = tape.gradient(loss, model.trainable_variables)
    for grad, var in zip(grads, model.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(model.trainable_variables)
