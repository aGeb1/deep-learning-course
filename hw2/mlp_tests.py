import pytest


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_mlp_dimensionality(num_outputs):
    import tensorflow as tf

    from mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10
    num_hidden_layers = 6
    hidden_layer_width = 5

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
    )

    a = rng.normal(shape=[1, num_inputs])
    z = mlp(a)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


def test_mlp_trainable():
    import tensorflow as tf

    from mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10
    num_outputs = 3
    num_hidden_layers = 5
    hidden_layer_width = 5

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
    )

    a = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        z = mlp(a)
        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, mlp.trainable_variables)

    for grad, var in zip(grads, mlp.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")

    assert len(grads) == len(mlp.trainable_variables)
    assert len(grads) == (num_hidden_layers + 1) * 2


@pytest.mark.parametrize(
    "num_inputs, num_outputs, num_hidden_layers, hidden_layer_width",
    [(1, 1, 1, 1), (5, 5, 5, 5), (1, 1, 5, 5), (5, 5, 1, 1)],
)
def test_layers_topology(
    num_inputs,
    num_outputs,
    num_hidden_layers,
    hidden_layer_width,
):
    import tensorflow as tf

    from mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
    )

    layer_shapes = list(map(lambda linear: tf.shape(linear.w), mlp.layers))

    assert len(layer_shapes) == num_hidden_layers + 1

    tf.assert_equal(layer_shapes[0], [num_inputs, hidden_layer_width])
    tf.assert_equal(
        layer_shapes[1:-1],
        [[hidden_layer_width, hidden_layer_width]] * (num_hidden_layers - 1),
    )
    tf.assert_equal(layer_shapes[-1], [hidden_layer_width, num_outputs])


def test_linear_activations():
    import tensorflow as tf

    from mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100

    num_hidden_layers = 5
    hidden_layer_width = 5

    linear = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])
    c = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(linear(a + b), linear(a) + linear(b), summarize=2)
    tf.debugging.assert_near(linear(a * c), linear(a) * c, summarize=2)


def test_no_hidden_layers():
    import tensorflow as tf

    from mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 5
    hidden_layer_width = 5

    one_layer = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        output_activation=tf.nn.relu,
    )

    a = rng.normal(shape=[1, num_inputs])
    zeros = tf.zeros(shape=[1, num_inputs])

    tf.debugging.assert_greater_equal(one_layer(a), zeros)
