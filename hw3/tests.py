import pytest


@pytest.mark.parametrize(
    "input_height, input_width, filter_height, filter_width, in_channels, out_channels, stride, padding",
    [
        (28, 28, 3, 3, 1, 16, 1, "SAME"),
        (14, 14, 3, 3, 16, 24, 1, "SAME"),
        (7, 7, 3, 3, 24, 32, 2, "VALID"),
        (7, 7, 5, 5, 24, 32, 3, "VALID"),
    ],
)
def test_conv2d_dimensionality(
    input_height,
    input_width,
    filter_height,
    filter_width,
    in_channels,
    out_channels,
    stride,
    padding,
):
    import tensorflow as tf

    from layers import Conv2d

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    conv2d = Conv2d(
        filter_height,
        filter_width,
        in_channels,
        out_channels,
        stride,
        padding,
    )

    x = rng.normal(shape=[1, input_height, input_width, in_channels])
    y = conv2d(x)

    if padding == "SAME":
        tf.assert_equal(
            tf.shape(y),
            [1, input_height // stride, input_width // stride, out_channels],
        )
    else:
        tf.assert_equal(
            tf.shape(y),
            [
                1,
                (input_height - (filter_height - stride)) // stride,
                (input_width - (filter_width - stride)) // stride,
                out_channels,
            ],
        )


@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_trainable(bias):
    import tensorflow as tf

    from layers import Conv2d

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    conv2d = Conv2d(5, 6, 7, 8, bias=bias)

    x = rng.normal(shape=[1, 5, 6, 7])

    with tf.GradientTape() as tape:
        y = conv2d(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, conv2d.trainable_variables)
    print(grads)

    for grad, var in zip(grads, conv2d.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(conv2d.trainable_variables)

    if bias:
        assert len(grads) == 2
    else:
        assert len(grads) == 1


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_fc_dimensionality(num_outputs):
    import tensorflow as tf

    from layers import FullyConnected

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10

    fc = FullyConnected(num_inputs, num_outputs)

    x = rng.normal(shape=[1, num_inputs])
    y = fc(x)

    tf.assert_equal(tf.shape(y)[-1], num_outputs)


def test_fc_trainable():
    import tensorflow as tf

    from layers import FullyConnected

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    num_inputs = 10
    num_outputs = 1

    fc = FullyConnected(num_inputs, num_outputs)

    x = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        y = fc(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, fc.trainable_variables)

    for grad, var in zip(grads, fc.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(fc.trainable_variables)

    assert len(grads) == 2


def test_model_dimensionality():
    import tensorflow as tf

    from classifier import model

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    x = rng.normal(shape=[15, 28, 28, 1])
    y = model(x)

    tf.assert_equal(tf.shape(y), [15, 10])


def test_model_trainable():
    import tensorflow as tf

    from classifier import model

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(472)

    x = rng.normal(shape=[15, 28, 28, 1])

    with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.math.reduce_mean(y**2)

    grads = tape.gradient(loss, model.trainable_variables)

    for grad, var in zip(grads, model.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")

    assert len(grads) == len(model.trainable_variables)
