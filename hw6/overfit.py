import tensorflow as tf
from tqdm import trange

from attention import DecoderOnlyModel


class AdamW:
    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        ep=1e-8,
        weight_decay=1e-3,
    ):
        # Initialize optimizer parameters and variable slots
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.ep = ep
        self.weight_decay = weight_decay
        self.t = 1.0
        self.v_dvar, self.s_dvar = [], []
        self.built = False

    def apply_gradients(self, grads, vars):
        # Initialize variables on the first call
        if not self.built:
            for var in vars:
                v = tf.Variable(tf.zeros(shape=var.shape))
                s = tf.Variable(tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        # Update the model variables given their gradients
        for i, (d_var, var) in enumerate(zip(grads, vars)):
            self.v_dvar[i].assign(
                self.beta_1 * self.v_dvar[i] + (1 - self.beta_1) * d_var
            )
            self.s_dvar[i].assign(
                self.beta_2 * self.s_dvar[i]
                + (1 - self.beta_2) * tf.square(d_var)
            )
            v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1**self.t))
            s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2**self.t))
            var.assign_sub(
                self.learning_rate
                * (v_dvar_bc / (tf.sqrt(s_dvar_bc) + self.ep))
                + self.learning_rate * self.weight_decay * var
            )
        self.t += 1.0
        return

    def reduce_learning_rate(self, learning_rate_drop):
        self.learning_rate *= learning_rate_drop


length = 512
transformer_blocks = 2
num_iters = 1000


class OverfitTest(tf.Module):
    def __init__(self):
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(472)

        self.embeddings = rng.normal(shape=[1, length, 64])
        self.model = DecoderOnlyModel(8, 64, 8, 256, length, transformer_blocks)

    def __call__(self):
        return self.model(self.embeddings)


adam = AdamW()
model = OverfitTest()

for i in trange(num_iters):
    with tf.GradientTape() as tape:
        y = model()
        loss = tf.nn.softmax_cross_entropy_with_logits(
            tf.expand_dims(tf.eye(length), 0),
            y,
        )

    grads = tape.gradient(loss, model.trainable_variables)
    adam.apply_gradients(grads, model.trainable_variables)

    if i == num_iters - 1:
        print(tf.reduce_mean(loss).numpy())
