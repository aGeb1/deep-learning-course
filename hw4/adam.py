# Code from https://www.tensorflow.org/guide/core/mlp_core

import tensorflow as tf


class Adam:
    def __init__(
        self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7, l=1e-3
    ):
        # Initialize optimizer parameters and variable slots
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.ep = ep
        self.l = l
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
                + self.learning_rate * self.l * var
            )
        self.t += 1.0
        return

    def reduce_learning_rate(self, learning_rate_drop):
        self.learning_rate *= learning_rate_drop
