import tensorflow as tf

from layers import FFN_ReLU, LayerNormalization, Linear


class MultiHeadAttention(tf.Module):
    def __init__(self, h, d_model, d_attn, masking=False):
        self.h = h
        self.d_attn = d_attn
        self.masking = masking

        self.W_k = Linear(d_model, d_attn * h, bias=False)
        self.W_q = Linear(d_model, d_attn * h, bias=False)
        self.W_v = Linear(d_model, d_attn * h, bias=False)
        self.W_o = Linear(d_attn * h, d_model, bias=False)

    def mask(_, Z):
        ones = tf.ones_like(Z[0, :, :])  # (d_k, d_k)
        ones = tf.linalg.LinearOperatorLowerTriangular(ones).to_dense()
        ones = tf.tile(tf.expand_dims(ones, 0), [tf.shape(Z)[0], 1, 1])

        mask = tf.ones_like(ones) * float("-inf")
        mask = tf.where(tf.equal(ones, 0), mask, Z)

        return mask

    def __call__(self, x):
        K = self.W_k(x)  # (N, l, d_k*h)
        Q = self.W_q(x)  # (N, l, d_k*h)
        V = self.W_v(x)  # (N, l, d_v*h)

        K = tf.concat(tf.split(K, self.h, axis=2), axis=0)  # (N*h, l, d_k)
        Q = tf.concat(tf.split(Q, self.h, axis=2), axis=0)  # (N*h, l, d_k)
        V = tf.concat(tf.split(V, self.h, axis=2), axis=0)  # (N*h, l, d_v)

        Z = Q @ tf.transpose(K, perm=(0, 2, 1))  # (N*h, l, l)
        if self.masking:
            Z = self.mask(Z)
        W = tf.nn.softmax(Z / tf.sqrt(float(self.d_attn)))  # (N*h, l, l)

        W = W @ V  # (N*h, l, d_v)
        W = tf.concat(tf.split(W, self.h, axis=0), axis=2)  # (N, l, d_v*h)

        return self.W_o(W)  # (N, l, d_model)


class TransformerBlock(tf.Module):
    def __init__(self, h, d_model, d_attn, d_ff, masking=False):
        self.attention = MultiHeadAttention(h, d_model, d_attn, masking)
        self.ffn = FFN_ReLU(d_model, d_ff)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

    def __call__(self, x):
        x = self.layer_norm1(x)
        x += self.attention(x)
        x = self.layer_norm2(x)
        x += self.ffn(x)
        return x


class DecoderOnlyModel(tf.Module):
    def __init__(self, h, d_model, d_attn, d_ff, vocab_size, layer_count):
        self.layers = []
        for _ in range(layer_count):
            self.layers.append(TransformerBlock(h, d_model, d_attn, d_ff, True))
        self.layers.append(Linear(d_model, vocab_size, bias=False))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
