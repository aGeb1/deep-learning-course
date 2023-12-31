# Much of the code in this file resembles code from the PyTorch tutorials.

import tensorflow as tf
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from layers import Linear
from optim import AdamW

rng = tf.random.get_global_generator()
rng.reset_from_seed(472)

dataset = load_dataset("ag_news")
pretrained_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = pretrained_model.encode(dataset["train"]["text"])
embeddings = tf.split(embeddings, 10)
train_embeddings = tf.concat(embeddings[1:], axis=0)
validation_embeddings = embeddings[0]

train_dataset = [train_embeddings, dataset["train"]["label"][12000:]]
validation_dataset = [validation_embeddings, dataset["train"]["label"][:12000]]
train_batch_size = 1024

test_embeddings = pretrained_model.encode(dataset["test"]["text"])
test_dataset = [test_embeddings, dataset["test"]["label"]]
test_batch_size = 128

model = Linear(384, 4)
loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
optimizer = AdamW(weight_decay=0.02)


def to_batches(dataset, batch_size, size):
    data_batches = [
        dataset[0][(batch_size * i) : (batch_size * (i + 1))]
        for i in range(size // batch_size)
    ]
    label_batches = [
        dataset[1][(batch_size * i) : (batch_size * (i + 1))]
        for i in range(size // batch_size)
    ]
    return [data_batches, label_batches]


def train_loop(
    train_dataset, validation_dataset, model, loss_fn, optimizer, batch_size
):
    size = len(train_dataset[0])
    batches = to_batches(train_dataset, batch_size, size)

    for batch in range(size // batch_size):
        with tf.GradientTape() as tape:
            pred = model(batches[0][batch])
            loss = loss_fn(batches[1][batch], pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads, model.trainable_variables)

        if batch % 100 == 0:
            batch_indices = rng.uniform(
                shape=[batch_size],
                maxval=len(validation_dataset[0]),
                dtype=tf.int32,
            )
            validation_batch = [
                tf.gather(validation_dataset[0], batch_indices),
                tf.gather(validation_dataset[1], batch_indices),
            ]

            validation_pred = model(validation_batch[0])
            validation_loss = loss_fn(validation_batch[1], validation_pred)
            print(f"Training Loss: {tf.reduce_mean(loss).numpy():>7f}")
            print(
                f"Validation Loss: {tf.reduce_mean(validation_loss).numpy():>7f}"
            )


def test_loop(dataset, model, batch_size):
    size = len(dataset[0])
    batches = to_batches(dataset, batch_size, size)
    num_batches = size // batch_size
    test_loss, correct = 0, 0

    for batch in range(num_batches):
        pred = model(batches[0][batch])
        correct += sum(
            tf.math.in_top_k(batches[1][batch], pred, 1).numpy().tolist()
        )

    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%\n")


epochs = 200
for t in range(epochs):
    if t == 60 or t == 120:
        optimizer.reduce_learning_rate(0.2)
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(
        train_dataset,
        validation_dataset,
        model,
        loss_fn,
        optimizer,
        train_batch_size,
    )
test_loop(test_dataset, model, test_batch_size)
print("Done!")
