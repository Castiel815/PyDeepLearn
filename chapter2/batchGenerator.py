import math
import tensorflow as tf
from tensorflow.keras import optimizers


class BatchGenerator:
    """
    批量生成器
    """
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index: self.index + self.batch_size]
        labels = self.labels[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions
        )
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


learning_rate = 1e-3


# def update_weights(gradients, weights):
#     for g, w in zip(gradients, weights):
#         w.assign_sub(g * learning_rate)


optimizer = optimizers.SGD(learning_rate=1e-3)


def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))


def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        batch_generator = BatchGenerator(images, labels, batch_size)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")
