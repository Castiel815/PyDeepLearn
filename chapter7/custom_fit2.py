from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf


loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")


# 实现自定义的训练步骤，并与fit()结合使用
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(targets, predictions)

        return {m.name:m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [loss_tracker]


def test():
    (images, labels), (test_images, test_labels) = mnist.load_data()
    images = images.reshape((60000, 28 * 28)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
    train_images, val_images = images[10000:], images[:10000]
    train_labels, val_labels = labels[10000:], labels[:10000]

    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    model.fit(train_images, train_labels, epochs=3)


if __name__ == "__main__":
    test()
