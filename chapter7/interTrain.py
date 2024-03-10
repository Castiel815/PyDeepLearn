from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from matplotlib import pyplot as plt


# 标准工作流程：compile()、fit()、evaluate()、predict()
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model


def train_and_evaluate():
    (images, labels), (test_images, test_labels) = mnist.load_data()
    images = images.reshape((60000, 28 * 28)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
    train_images, val_images = images[10000:], images[:10000]
    train_labels, val_labels = labels[10000:], labels[:10000]

    model = get_mnist_model()
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=3, validation_data=(val_images, val_labels))
    test_metrics = model.evaluate(test_images, test_labels)
    predictions = model.predict(test_images)

    model = get_mnist_model()
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", RootMeanSquaredError()])
    model.fit(train_images, train_labels,
              epochs=3,
              validation_data=(val_images, val_labels))
    test_metrics = model.evaluate(test_images, test_labels)

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="checkpoint_path.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]
    model = get_mnist_model()
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_images, train_labels,
              epochs=10,
              callbacks=callbacks_list,
              validation_data=(val_images, val_labels))

    model = get_mnist_model()
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_images, train_labels,
              epochs=10,
              callbacks=[LossHistory()],
              validation_data=(val_images, val_labels))


class RootMeanSquaredError(keras.metrics.Metric):
    """
    通过将Metric类子类化来实现自定义指标
    """
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros", dtype="int32"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        在update_state()中实现状态更新逻辑
        :param y_true: 一个数据批量对应的目标(或标签)
        :param y_pred: 相应的模型预测值
        :param sample_weight:
        :return:
        """
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        """
        返回指标的当前值
        :return:
        """
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))

    def reset_state(self):
        """
        重置指标状态
        :return:
        """
        self.mse_sum.assign_add(0.)
        self.total_samples.assign_add(0)


class LossHistory(keras.callbacks.Callback):
    """
    通过对Callback类子类化来创建自定义回调函数
    """
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses,
                 label="Training loss for each batch")
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}")
        self.per_batch_losses = []


if __name__ == "__main__":
    train_and_evaluate()
