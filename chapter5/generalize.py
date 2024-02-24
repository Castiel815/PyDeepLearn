from tensorflow.keras.datasets import mnist
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


def generalise():
    # 向MNIST数据集添加白噪声通道或全零通道
    (train_images, train_labels), _ = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    train_images_with_noise_channels = np.concatenate([train_images, np.random.random((len(train_images), 784))],
                                                      axis=1)
    train_images_with_zeros_channels = np.concatenate([train_images, np.zeros((len(train_images), 784))], axis=1)

    model = get_model()
    history_noise = model.fit(
        train_images_with_noise_channels, train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2
    )

    model = get_model()
    history_zeros = model.fit(
        train_images_with_zeros_channels, train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2
    )

    # 绘图比较验证精度
    val_acc_noise = history_noise.history["val_accuracy"]
    val_acc_zeros = history_zeros.history["val_accuracy"]
    epochs = range(1, 11)
    plt.plot(epochs, val_acc_noise, "b-",
             label="Validation accuracy with noise channels")
    plt.plot(epochs, val_acc_zeros, "b--",
             label="Validation accuracy with zeros channels")
    plt.title("Effect of noise channels on validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


# 对于带有噪声通道或全零通道的MNIST数据，训练相同的模型
def get_model():
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":
    generalise()
