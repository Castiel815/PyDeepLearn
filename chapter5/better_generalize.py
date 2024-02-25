# 提高泛化能力
import keras
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

# 初始模型
(train_data, train_labels), _ = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


train_data = vectorize_sequences(train_data)
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original = model.fit(train_data, train_labels,
                             epochs=20, batch_size=512, validation_split=0.4)


# 容量更小的模型
model = keras.Sequential([
    layers.Dense(4, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_smaller_model = model.fit(
    train_data, train_labels,
    epochs=20, batch_size=512, validation_split=0.4
)


# 容量更大的模型
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_larger_model = model.fit(train_data, train_labels,
                                 epochs=20, batch_size=512, validation_split=0.4)


# 向模型中添加L2权重正则化
model = keras.Sequential([
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.002),
                 activation="relu"),
    layers.Dense(16,
                 kernel_regularizer=regularizers.l2(0.002),
                 activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_l2_reg = model.fit(train_data, train_labels,
                           epochs=20, batch_size=512, validation_split=0.4)


# 向IMDB模型中添加dropout
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_dropout = model.fit(train_data, train_labels,
                            epochs=20, batch_size=512, validation_split=0.4)

if __name__ == "__main__":
    ori_loss_values = history_original.history["val_loss"]
    smaller_loss = history_smaller_model.history["val_loss"]
    epochs = range(1, len(ori_loss_values) + 1)
    plt.plot(epochs, ori_loss_values, "b--", label="Validation loss of original model")  # 蓝色圆点
    plt.plot(epochs, smaller_loss, "b-", label="Validation loss of smaller model")  # 蓝色实线
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()
    ori_loss_values = history_original.history["val_loss"]
    larger_loss = history_larger_model.history["val_loss"]
    epochs = range(1, len(ori_loss_values) + 1)
    plt.plot(epochs, ori_loss_values, "b--", label="Validation loss of original model")  # 蓝色圆点
    plt.plot(epochs, larger_loss, "b-", label="Validation loss of larger model")  # 蓝色实线
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()
    ori_loss_values = history_original.history["val_loss"]
    l2_loss = history_l2_reg.history["val_loss"]
    epochs = range(1, len(ori_loss_values) + 1)
    plt.plot(epochs, ori_loss_values, "b--", label="Validation loss of original model")  # 蓝色圆点
    plt.plot(epochs, l2_loss, "b-", label="Validation loss of L2-regularized model")  # 蓝色实线
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()
    ori_loss_values = history_original.history["val_loss"]
    dropout_loss = history_dropout.history["val_loss"]
    epochs = range(1, len(ori_loss_values) + 1)
    plt.plot(epochs, ori_loss_values, "b--", label="Validation loss of original model")  # 蓝色圆点
    plt.plot(epochs, dropout_loss, "b-", label="Validation loss of dropout-regularized model")  # 蓝色实线
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    print(1)
