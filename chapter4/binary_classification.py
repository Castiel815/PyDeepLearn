from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def binary_classification_model():
    """
    二分类问题
    :return:
    """
    # 加载IMDB数据集
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=10000)    # 仅保留训练数据中前10000个最常出现的单词

    # 将评论解码为文本
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    decoded_review = " ".join(
        [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
    )

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    print(x_train[0])

    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")

    # 模型定义
    model = keras.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    # 编译模型
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    # 留出验证集
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    # 训练模型
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    # 绘制训练损失和验证损失
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")  # 蓝色圆点
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")  # 蓝色实线
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 绘制训练精度和验证精度
    plt.clf()
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # 从头开始训练一个模型
    model = keras.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)
    print(results)

    print(model.predict(x_test))


def vectorize_sequences(sequences, dimension=10000):
    """
    用multi-hot编码对整数序列进行编码
    :param sequences:
    :param dimension:
    :return:
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        for j in sequences:
            results[i, j] = 1.
    return results


if __name__ == "__main__":
    print(1)
    binary_classification_model()
