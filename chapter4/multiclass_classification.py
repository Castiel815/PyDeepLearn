import keras
from tensorflow.keras.datasets import reuters
import binary_classification
import numpy as np
from tensorflow.keras.utils import to_categorical   # Keras内置编码
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def multiclass_classification():
    """
    多分类问题-路透社数据集
    :return:
    """
    # 加载路透社数据集, 将数据限定为前10000个最常出现的单词
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    print(len(train_data))
    print(len(test_data))

    # 将新闻解码为文本
    word_index = reuters.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()]
    )
    decoded_newswire = " ".join(
        [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
    )

    # 编码输入数据
    # 将训练数据向量化
    x_train = binary_classification.vectorize_sequences(train_data)
    x_test = binary_classification.vectorize_sequences(test_data)
    # 将训练标签向量化
    y_train = to_one_hot(train_labels)
    y_test = to_one_hot(test_labels)

    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    # 模型定义
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax")
    ])

    # 编译模型
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    # 留出验证集
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]

    # 训练模型
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    # 绘制训练损失和验证损失
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 绘制训练精度和验证精度
    plt.clf()
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # 从头开始训练一个模型
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax")
    ])
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=9, batch_size=512)
    results = model.evaluate(x_test, y_test)
    print(results)

    predictions = model.predict(x_test)


def to_one_hot(labels, dimension=46):
    """
    编码标签
    :param labels:
    :param dimension:
    :return:
    """
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


if __name__ == "__main__":
    print(1)
    multiclass_classification()
