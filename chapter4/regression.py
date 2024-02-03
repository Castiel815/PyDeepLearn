from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def regression():
    """
    标量回归问题
    :return:
    """
    # 加载波士顿房价数据集
    (train_data, train_targets), (test_data, test_targets) = (
        boston_housing.load_data())

    # 数据标准化
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std

    # K折交叉验证
    k = 4
    num_val_samples = len(train_data) // k
    # num_epochs = 100
    # all_scores = []
    # for i in range(k):
    #     print(f"Processing fold #{i}")
    #     val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    #     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    #     partial_train_data = np.concatenate(
    #         [train_data[:i * num_val_samples],
    #          train_data[(i + 1 * num_val_samples):]],
    #         axis=0)
    #     partial_train_targets = np.concatenate(
    #         [train_targets[:i * num_val_samples],
    #          train_targets[(i + 1 * num_val_samples):]],
    #         axis=0)
    #     model = build_model()
    #     model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, verbose=0)
    #     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    #     all_scores.append(val_mae)
    #
    # print(all_scores)

    # K折交叉验证-保存每折的验证分数
    num_epochs = 500
    all_mae_histories = []
    for i in range(k):
        print(f"Processing fold#{i}")
        val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
            [train_data[: i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[: i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=16, verbose=0)
        mae_history = history.history["val_mae"]
        all_mae_histories.append(mae_history)

    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    # 绘制验证MAE曲线
    truncated_mae_history = average_mae_history[10:]  # 剔除前10个数据点
    plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()


def build_model():
    """
    模型定义
    :return:
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


if __name__ == "__main__":
    print(1)
    regression()
