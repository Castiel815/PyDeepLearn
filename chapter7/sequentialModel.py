from tensorflow import keras
from tensorflow.keras import layers


# 序贯模型
def sequential_model():
    # 1.Sequential类
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    # 2.逐步构建序贯模型
    model2 = keras.Sequential()
    model2.add(layers.Dense(64, activation="relu"))
    model2.add(layers.Dense(10, activation="softmax"))

    # 通过第一次调用模型来完成构建
    model.build(input_shape=(None, 3))

    # summary()方法
    model.summary()

    # 利用name参数命名模型和层
    model3 = keras.Sequential(name="my_example_model")
    model3.add(layers.Dense(64, activation="relu", name="my_first_layer"))
    model3.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
    model3.build((None, 3))
    model3.summary()


if __name__ == "__main__":
    sequential_model()
    print(1)
