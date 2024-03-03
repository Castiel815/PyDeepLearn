from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# 多输入、多输出的函数式模型
def functional_api():
    vocabulary_size = 10000
    num_tags = 100
    num_departments = 4

    title = keras.Input(shape=(vocabulary_size,), name="title")
    text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
    tags = keras.Input(shape=(num_tags,), name="tags")

    # 通过拼接将输入特征组合成张量features
    features = layers.Concatenate()([title, text_body, tags])
    # 利用中间层，将输入特征重组为更加丰富的表示
    features = layers.Dense(64, activation="relu")(features)

    # 定义模型输出
    priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
    department = layers.Dense(num_departments, activation="softmax", name="department")(features)

    # 通过指定输入和输出来创建模型
    model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])

    # 通过给定输入和目标组成的列表来训练模型
    num_samples = 1280

    title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

    priority_data = np.random.random(size=(num_samples, 1))
    department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

    model.compile(optimizer="rmsprop",
                  loss=["mean_squared_error", "categorical_crossentropy"],
                  metrics=[["mean_absolute_error"], ["accuracy"]])
    model.fit([title_data, text_body_data, tags_data],
              [priority_data, department_data],
              epochs=1)
    model.evaluate([title_data, text_body_data, tags_data],
                   [priority_data, department_data])
    priority_preds, department_preds = model.predict(
        [title_data, text_body_data, tags_data]
    )

    keras.utils.plot_model(model, "ticket_classifier.png")


if __name__ == "__main__":
    functional_api()
