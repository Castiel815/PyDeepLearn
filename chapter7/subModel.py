from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# 简单的子类化模型
class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax"
        )

    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department


def sub_model():
    vocabulary_size = 10000
    num_tags = 100
    num_departments = 4

    # 通过给定输入和目标组成的列表来训练模型
    num_samples = 1280
    title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))
    priority_data = np.random.random(size=(num_samples, 1))
    department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

    model = CustomerTicketModel(num_departments=4)
    priority, department = model({
        "title": title_data, "text_body": text_body_data, "tags": tags_data
    })
    model.compile(optimizer="rmsprop",
                  loss=["mean_squared_error", "categorical_crossentropy"],
                  metrics=[["mean_absolute_error"], ["accuracy"]])
    model.fit({
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
    }, [priority_data, department_data])
    model.evaluate({"title": title_data,
                    "text_body": text_body_data,
                    "tags": tags_data},
                   [priority_data, department_data])
    priority_preds, department_preds = model.predict({"title": title_data,
                                                      "text_body": text_body_data,
                                                      "tags": tags_data})


if __name__ == "__main__":
    sub_model()
    print(1)
