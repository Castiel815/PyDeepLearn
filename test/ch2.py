from tensorflow.keras.datasets import mnist
import chapter2.naiveDense as naiveDense
import numpy as np
import chapter2.batchGenerator as batchGenerator


def test_ch2():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    batchGenerator.fit(naiveDense.model, train_images, train_labels, epochs=10, batch_size=128)

    predictions = naiveDense.model(test_images)
    predictions = predictions.numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    print(f"accuracy: {matches.mean():.2f}")


if __name__ == "__main__":
    test_ch2()
