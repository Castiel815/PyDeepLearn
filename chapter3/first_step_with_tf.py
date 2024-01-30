import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 在二维平面上随机生成两个类别的点
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)

# 将两个类别堆叠成一个形状位(2000, 2)的数组
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

# 生成对应的目标标签(0和1)
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))

# 绘制两个点类的图像
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

# 创建线性分类器的变量
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


# 前向传播函数
def model(inputs):
    return tf.matmul(inputs, W) + b


# 均方误差损失函数
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


# 训练步骤函数
learning_rate = 0.1


def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss


def test():
    # 批量训练循环
    for step in range(40):
        loss = training_step(inputs, targets)
        print(f"Loss at step {step}: {loss:.4f}")

    predictions = model(inputs)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    plt.show()

    x = np.linspace(-1, 4, 100)
    y = -W[0] / W[1] * x + (0.5 - b) / W[1]
    plt.plot(x, y, "-r")
    plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    plt.show()


if __name__ == "__main__":
    test()
