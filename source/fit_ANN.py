import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import random
import numpy as np

df = pd.read_csv('../data/housing_clean.csv')
features = [df.subway, df.bus_stop, df.accommodates, df.bathroom, df.bedroom, df.beds,
            df.guests, df.num_of_review, df.review_score, df.Entire_home, df.crime_rate]
X = pd.concat(features, axis=1).dropna().astype(dtype='float64', copy=False)
y = df.daily_price.dropna().astype(dtype='float64', copy=False)

X_sc = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)

sess = tf.Session()


def init_NeuralNetwork(_input, _classes):           # 神经网络架构初始化
    n_input = _input
    n_classes = _classes
    n_hidden1 = 16
    n_hidden2 = 20
    n_hidden3 = 28
    n_hidden4 = 32
    n_hidden5 = 16
    n_hidden6 = 8
    n_hidden7 = 5
    n_hidden8 = 3

    data = tf.placeholder("float", [None, n_input])
    label = tf.placeholder("float", [None, n_classes])

    weights = {
        "w1": tf.Variable(tf.random_normal([n_input, n_hidden1], stddev=0.1)),
        "w2": tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev=0.1)),
        "w3": tf.Variable(tf.random_normal([n_hidden2, n_hidden3], stddev=0.1)),
        "w4": tf.Variable(tf.random_normal([n_hidden3, n_hidden4], stddev=0.1)),
        "w5": tf.Variable(tf.random_normal([n_hidden4, n_hidden5], stddev=0.1)),
        "w6": tf.Variable(tf.random_normal([n_hidden5, n_hidden6], stddev=0.1)),
        "w7": tf.Variable(tf.random_normal([n_hidden6, n_hidden7], stddev=0.1)),
        "w8": tf.Variable(tf.random_normal([n_hidden7, n_hidden8], stddev=0.1)),
        "out": tf.Variable(tf.random_normal([n_hidden8, n_classes], stddev=0.1)),
    }
    biases = {
        "b1": tf.Variable(tf.random_normal([n_hidden1], stddev=0.1)),
        "b2": tf.Variable(tf.random_normal([n_hidden2], stddev=0.1)),
        "b3": tf.Variable(tf.random_normal([n_hidden3], stddev=0.1)),
        "b4": tf.Variable(tf.random_normal([n_hidden4], stddev=0.1)),
        "b5": tf.Variable(tf.random_normal([n_hidden5], stddev=0.1)),
        "b6": tf.Variable(tf.random_normal([n_hidden6], stddev=0.1)),
        "b7": tf.Variable(tf.random_normal([n_hidden7], stddev=0.1)),
        "b8": tf.Variable(tf.random_normal([n_hidden8], stddev=0.1)),
        "out": tf.Variable(tf.random_normal([n_classes], stddev=0.1)),
    }
    print("Neural Network ready!\n")

    return data, label, weights, biases

def forward_propagation(x, weights, biases):        # 前向传播计算
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights["w1"]), biases["b1"]))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights["w2"]), biases["b2"]))
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights["w3"]), biases["b3"]))
    layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, weights["w4"]), biases["b4"]))
    layer5 = tf.nn.relu(tf.add(tf.matmul(layer4, weights["w5"]), biases["b5"]))
    layer6 = tf.nn.relu(tf.add(tf.matmul(layer5, weights["w6"]), biases["b6"]))
    layer7 = tf.nn.relu(tf.add(tf.matmul(layer6, weights["w7"]), biases["b7"]))
    layer8 = tf.nn.relu(tf.add(tf.matmul(layer7, weights["w8"]), biases["b8"]))
    return tf.add(tf.matmul(layer8, weights["out"]), biases["out"])

def init_global():
    __pred = forward_propagation(Data, Weights, Biases)  # 前向传播得到预测值
    __cost = tf.sqrt(tf.reduce_mean(tf.square(__pred - Label)))  # 选取MSE作为损失函数
    # __cost = mse(Label, __pred)
    __optm = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(__cost)
    # __optm = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(__cost)  # 选取梯度下降作为优化器
    __correct = tf.reduce_mean(1 - (__cost / tf.squared_difference(x=Label, y=tf.reduce_mean(Label))))
    # __correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(__pred, 1), tf.argmax(Label, 1)), "float"))
    __init = tf.global_variables_initializer()
    print("Functions ready!\n")
    return __pred, __cost, __optm, __correct, __init

def training():
    train_epochs = 1000
    batch_size = 2000
    num_batch = int(len(X_train) / batch_size)

    for epoch in range(train_epochs):
        aver_cost = 0
        feeds = {}
        for i in range(num_batch):
            batch_x = random.sample(list(X_train), batch_size)
            batch_y = random.sample(list(np.reshape(y_train, (-1, 1))), batch_size)
            feeds = {Data: batch_x, Label: batch_y}

            sess.run(optm, feed_dict=feeds)
            aver_cost += sess.run(cost, feed_dict=feeds)
        aver_cost /= num_batch

        # 训练相关信息
        train_feeds = feeds
        test_feeds = {Data: X_test, Label: np.reshape(y_test, (-1, 1))}

        # pred_train = sess.run(pred, feed_dict=train_feeds)
        # train_mean = y_train.mean()
        # train_accr = 1 - np.sum(np.subtract(y_train, pred_train) ** 2) / np.sum(np.subtract(y_train, train_mean) ** 2)
        train_accr = sess.run(correct, feed_dict=train_feeds)
        test_accr = sess.run(correct, feed_dict=test_feeds)

        print(
            "Epoch: %03d / %03d cost: %.9f train_accuracy: %.4f test_accuracy: %.4f\n" %
            (epoch + 1, train_epochs, aver_cost, float(train_accr), test_accr))

        # result = sess.run(pred, feed_dict=test_feeds)
        # for i in range(len(result)):
        #     if y_test[i] < 200:
        #         print(result[i], y_test[i])


if __name__ == '__main__':
    # mnist = input_data.read_data_sets('./MNIST/', one_hot=True)
    Data, Label, Weights, Biases = init_NeuralNetwork(len(features), 1)
    pred, cost, optm, correct, init = init_global()

    sess.run(init)

    training()
