from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression

import os
from glob import glob

from utils import *
from ops import *


MODEL_DIR = './checkpoint'
LOG_DIR = './log'

DATA_DIR = './data'
DATASET_NAME_DOGS = 'dogs'
DATASET_NAME_CATS = 'cats'
INPUT_FNAME_PATTERN = '*.png'

IMAGE_H = 64
IMAGE_W = 64
N_CLASS = 2

test_size = 0.20
learning_rate = 0.0001
epochs = 10
batch_size = 64


def load_data():
    path = os.path.join(DATA_DIR, DATASET_NAME_DOGS, INPUT_FNAME_PATTERN)
    data_dogs = glob(path)
    if len(data_dogs) == 0:
        raise Exception("[!] No data found in '" + path + "'")

    # Get cat data
    path = os.path.join(DATA_DIR, DATASET_NAME_CATS, INPUT_FNAME_PATTERN)
    data_cats = glob(path)
    if len(data_cats) == 0:
        raise Exception("[!] No data found in '" + path + "'")

    Y = np.array([[1, 0]] * len(data_cats) + [[0, 1]] * len(data_dogs))
    X = np.array(data_dogs + data_cats)
    return X, Y


# Get the data
X, Y = load_data()

# Check for grayscale data
imreadImg = imread(X[0])
if len(imreadImg.shape) >= 3:
    n_channel = imread(X[0]).shape[-1]
else:
    n_channel = 1
grayscale = (n_channel == 1)

# Split in test and train set
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_size, random_state=415)

# Build model
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, IMAGE_H, IMAGE_W, n_channel], name='images')
y_ = tf.placeholder(tf.float32, [None, N_CLASS], name='true_labels')


def model(images, reuse=False):
    with tf.variable_scope("classifier", reuse=reuse):
        conv1 = tf.layers.conv2d(
            images,
            filters=32,
            kernel_size=[5, 5],
            padding="SAME",
            activation=tf.nn.relu
        )
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=[2, 2],
            strides=2
        )
        conv2 = tf.layers.conv2d(
            pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="SAME",
            activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=[2, 2],
            strides=2
        )
        conv3 = tf.layers.conv2d(
            pool2,
            filters=32,
            kernel_size=[5, 5],
            padding="SAME",
            activation=tf.nn.relu
        )
        pool3 = tf.layers.max_pooling2d(
            conv3,
            pool_size=[2, 2],
            strides=2
        )
        # conv4 = tf.layers.conv2d(
        #     pool3,
        #     filters=64,
        #     kernel_size=[5, 5],
        #     padding="SAME",
        #     activation=tf.nn.relu
        # )
        # pool4 = tf.layers.max_pooling2d(
        #     conv4,
        #     pool_size=[2, 2],
        #     strides=2
        # )
        # conv5 = tf.layers.conv2d(
        #     pool4,
        #     filters=32,
        #     kernel_size=[5, 5],
        #     padding="SAME",
        #     activation=tf.nn.relu
        # )
        # pool5 = tf.layers.max_pooling2d(
        #     conv5,
        #     pool_size=[2, 2],
        #     strides=2
        # )
        reshape5 = tf.reshape(pool3, [-1, 2048])
        h6 = tf.layers.dropout(tf.layers.dense(reshape5, 256, activation=tf.nn.relu), rate=0.8)
        out = tf.layers.dense(h6, 2, activation=None)
    return out

y = model(x)

# with tf.variable_scope('model') as scope:
#     h0 = maxpool2d(relu(conv2d(x, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h0")), 2)
#     h1 = maxpool2d(relu(conv2d(h0, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h1")), 2)
#     h2 = maxpool2d(relu(conv2d(h1, 128, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h2")), 2)
#     h3 = maxpool2d(relu(conv2d(h2, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h3")), 2)
#     h4 = maxpool2d(relu(conv2d(h3, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h4")), 2)
#     h5 = dropout(relu(linear(tf.reshape(h4, [batch_size, -1]), 1024, scope='lin_h5')), rate=0.8)
#     y = linear(h5, 2)

saver = tf.train.Saver()

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
training_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(cost_function)

# Metrics
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
mse = tf.reduce_mean(tf.square(y - y_))

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

# create batches..
batch_idx_train = len(train_x) // batch_size
batch_idx_test = len(test_x) // batch_size

for epoch in range(epochs):
    for idx in range(batch_idx_train):
        batch_images, batch_labels = get_batch(train_x, train_y, batch_size, idx, IMAGE_H, IMAGE_W, grayscale=grayscale)
        _, batch_cost = sess.run([training_step, cost_function], feed_dict={x: batch_images, y_: batch_labels})
        batch_mse = sess.run(mse, feed_dict={x: batch_images, y_: batch_labels})
        batch_accuracy = sess.run(accuracy, feed_dict={x: batch_images, y_: batch_labels})
        print('epoch:', epoch, ' - batch:', idx, '/', batch_idx_train, ' - cost:', batch_cost, ' - MSE:', batch_mse, ' - Train Accuracy:' ,
              batch_accuracy)

    test_X, test_Y = get_batch(test_x, test_y, len(test_x), 0, IMAGE_H, IMAGE_W, grayscale=grayscale)
    acc = sess.run(accuracy, feed_dict={x: test_X, y_: test_Y})
    print('Test Accuracy:', acc)

