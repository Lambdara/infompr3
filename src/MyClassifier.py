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

from src.utils import *
from src.ops import *


MODEL_DIR = '../checkpoint'
LOG_DIR = '../log'

DATA_DIR = '../data'
DATASET_NAME_DOGS = 'dogs'
DATASET_NAME_CATS = 'cats'
INPUT_FNAME_PATTERN = '*.png'

IMAGE_H = 64
IMAGE_W = 64
N_CLASS = 2

test_size = 0.20
learning_rate = 0.01
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

# # Look at sample image
# im = get_image(X[0], IMAGE_H, IMAGE_W, resize_height=IMAGE_H, resize_width=IMAGE_W, grayscale=False)
# print(im)
# print(type(im))
# print(np.shape(im))
# print(len(im))

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
x = tf.placeholder(tf.float32, [batch_size, IMAGE_H, IMAGE_W, n_channel])
y_ = tf.placeholder(tf.float32, [batch_size, N_CLASS])

h0 = maxpool2d(relu(conv2d(x, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h0")), 5)
h1 = maxpool2d(relu(conv2d(h0, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h1")), 5)
h2 = maxpool2d(relu(conv2d(h1, 128, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h2")), 5)
h3 = maxpool2d(relu(conv2d(h2, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h3")), 5)
h4 = maxpool2d(relu(conv2d(h3, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_h4")), 5)
h5 = dropout(relu(linear(h4, 1024)), rate=0.8)
y = linear(h5, 2)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

# create batches..
batch_idx = len(test_x) // batch_size

for epoch in range(epochs):
    # for batch in batches
        # run..


