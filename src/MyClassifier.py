from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import data
import os
from glob import glob

from utils import *


DATA_DIR = '../data'
DATASET_NAME_DOGS = 'dogs'
DATASET_NAME_CATS = 'cats'
INPUT_FNAME_PATTERN = '*.png'

test_size = 0.20


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

    Y = np.array([[1, 0]] * len(data_cats) + [[1, 0]] * len(data_dogs))
    X = np.array(data_dogs + data_cats)
    return X, Y


X, Y = load_data()

X, Y = shuffle(X, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size= test_size, random_state=415)

print(get_image(train_x[0], 64, 64))















def load_data_old():
    cats, dogs = data.get_cats_and_dogs()
    X = np.concatenate([cats, dogs])
    Y = np.array([[1,0]] *len(cats) + [[0,1]] * len(dogs))
    return X, Y