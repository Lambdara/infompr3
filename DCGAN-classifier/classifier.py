from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class Classifier(object):
    def __init__(self, sess, input_height=108, input_width=108, resize_height=None, resize_width=None, crop=True,
                 batch_size=64, cf_dim=64, dataset_dogs='dogs', dataset_cats='cats',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, data_dir='./data'):
        """

            Args:
              sess: TensorFlow session
              batch_size: The size of batch. Should be specified before training.
              y_dim: (optional) Dimension of dim for y. [None]
              df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
              dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            """
        self.sess = sess
        self.crop = crop
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width

        if not resize_height:
            self.resize_height = resize_height
            self.resize_width = resize_width
        else:
            self.resize_height = input_height
            self.resize_width = input_width

        self.cf_dim = cf_dim        # Number of filters of first convolutional layer

        self.c_bn1 = batch_norm(name='c_bn1')
        self.c_bn2 = batch_norm(name='c_bn2')
        self.c_bn3 = batch_norm(name='c_bn3')

        self.dataset_name = dataset_dogs + dataset_cats
        self.dataset_dogs = dataset_dogs
        self.dataset_cats = dataset_cats

        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir

        # Get dog data
        data_dogs_path = os.path.join(self.data_dir, self.dataset_dogs, self.input_fname_pattern)
        data_dogs = glob(data_dogs_path)
        if len(data_dogs) == 0:
            raise Exception("[!] No data found in '" + data_dogs_path + "'")

        # Get cat data
        data_cats_path = os.path.join(self.data_dir, self.dataset_cats, self.input_fname_pattern)
        data_cats = glob(data_cats_path)
        if len(data_cats) == 0:
            raise Exception("[!] No data found in '" + data_cats_path + "'")

        # Concatinate the data
        self.labels = np.append(np.ones_like(data_dogs, dtype=float), np.zeros_like(data_cats, dtype=float))
        self.data = np.array(data_dogs + data_cats)

        if len(self.data) < self.batch_size:
            raise Exception("[!] Entire dataset size is less than the configured batch_size")

        shuffle_idx = np.random.permutation(len(self.data))
        self.data, self.labels = self.data[shuffle_idx], self.labels[shuffle_idx]

        # Check if image is a non-grayscale image by checking channel number
        imreadImg = imread(self.data[0])
        if len(imreadImg.shape) >= 3:
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1
        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='images')
        inputs = self.inputs
        self.C, self.C_logits = self.model(inputs, reuse=False)
        self.c_sum = histogram_summary("c", self.C)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.y = tf.placeholder(
            tf.float32, [self.batch_size, 1], name='true_labels')
        y = self.y

        self.c_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.C_logits, y))
        self.c_loss_sum = scalar_summary("c_loss", self.c_loss)

        t_vars = tf.trainable_variables()

        self.vars = [var for var in t_vars if 'c_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.c_loss, var_list=self.vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.c_sum = merge_summary([self.c_sum, self.c_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            data_dogs_path = os.path.join(self.data_dir, self.dataset_dogs, self.input_fname_pattern)
            data_dogs = glob(data_dogs_path)
            data_cats_path = os.path.join(self.data_dir, self.dataset_cats, self.input_fname_pattern)
            data_cats = glob(data_cats_path)

            self.labels = np.append(np.ones_like(data_dogs, dtype=float), np.zeros_like(data_cats, dtype=float))
            self.data = np.array(data_dogs + data_cats)

            shuffle_idx = np.random.permutation(len(self.data))
            self.data, self.labels = self.data[shuffle_idx], self.labels[shuffle_idx]

            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, int(batch_idxs)):
                batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [
                    get_image(batch_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.resize_height,
                              resize_width=self.resize_width,
                              crop=self.crop,
                              grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_labels = np.reshape(self.labels[idx * config.batch_size:(idx + 1) * config.batch_size],
                                          [config.batch_size, 1])

                # Update network
                _, summary_str = self.sess.run([optim, self.c_sum],
                                               feed_dict={self.inputs: batch_images, self.y: batch_labels})
                self.writer.add_summary(summary_str, counter)

                err = self.c_loss.eval({self.inputs: batch_images, self.y: batch_labels})

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, c_loss: %.8f" \
                      % (epoch, config.epoch, idx, batch_idxs,
                         time.time() - start_time, err))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def model(self, image, reuse=False):
        with tf.variable_scope("classifier") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.cf_dim, name='c_h0_conv'))
            h1 = lrelu(self.c_bn1(conv2d(h0, self.cf_dim * 2, name='c_h1_conv')))
            h2 = lrelu(self.c_bn2(conv2d(h1, self.cf_dim * 4, name='c_h2_conv')))
            h3 = lrelu(self.c_bn3(conv2d(h2, self.cf_dim * 8, name='c_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'c_h4_lin')

            return tf.nn.sigmoid(h4), h4

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.resize_height, self.resize_width)

    def save(self, checkpoint_dir, step):
        model_name = "classifier.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
