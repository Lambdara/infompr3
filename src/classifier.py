import tensorflow as tf
import numpy as np
import data
from PIL import Image

test_size = 500
learning_rate = 0.0001

def load_data():
    global training_data, training_labels, test_data, test_labels

    cats,dogs = data.get_cats_and_dogs()

    training_cats = cats[:-test_size]
    training_dogs = dogs[:-test_size]
    training_data = np.concatenate([training_cats, training_dogs]).reshape(-1,32*32*3)/255
    training_labels = np.array([[1,0]] * len(training_cats) + [[0,1]] * len(training_dogs))

    test_cats = cats[-test_size:]
    test_dogs = dogs[-test_size:]
    test_data = np.concatenate([test_cats, test_dogs]).reshape(-1,32*32*3)/255
    test_labels = np.array([[1,0]] * len(test_cats) + [[0,1]] * len(test_dogs))


if 'training_data' not in globals() or 'training_labels' not in globals():
    load_data()


def load_gan():
    global optimizer, loss, sess, classifier, x, y

    tf.reset_default_graph()

    def classifier_network(X,reuse=False):
        with tf.variable_scope("classifier",reuse=reuse):
            shape1 = tf.reshape(X, [-1, 32, 32, 3])
            conv1 = tf.layers.conv2d(
                shape1,
                filters=16,
                kernel_size=[4,4],
                padding="same",
                activation=tf.nn.relu
            )
            pool1 = tf.layers.max_pooling2d(
                conv1,
                pool_size=[2,2],
                strides=2
            )
            conv2 = tf.layers.conv2d(
                pool1,
                filters=16,
                kernel_size=[4,4],
                padding="same",
                activation=tf.nn.relu
                )
            pool2 = tf.layers.max_pooling2d(
                conv2,
                pool_size=[2,2],
                strides=2
            )
            shape2 = tf.reshape(pool2, [-1, 1024])
            h1 = tf.layers.dense(shape2,256,activation=tf.nn.sigmoid)
            h2 = tf.layers.dense(h1,64,activation=tf.nn.sigmoid)
            out = tf.layers.dense(h2,2)
        return out

    
    x = tf.placeholder(tf.float32,[None,3072])
    y = tf.placeholder(tf.float32,[None,2])

    classifier = classifier_network(x)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits = classifier,
            labels = y
        )
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)


load_gan()


def run(steps=10,batch_size=10):
    average_loss = 0
    global prediction

    for _ in range(steps):
        batch_indices = np.random.randint(len(training_data),size=batch_size)
        x_batch = training_data[batch_indices]
        y_batch = training_labels[batch_indices]
        _, classifier_loss = sess.run(
            [optimizer, loss],
            feed_dict = {
                x: x_batch,
                y: y_batch
            }
        )
        average_loss += classifier_loss/steps

    prediction = tf.argmax(classifier,1).eval(feed_dict={x:test_data}, session=sess)
    hits = 0
    for i in range(test_size*2):
        if prediction[i] == test_labels[i][1]:
            hits += 1
    accuracy = hits/(test_size*2)

    print("Loss:     " + str(classifier_loss))
    print("Accuracy: " + str(accuracy))

    return accuracy


def go():
    steps = 100
    batch_size = 100

    accuracy = 0
    while accuracy < 0.85:
        accuracy = run(steps,batch_size)

