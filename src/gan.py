import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data
from PIL import Image

cats,_ = data.get_cats_and_dogs()

def generator(Z,hsize=[64,64],reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,3072)
    return out

def discriminator(X,hsize=[64,64],reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,1)
    return out

sample_data = np.array([img.reshape(3072)/255*2-1 for img in cats])

X = tf.placeholder(tf.float32,[None,3072])
Z = tf.placeholder(tf.float32,[None,3072])

gen = generator(Z)
reals = discriminator(X)
fakes = discriminator(gen,reuse=True)

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reals,labels=tf.ones_like(reals)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=fakes,labels=tf.zeros_like(fakes)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakes,labels=tf.ones_like(fakes)))

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

def run(steps,batch_size=10):
    for i in range(steps):
        X_batch = sample_data[np.random.randint(len(cats),size=batch_size),:]
        Z_batch = np.random.uniform(size=(batch_size,3072))
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

        print ("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))

def get_image(path):
    Image.fromarray(gen.eval(session=sess,feed_dict={Z: np.random.uniform(size=(1,3072))})[0].reshape(32,32,3).astype(np.uint8)).save(path)
