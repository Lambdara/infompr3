import tensorflow as tf
import numpy as np
import data
from PIL import Image

(sample_data,_),_ = tf.keras.datasets.mnist.load_data()
sample_data = sample_data.reshape(-1,28*28)/255*2-1

gen_learning_rate = 0.0001
dis_learning_rate = 0.0001
noise_size = 8

def load_gan():
    global disc_step, disc_loss, gen_step, gen_loss, sess, gen, Z, X

    tf.reset_default_graph()

    def generator(Z,hsize=[28,10],reuse=False):
        with tf.variable_scope("GAN/Generator",reuse=reuse):
            h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.sigmoid)
            h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.sigmoid)
            out = tf.layers.dense(h2,28*28)
        return out

    def discriminator(X,hsize=[128,60],reuse=False):
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.sigmoid)
            h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.sigmoid)
            h3 = tf.layers.dense(h2,12,activation=tf.nn.sigmoid)
            out = tf.layers.dense(h3,1)
        return out

    X = tf.placeholder(tf.float32,[None,28*28])
    Z = tf.placeholder(tf.float32,[None,noise_size])

    gen = generator(Z)
    reals = discriminator(X)
    fakes = discriminator(gen,reuse=True)

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reals,labels=tf.ones_like(reals)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=fakes,labels=tf.zeros_like(fakes)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fakes,labels=tf.ones_like(fakes)))

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    gen_step = tf.train.RMSPropOptimizer(learning_rate=gen_learning_rate).minimize(gen_loss,var_list = gen_vars) # G Train step
    disc_step = tf.train.RMSPropOptimizer(learning_rate=dis_learning_rate).minimize(disc_loss,var_list = disc_vars) # D Train step

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

def run(steps,batch_size=10):
    for i in range(steps):
        X_batch = sample_data[np.random.randint(len(sample_data),size=batch_size),:]
        Z_batch = np.random.uniform(size=(batch_size,noise_size))
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    print('done')
        #print ("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))


def get_image(path):
    Image.fromarray((((gen.eval(session=sess,feed_dict={Z: np.random.uniform(size=(1,noise_size))})[0]+1)/2)*255).reshape(28,28).astype(np.uint8)).save(path)


def go(prefix):
    i = 1
    while True:
        get_image(prefix+str(i)+'.png')
        i = i + 1
        run(10,10)

load_gan()
