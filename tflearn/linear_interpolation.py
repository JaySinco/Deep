import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

sess = tf.Session()
saver = tf.train.import_meta_graph('checkpoint/multilayer_perceptron.ckpt.meta')
saver.restore(sess, 'checkpoint/multilayer_perceptron.ckpt')
var_list_opt = [None, None, None, None, None, None]
var_list_name = {
    'h1/W:0': 0,
    'h1/b:0': 1,
    'h2/W:0': 2,
    'h2/b:0': 3,
    'output/W:0': 4,
    'output/b:0': 5
}

for x in tf.trainable_variables():
    if x.name in var_list_name:
        index = var_list_name[x.name]
        var_list_opt[index] = x

rW1 = tf.get_variable('W1', [784, 256], initializer=tf.random_normal_initializer(stddev=(2.0/784)**0.5))
rb1 = tf.get_variable('b1', [256], initializer=tf.constant_initializer(value=0.1))
rW2 = tf.get_variable('W2', [256, 256], initializer=tf.random_normal_initializer(stddev=(2.0/256)**0.5))
rb2 = tf.get_variable('b2', [256], initializer=tf.constant_initializer(value=0.1))
roW = tf.get_variable('Wo', [256, 10], initializer=tf.random_normal_initializer(stddev=(2.0/256)**0.5))
rob = tf.get_variable('bo', [10], initializer=tf.constant_initializer(value=0.1))

alpha = tf.placeholder(tf.float32)
beta = 1 - alpha
iW1 = var_list_opt[0] * beta + rW1 * alpha
ib1 = var_list_opt[1] * beta + rb1 * alpha
iW2 = var_list_opt[2] * beta + rW2 * alpha
ib2 = var_list_opt[3] * beta + rb2 * alpha
ioW = var_list_opt[4] * beta + roW * alpha
iob = var_list_opt[5] * beta + rob * alpha

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
ih1 = tf.nn.relu(tf.matmul(x, iW1) + ib1)
ih2 = tf.nn.relu(tf.matmul(ih1, iW2) + ib2)
io = tf.nn.relu(tf.matmul(ih2, ioW) + iob)
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=io, labels=y)
icost = tf.reduce_mean(xentropy)

sess.run(tf.global_variables_initializer())

xs = []
results = []
for a in np.arange(-5, 5, 0.01):
    feed_dict = {
        x: mnist.test.images,
        y: mnist.test.labels,
        alpha: a,
    }
    xs.append(a)
    results.append(sess.run(icost, feed_dict=feed_dict))

plt.plot(xs, results, 'ro')
plt.ylabel('Incurred Error')
plt.xlabel('Alpha')
plt.show()