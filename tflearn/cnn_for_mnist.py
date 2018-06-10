''' Test Accuracy:  0.9916 '''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def layer(input, weight_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', weight_shape, initializer=w_init)
    b = tf.get_variable('b', [weight_shape[1]], initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)

def conv2d(input, weight_shape):
    weight_stddev = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2/weight_stddev)**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', weight_shape, initializer=weight_init)
    b = tf.get_variable('b', [weight_shape[3]], initializer=bias_init)
    conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(conv_out, b))

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def inference(x, keep_prob):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    with tf.variable_scope('conv_1'):
        conv_1 = conv2d(x, [5, 5, 1, 32])
        pool_1 = max_pool(conv_1)
    with tf.variable_scope('conv_2'):
        conv_2 = conv2d(pool_1, [5, 5, 32, 64])
        pool_2 = max_pool(conv_2)
    with tf.variable_scope('fc'):
        pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])
        fc_1 = layer(pool_2_flat, [7*7*64, 1024])
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
    with tf.variable_scope('output'):
        output = layer(fc_1_drop, [1024, 10])
    return output

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

is_train = False
is_load_from_ckpt = True
training_epochs = 3
batch_size = 50

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
with tf.variable_scope('cnn') as scope:
    output_train = inference(x, 0.5)
    scope.reuse_variables()
    output_real = inference(x, 1)

def check(index):
    guess = sess.run(tf.argmax(output_real, axis=1), feed_dict={x: [mnist.test.images[index]]})[0]
    print('MNIST test set with index %d guessed as %d' % (index, guess))
    plt.imshow(np.reshape(mnist.test.images[index], [28, 28]), cmap='gray')
    plt.show()

accuracy_real = evaluate(output_real, y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_train, labels=y))
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(cost)

checkpoint_filename = "checkpoint/cnn_for_mnist.ckpt"
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if is_load_from_ckpt:
    saver.restore(sess, checkpoint_filename)

if is_train:
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: mbatch_x, y: mbatch_y})
        saver.save(sess, checkpoint_filename)
        print(epoch+1, "Validation Accuracy: ", sess.run(accuracy_real,
            feed_dict={x: mnist.validation.images, y: mnist.validation.labels}))
    print("Optimization Finished!")

acy1 = sess.run(accuracy_real, feed_dict={x: mnist.test.images[:5000], y: mnist.test.labels[:5000]})
acy2 = sess.run(accuracy_real, feed_dict={x: mnist.test.images[5001:], y: mnist.test.labels[5001:]})
print("Test Accuracy: ", (acy1*5000+acy2*5000)/10000)
