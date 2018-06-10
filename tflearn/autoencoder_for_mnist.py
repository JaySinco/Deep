''' Test Cost: [RELU] 4.926; [SIGMOID] 4.971 '''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def layer_batch_norm(input, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
    beta = tf.get_variable('beta', [input.shape[-1]], initializer=beta_init)
    gamma = tf.get_variable('gamma', [input.shape[-1]], initializer=gamma_init)
    axises = list(range(len(input.shape) - 1))
    batch_mean, batch_var = tf.nn.moments(input, axises, name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train, mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
    return normed

def layer(input, weight_shape, phase_train):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', weight_shape, initializer=w_init)
    b = tf.get_variable('b', [weight_shape[1]], initializer=bias_init)
    logits = tf.matmul(input, W) + b
    # use tf.nn.sigmoid instead of tf.nn.relu to avoid sharp transitions
    return tf.nn.sigmoid(layer_batch_norm(logits, phase_train))

def encoder(input, phase_train):
    with tf.variable_scope('encoder'):
        with tf.variable_scope('fc_1'):
            fc_1 = layer(input, [784, 1000], phase_train)
        with tf.variable_scope('fc_2'):
            fc_2 = layer(fc_1, [1000, 500], phase_train)
        with tf.variable_scope('fc_3'):
            fc_3 = layer(fc_2, [500, 250], phase_train)
        with tf.variable_scope('code'):
            code = layer(fc_3, [250, 2], phase_train)
    return code

def decoder(code, phase_train):
    with tf.variable_scope('decoder'):
        with tf.variable_scope('fc_3'):
            fc_3 = layer(code, [2, 250], phase_train)
        with tf.variable_scope('fc_2'):
            fc_2 = layer(fc_3, [250, 500], phase_train)
        with tf.variable_scope('fc_1'):
            fc_1 = layer(fc_2, [500, 1000], phase_train)
        with tf.variable_scope('code'):
            output = layer(fc_1, [1000, 784], phase_train)
    return output


is_load_from_ckpt = True

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
phase_train = tf.placeholder(tf.bool)

code = encoder(x, phase_train)
output = decoder(code, phase_train)
l2 = tf.sqrt(tf.reduce_sum(tf.square(output-x), 1))
cost_op = tf.reduce_mean(l2)
optimizer = tf.train.AdamOptimizer(0.01)
train_op = optimizer.minimize(cost_op)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# checkpoint_filename = 'checkpoint/autoencoder_for_mnist_relu.ckpt'
checkpoint_filename = 'checkpoint/autoencoder_for_mnist_sigmoid.ckpt'
if is_load_from_ckpt:
    saver.restore(sess, checkpoint_filename)

def train(training_epochs=1, batch_size=128):
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: mbatch_x, phase_train: True})
        saver.save(sess, checkpoint_filename)
        print(epoch+1, "Validation Cost: ", sess.run(cost_op,
            feed_dict={x: mnist.validation.images, phase_train: False}))
    print("Optimization Finished!")

def check(index):
    cde = sess.run(code, feed_dict={x: [mnist.test.images[index]], phase_train: False})[0]
    fake = sess.run(output, feed_dict={x: [mnist.test.images[index]], phase_train: False})[0]
    print("MNIST test with index [{}] coded as ({:.3f}, {:.3f})".format(index, cde[0], cde[1]))
    plt.subplot(1, 2, 1)
    plt.title('Real')
    plt.imshow(np.reshape(mnist.test.images[index], [28, 28]), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Fake')
    plt.imshow(np.reshape(fake, [28, 28]), cmap='gray')
    plt.show()

def scatter():
    colors = [
        ('#27ae60', 'o'),
        ('#2980b9', 'o'),
        ('#8e44ad', 'o'),
        ('#f39c12', 'o'),
        ('#c0392b', 'o'),
        ('#27ae60', 'x'),
        ('#2980b9', 'x'),
        ('#8e44ad', 'x'),
        ('#c0392b', 'x'),
        ('#f39c12', 'x'),
    ]
    codes = sess.run(code, feed_dict={x: mnist.test.images, phase_train: False})
    codes_t = np.transpose(codes, [1, 0])
    labels = np.argmax(mnist.test.labels, axis=1)
    for n in range(10):
        plt.scatter(
            x=[codes_t[0][i] for i, label in enumerate(labels) if label == n],
            y=[codes_t[1][i] for i, label in enumerate(labels) if label == n],
            s=10,
            label=str(n),
            c=colors[n][0],
            marker=colors[n][1])
    plt.legend()
    plt.show()

