''' Test Accuracy:  0.733 '''
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

class CIFAR10:
    def __init__(self, dirpath='data/cifar-10-batches-py/'):
        self.ptr = 0
        self.cur_file_no = 1
        self.dirpath = dirpath
        with open(self.dirpath+'batches.meta', 'rb')as fo:
            meta = pickle.load(fo, encoding='bytes')
            self.category = [bytes.decode(label) for label in meta[b'label_names']]
            self.num_per_file = meta[b'num_cases_per_batch']
        self.cur_imgX, self.cur_imgY = self.read_file('data_batch_1')
        self.test_imgX, self.test_imgY = self.read_file('test_batch')

    def next_train_batch(self, n):
        assert n < self.num_per_file
        batchX, batchY = [], []
        if self.ptr + n > self.num_per_file - 1:
            batchX = self.cur_imgX[self.ptr:, :, :, :]
            batchY = self.cur_imgY[self.ptr:, :]
            self.cur_file_no = 1 if self.cur_file_no+1 > 5 else self.cur_file_no+1
            self.cur_imgX, self.cur_imgY = self.read_file('data_batch_'+str(self.cur_file_no))
            self.ptr = self.ptr + n - self.num_per_file
            batchX = np.concatenate((batchX, self.cur_imgX[:self.ptr, :, :, :]))
            batchY = np.concatenate((batchY, self.cur_imgY[:self.ptr, :]))
        else:
            batchX = self.cur_imgX[self.ptr:self.ptr+n, :, :, :]
            batchY = self.cur_imgY[self.ptr:self.ptr+n, :]
            self.ptr = self.ptr + n
        return batchX, batchY

    def read_file(self, filename='data_batch_1'):
        def one_hot(index):
            item = np.zeros(10)
            item[index] = 1
            return item
        with open(self.dirpath+filename, 'rb')as fo:
            data = pickle.load(fo, encoding='bytes')
            imgX = np.transpose(np.reshape(data[b'data'], [self.num_per_file, 3, 32, 32]), [0, 2, 3, 1])
            imgY = np.array(data[b'labels'])
        return (imgX/255, np.array(list(map(one_hot, imgY))))

cifar10 = CIFAR10('data/cifar-10-batches-py/')

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
    return tf.nn.relu(layer_batch_norm(logits, phase_train))

def conv2d(input, weight_shape, phase_train):
    weight_stddev = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2/weight_stddev)**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', weight_shape, initializer=weight_init)
    b = tf.get_variable('b', [weight_shape[3]], initializer=bias_init)
    conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
    logits = tf.nn.bias_add(conv_out, b)
    return tf.nn.relu(layer_batch_norm(logits, phase_train))

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def inference(x, phase_train):
    with tf.variable_scope('conv_1'):
        conv_1 = conv2d(x, [5, 5, 3, 64], phase_train)
        pool_1 = max_pool(conv_1)
    with tf.variable_scope('conv_2'):
        conv_2 = conv2d(pool_1, [5, 5, 64, 64], phase_train)
        pool_2 = max_pool(conv_2)
    with tf.variable_scope('fc_1'):
        dim = 1
        for d in pool_2.get_shape()[1:].as_list():
            dim *= d
        pool_2_flat = tf.reshape(pool_2, [-1, dim])
        fc_1 = layer(pool_2_flat, [dim, 384], phase_train)
    with tf.variable_scope('fc_2'):
        fc_2 = layer(fc_1, [384, 192], phase_train)
    with tf.variable_scope('output'):
        output = layer(fc_2, [192, 10], phase_train)
    return output

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


is_load_from_ckpt = True

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
phase_train = tf.placeholder(tf.bool)

output = inference(x, phase_train)
accuracy_op = evaluate(output, y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(0.01)
train_op = optimizer.minimize(cost)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

checkpoint_filename = 'checkpoint/cnn_with_bn_for_cifar10.ckpt'
if is_load_from_ckpt:
    saver.restore(sess, checkpoint_filename)

def check(index):
    guess = sess.run(tf.argmax(output, axis=1),
        feed_dict={x: [cifar10.test_imgX[index]],  phase_train: False})[0]
    print("CIFAR10 test with index [%d] guessed as '%s', actually is '%s'"
        %(index, cifar10.category[guess], cifar10.category[np.argmax(cifar10.test_imgY[index])]))
    plt.imshow(cifar10.test_imgX[index])
    plt.show()

def test(fr=0, to=1000):
    print("Test Accuracy[{}-{}]: ".format(fr, to), sess.run(accuracy_op,
            feed_dict={x: cifar10.test_imgX[fr:to, :, :, :],
            y: cifar10.test_imgY[fr:to, :], phase_train: False}))

def train(training_epochs=1, batch_size=128):
    for epoch in range(training_epochs):
        total_batch = int(cifar10.num_per_file * 5/ batch_size)
        for i in range(total_batch):
            mbatch_x, mbatch_y = cifar10.next_train_batch(batch_size)
            sess.run(train_op, feed_dict={x: mbatch_x, y: mbatch_y, phase_train: True})
        saver.save(sess, checkpoint_filename)
        test(0, 1000)
    print("Optimization Finished!")


def visual_first_layer_filter():
    with tf.variable_scope('conv_1') as scope:
        scope.reuse_variables()
        w_raw = sess.run(tf.get_variable('W'))
        w = np.transpose(w_raw, axes=[3, 0, 1, 2])
        for i in range(w.shape[0]):
            plt.imshow(w[i])
            plt.show()
