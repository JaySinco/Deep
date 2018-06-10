''' Test Accuracy:  0.xxx '''
import tensorflow as tf
import numpy as np
import sys
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

class IMDBDataset:
    def __init__(self, datapath='data/imdb/imdb.pkl'):
        (trainX, trainY), (testX, testY), _ = imdb.load_data(path=datapath, n_words=30000)
        self.trainX = pad_sequences(trainX, maxlen=250, value=0.)
        self.trainY = to_categorical(trainY, nb_classes=2)
        self.testX = pad_sequences(testX, maxlen=250, value=0.)    
        self.testY = to_categorical(testY, nb_classes=2)
        self.num_examples = len(trainX)
        self.ptr = 0

    def next_train_batch(self, size):
        batchX, batchY = [], []
        if self.ptr + size < len(self.trainX):
            batchX = self.trainX[self.ptr:self.ptr+size]
            batchY = self.trainY[self.ptr:self.ptr+size]
        else:
            batchX = np.concatenate((self.trainX[self.ptr:], self.trainX[:size-len(self.trainX[self.ptr:])]))
            batchY = np.concatenate((self.trainY[self.ptr:], self.trainY[:size-len(self.trainY[self.ptr:])]))
        self.ptr = (self.ptr + size) % len(self.trainX)
        return batchX, batchY

imdb = IMDBDataset()

def embedding_layer(input, weight_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    E = tf.get_variable('E', weight_shape, initializer=w_init)
    embedding = tf.nn.embedding_lookup(E, tf.cast(input, tf.int32))
    return embedding

def lstm(input, hidden_dim):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
    dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.5)
    outputs, states = tf.nn.dynamic_rnn(dropout_lstm, input, dtype=tf.float32)
    # shape: [batch_size, timestep_size, hidden_dim] 
    return outputs[:, -1, :]

def layer(input, weight_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', weight_shape, initializer=w_init)
    b = tf.get_variable('b', [weight_shape[1]], initializer=bias_init)
    return tf.nn.relu( tf.matmul(input, W) + b)

def inference(x):
    with tf.variable_scope('embedding'):
        embedding = embedding_layer(x, [30000, 50])
    with tf.variable_scope('lstm'):
        lstm_output = lstm(embedding, 50)
    with tf.variable_scope('fc'):
        output = layer(lstm_output, [50, 2])
    return output

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


is_load_from_ckpt = False

x = tf.placeholder(tf.float32, [None, 250])
y = tf.placeholder(tf.float32, [None, 2])

output = inference(x)
accuracy_op = evaluate(output, y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cost)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

checkpoint_filename = 'checkpoint/lstm_rnn_for_imdb.ckpt'
if is_load_from_ckpt:
    saver.restore(sess, checkpoint_filename)

def train(training_epochs=1, batch_size=32):
    for epoch in range(training_epochs):
        total_batch = int(imdb.num_examples / batch_size)
        for i in range(total_batch):
            mbatch_x, mbatch_y = imdb.next_train_batch(batch_size)
            sess.run(train_op, feed_dict={x: mbatch_x, y: mbatch_y})
           # if i%50 == 0:
            print("[{:2d}] {:4d}/{:4d}  Test Accuracy: {:.3f}".format(epoch+1, i+1, total_batch,
                sess.run(accuracy_op, feed_dict={x: imdb.testX, y: imdb.testY})))
            #saver.save(sess, checkpoint_filename)
    print("Optimization Finished!")