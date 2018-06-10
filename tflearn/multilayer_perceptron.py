''' Test Accuracy:  0.971 '''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def layer(input, weight_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0.1)
    W = tf.get_variable('W', weight_shape, initializer=w_init)
    b = tf.get_variable('b', [weight_shape[1]], initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)

is_train = False
learning_rate = 0.1
training_epochs = 30
batch_size = 100

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
with tf.variable_scope("h1"):
    h1 = layer(x, [784, 256])
with tf.variable_scope("h2"):
    h2 = layer(h1, [256, 256])
with tf.variable_scope("output"):
    output = layer(h2, [256, 10])
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
cost = tf.reduce_mean(xentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if is_train:
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: mbatch_x, y: mbatch_y})
        saver.save(sess, "checkpoint/multilayer_perceptron.ckpt")
        print(epoch+1, "Validation Accuracy: ", sess.run(accuracy,
            feed_dict={x: mnist.validation.images, y: mnist.validation.labels}))
    print("Optimization Finished!")

if not is_train:
    saver.restore(sess, "checkpoint/multilayer_perceptron.ckpt")

print("Test Accuracy: ", sess.run(accuracy,
    feed_dict={x: mnist.test.images, y: mnist.test.labels}))
