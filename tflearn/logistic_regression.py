''' Test Accuracy:  0.9211 '''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

is_train = False
learning_rate = 0.5
training_epochs = 3
batch_size = 100

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
init = tf.constant_initializer(value=0)
W = tf.get_variable("W", [784, 10], initializer=init)
b = tf.get_variable("b", [10], initializer=init)
output = tf.nn.softmax(tf.matmul(x, W) + b)
xentropy = -tf.reduce_sum(y * tf.log(output), reduction_indices=1)
cost = tf.reduce_mean(xentropy)
#tf.summary.scalar("cost", cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
global_step = tf.Variable(0, name='step', trainable=False)
train = optimizer.minimize(cost, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#tf.summary.scalar("error", 1.0 - accuracy)
#summary = tf.summary.merge_all()

sess = tf.Session()
#summary_writer = tf.summary.FileWriter("summary", graph=sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if is_train:
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: mbatch_x, y: mbatch_y})
            #summary_str = sess.run(summary, feed_dict={x: mbatch_x, y:  mbatch_y})
            #summary_writer.add_summary(summary_str, sess.run(global_step))
        saver.save(sess, "checkpoint/logistic_regression.ckpt")
        print(epoch+1, "Validation Accuracy: ", sess.run(accuracy,
            feed_dict={x: mnist.validation.images, y: mnist.validation.labels}))
    print("Optimization Finished!")

if not is_train:
    saver.restore(sess, "checkpoint/logistic_regression.ckpt")

print("Test Accuracy: ", sess.run(accuracy,
    feed_dict={x: mnist.test.images, y: mnist.test.labels}))
