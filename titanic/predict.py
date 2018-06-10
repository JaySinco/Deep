import tensorflow as tf
import numpy as np
import csv


def transform(raw_list):    
    pclass = int(raw_list[0])
    sex = 0 if raw_list[1] == 'male' else 1
    age = float('30' if raw_list[2] == '' else raw_list[2])
    sibsp = int(raw_list[3])
    parch = int(raw_list[4])
    fare = float('32' if raw_list[5] == '' else raw_list[5])
    embark = 1 if raw_list[6] == 'C' else (2 if raw_list[6] == 'Q' else (3 if raw_list[6] == 'S' else 0))
    return [pclass,sex,age,sibsp,parch,fare,embark]

class TrainSet:
    def __init__(self, file_name):
        self.X = []
        self.Y = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for index, row in enumerate(reader):
                if index == 0: 
                    continue
                self.X.append(transform([v for i,v in enumerate(row) if i in [2,4,5,6,7,9,11]]))
                self.Y.append([1, 0] if row[1] == "0" else [0, 1])
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        assert len(self.X) == len(self.Y)
        # ratio = int(len(self.X) * 0.8)
        ratio = int(len(self.X) * 1)
        self.X, self.Vx = self.X[:ratio], self.X[ratio:]
        self.Y, self.Vy = self.Y[:ratio], self.Y[ratio:]        
        self.ptr = 0
        self.len = len(self.X)

    def next_batch(self, n):
        assert n < self.len
        batch = []
        start, end = self.ptr, 0
        if self.ptr + n <= self.len:
            end = self.ptr + n
            batch = [self.X[start:end], self.Y[start:end]]      
        else:
            batch_1_x = self.X[start:self.len]
            batch_1_y = self.Y[start:self.len]
            end = self.ptr + n - self.len
            batch_2_x = self.X[0:end]
            batch_2_y = self.Y[0:end]
            batch = [np.concatenate((batch_1_x, batch_2_x)), np.concatenate((batch_1_y, batch_2_y))]
        assert len(batch[0]) == n and len(batch[1]) == n
        self.ptr = end
        return batch

class TestSet:
    def __init__(self, file_name):
        self.X = []
        self.psg_id = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for index, row in enumerate(reader):
                if index == 0: 
                    continue
                self.X.append(transform([v for i,v in enumerate(row) if i in [1,3,4,5,6,8,10]]))
                self.psg_id.append(row[0])
        self.X = np.array(self.X)
        self.len = len(self.X)

class Data:
    def __init__(self): 
        self.train = TrainSet('data/train.csv')
        self.test = TestSet('data/test.csv')


data = Data()

x = tf.placeholder(tf.float32, shape=[None, 7])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W1 = tf.Variable(tf.truncated_normal([7, 100], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[100]))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([100, 30], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[30]))
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
keep_prob = tf.placeholder(tf.float32)
h2_drop = tf.nn.dropout(h2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([30, 2], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[2]))
y  = tf.matmul(h2_drop, W3) + b3

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def train(loop_n=5000, batch_size=30, log_n=100):
    for i in range(loop_n):
        batch = data.train.next_batch(batch_size)
        if i % log_n == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: data.train.X, y_: data.train.Y, keep_prob: 1}) 
                # x: data.train.Vx, y_: data.train.Vy, keep_prob: 1}) 
            print("step %d, training accuracy %g" %(i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    sample_accuracy = sess.run(accuracy, feed_dict={
        x: data.train.X, y_: data.train.Y, keep_prob: 1}) 
    print("whole training sample: accuracy %g" %(sample_accuracy))

def submit(file_name='submission.csv'):
    prediction = sess.run(tf.argmax(y,1), feed_dict={x: data.test.X, keep_prob: 1})
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PassengerId','Survived'])
        for psg, suv in zip(data.test.psg_id, prediction):
            writer.writerow([psg, suv])

#sess.close()