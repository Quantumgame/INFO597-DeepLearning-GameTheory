import tensorflow as tf
import numpy as np
from tqdm import tqdm


class LSTM:

    def __init__(self, num_hidden=25):

        self.predict_results, self.train_error, self.test_error = [], [], []

        self.data = tf.placeholder(tf.float32, [None, 6, 1])
        self.target = tf.placeholder(tf.float32, [None, 2])
        self.num_hidden = num_hidden

        cell = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)

        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([num_hidden, int(self.target.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        self._pred = tf.argmax(prediction, 1)
        cross_entropy = -tf.reduce_sum(self.target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

        optimizer = tf.train.AdamOptimizer()
        self.minimize = optimizer.minimize(cross_entropy)

        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(prediction, 1))
        self.error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def reset(self):
        tf.reset_default_graph()
        del self.predict_results
        del self.train_error
        del self.test_error
        self.predict_results, self.train_error, self.test_error = [], [], []


    def run(self, features, labels, test_features=None, test_labels=None, batch_size=1000, epoch=5000):
        sess = tf.Session()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        no_of_batches = int(len(features)/batch_size)


        for i in tqdm(range(epoch), total=epoch, desc='Epoch'):
            ptr = 0
            avg_batch_error = 0
            for j in range(no_of_batches):
                inp, out = features[ptr:ptr+batch_size], labels[ptr:ptr+batch_size]
                ptr+=batch_size
                sess.run(self.minimize, {self.data: inp, self.target: out})
                _ = sess.run(self.error,{self.data: inp, self.target: out})
                avg_batch_error += _
                #if _ < best:
                #    best= _
                #self.predict_results.append(sess.run(self._pred, {self.data: inp, self.target: out}))
            self.train_error.append(avg_batch_error/no_of_batches)
        if test_features is not None and test_labels is not None:
            self.test_error = sess.run(self.error, {self.data: inp, self.target: out})
        sess.close()
        return self.predict_results, self.train_error, self.test_error
