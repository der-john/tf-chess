# I keep this file here for documenting some earlier work of mine.
# I don't use it anymore, because its calculations are SO much slower than `fast_trainer.py`.

# Common imports
import numpy as np
import os
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import pickle
import random
import itertools
import scipy.sparse
import h5py
import math
import time
from time import gmtime, strftime

def rnd_array(m,n):
    RNG = np.random
    return np.asarray(RNG.uniform(
        low = -np.sqrt(1. / (m+n)),
        high = np.sqrt(1. / (m+n)),
        size = (m, n)), dtype=np.float64)

def dsig(a):
    return (1-a)*a

n_inputs = 64*12  # 'half-deep-pink setup'
n_hidden1 = 1024
n_hidden2 = 1024
n_outputs = 1

n_epochs = 1
batch_size = 50
learning_rate = 0.01


def load_data(dir='game-files/'):
    for fn in os.listdir(dir):
        if not fn.endswith('.hdf5'):
            continue

        fn = os.path.join(dir, fn)
        try:
            yield h5py.File(fn, 'r')
        except:
            print('could not read', fn)

def convert_boards(series):
    out = []
    for subseries in series:

        vectors = []
        for board in subseries:
            # Convert input into a 12 * 64 list
            board.flatten()
            X = []
            for sq in board:
                for piece in [1,2,3,4,5,6, 8,9,10,11,12,13]:
                    X.append(np.float64(sq == piece))
            vectors.append(X)
        out.append(vectors)
    return out

def get_data(series=['x', 'xr']):
    data = [[] for s in series]
    for f in load_data():
        try:
            for i, s in enumerate(series):
                data[i].append(f[s].value)
        except:
            raise
            print('failed reading from', f)

    def stack(vectors):
        if len(vectors[0].shape) > 1:
            return np.vstack(vectors)
        else:
            return np.hstack(vectors)

    data = [stack(d) for d in data]

    data = convert_boards(data)

    test_size = int(len(data[0]) / 5)
    print('Splitting', len(data[0]), 'entries into train/test set')
    data = train_test_split(*data, test_size=test_size)

    print(len(data[0]), 'train set', len(data[1]), 'test set')

    return data

X = tf.placeholder(tf.float64, shape=(None, n_inputs), name="X")
fXc = tf.placeholder(tf.float64, shape=(None), name="a3_c")
fXr = tf.placeholder(tf.float64, shape=(None), name="a3_r")
fXp = tf.placeholder(tf.float64, shape=(None), name="a3_p")
theta1 = tf.placeholder(tf.float64, shape=(None, None), name="theta_1")
theta2 = tf.placeholder(tf.float64, shape=(None, None), name="theta_2")
theta3 = tf.placeholder(tf.float64, shape=(None), name="theta_3")
a1 = tf.placeholder(tf.float64, shape=(None), name="a_1")
a2 = tf.placeholder(tf.float64, shape=(None), name="a_2")
a3 = tf.placeholder(tf.float64, shape=(None), name="a_3")

with tf.name_scope("loss"):
    kappa = 10
    loss = tf.reduce_mean(-tf.log(tf.sigmoid(fXc - fXr)) - kappa * tf.log(tf.sigmoid(fXc + fXp)) - kappa * tf.log(tf.sigmoid(-fXc - fXp)))

# Cf. https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm
with tf.name_scope("backprop"):
    # delta 4 has dimensions ()
    delta4 = loss
    delta3 = theta3 * delta4 * dsig(a3)
    delta2 = tf.matmul(theta2, delta3) * dsig(a2)
    delta1 = tf.matmul(theta1, delta2) * dsig(a1)

def dnn_layer(input_tensor, n_input_units, n_output_units, name, weights, biases, activation_fn=None):
    with tf.variable_scope(name):
        input_tensor = tf.cast(input_tensor, dtype=tf.float64)
        weights = tf.cast(weights, dtype=tf.float64)
        biases = tf.cast(biases, dtype=tf.float64)
        act = tf.matmul(input_tensor, weights) + biases
        if activation_fn is not None:
            act = activation_fn(act)
    return act

def calc_and_update(sess, Xc, Xr, Xp, layers, learning_rate):
    h1, h2, o = layers
    h1_w, h1_b = h1
    h2_w, h2_b = h2
    o_w, o_b = o

    with tf.name_scope("dnn"):
        hidden1 = dnn_layer(input_tensor=X, 
                       n_input_units=n_inputs, 
                       n_output_units=n_hidden1, 
                       name='hidden1',
                       weights=h1_w, 
                       biases=h1_b,
                       activation_fn=tf.nn.relu)
        hidden2 = dnn_layer(input_tensor=hidden1, 
                       n_input_units=n_hidden1, 
                       n_output_units=n_hidden2, 
                       name='hidden2',
                       weights=h2_w, 
                       biases=h2_b,
                       activation_fn=tf.nn.relu)
        out = dnn_layer(input_tensor=hidden2, 
                       n_input_units=n_hidden2, 
                       n_output_units=n_outputs, 
                       name='outputs',
                       weights=o_w, 
                       biases=o_b)

    for c,r,p in np.reshape([Xc,Xr,Xp], (batch_size, 3, n_inputs)):
        # forward propagation
        # a1_c and a2_c each have the dimensions (1, 1024)
        a1_c, a2_c, a3_c = sess.run([hidden1, hidden2, out], feed_dict={X: [c]})
        _, _, a3_r = sess.run([hidden1, hidden2, out], feed_dict={X: [r]})
        _, _, a3_p = sess.run([hidden1, hidden2, out], feed_dict={X: [p]})

        # backward propagation

        # del4 has the dimensions ()
        del4 = sess.run(delta4, feed_dict={fXc: a3_c, fXr: a3_r, fXp: a3_p})
        # dsig(a1) has the dimensions (1, 1)
        # del3_w has the dimensions (1024, 1), del3_b (1, 1)
        del3_w = sess.run(delta3, feed_dict={theta3: o_w, delta4: del4, a3: a3_c})
        del3_b = sess.run(delta3, feed_dict={theta3: o_b, delta4: del4, a3: a3_c})
        # dsig(a2) has the dimensions (1, 1024)
        # del2_w has the dimensions (1024, 1024), del2_b (1024, 1024)
        del2_w = sess.run(delta2, feed_dict={theta2: h2_w, delta3: del3_w, a2: a2_c})
        del2_b = sess.run(delta2, feed_dict={theta2: h2_b, delta3: del3_w, a2: a2_c})
        # dsig(a1) has the dimensions (1, 1024)
        # del1_w has the dimensions (768, 1024), del1_b (1, 1024)
        del1_w = sess.run(delta1, feed_dict={theta1: h1_w, delta2: del2_w, a1: a1_c})
        del1_b = sess.run(delta1, feed_dict={theta1: h1_b, delta2: del2_w, a1: a1_c})

    # upgrade
    h1_w -= [learning_rate * v for v in del1_w]
    h1_b -= [learning_rate * v for v in del1_b]
    h2_w -= [learning_rate * v for v in del2_w]
    h2_b -= [learning_rate * v for v in del2_b]
    o_w -= [learning_rate * v for v in del3_w]
    o_b -= [learning_rate * v for v in del3_b]

    layers = np.array([np.array([h1_w, h1_b]), np.array([h2_w, h2_b]), np.array([o_w, o_b])])

    return layers, del4


def calc_loss(sess, Xc, Xr, Xp, layers):
    h1, h2, o = layers
    h1_w, h1_b = h1
    h2_w, h2_b = h2
    o_w, o_b = o

    with tf.name_scope("dnn"):
        hidden1 = dnn_layer(input_tensor=X, 
                       n_input_units=n_inputs, 
                       n_output_units=n_hidden1, 
                       name='hidden1',
                       weights=h1_w, 
                       biases=h1_b,
                       activation_fn=tf.nn.relu)
        hidden2 = dnn_layer(input_tensor=hidden1, 
                       n_input_units=n_hidden1, 
                       n_output_units=n_hidden2, 
                       name='hidden2',
                       weights=h2_w, 
                       biases=h2_b,
                       activation_fn=tf.nn.relu)
        out = dnn_layer(input_tensor=hidden2, 
                       n_input_units=n_hidden2, 
                       n_output_units=n_outputs, 
                       name='outputs',
                       weights=o_w, 
                       biases=o_b)

    for c,r,p in np.reshape([Xc,Xr,Xp], (len(Xc), 3, n_inputs)):
        # forward propagation
        # a1_c and a2_c each have the dimensions (1, 1024)
        a1_c, a2_c, a3_c = sess.run([hidden1, hidden2, out], feed_dict={X: [c]})
        _, _, a3_r = sess.run([hidden1, hidden2, out], feed_dict={X: [r]})
        _, _, a3_p = sess.run([hidden1, hidden2, out], feed_dict={X: [p]})
        del4 = sess.run(delta4, feed_dict={fXc: a3_c, fXr: a3_r, fXp: a3_p})
    return del4

init = tf.global_variables_initializer()

def train():
    Xc_train, Xc_test, Xr_train, Xr_test, Xp_train, Xp_test = get_data(['x', 'xr', 'xp'])

    min_test_loss = float('inf')

    layers = np.array(
        [np.array([rnd_array(n_inputs, n_hidden1), rnd_array(1, n_hidden1)]),
        np.array([rnd_array(n_hidden1, n_hidden2), rnd_array(1, n_hidden2)]),
        np.array([rnd_array(n_hidden2, 1), rnd_array(1, 1)])]
        )

    h1, h2, o = layers
    h1_w, h1_b = h1
    h2_w, h2_b = h2
    o_w, o_b = o

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(Xc_train) // batch_size):
                batch_index = random.randint(0, int(len(Xc_train) / batch_size) - 1)
                lo, hi = batch_index * batch_size, (batch_index + 1) * batch_size
                Xc_batch = Xc_train[lo:hi]
                Xr_batch = Xr_train[lo:hi]
                Xp_batch = Xp_train[lo:hi]
                layers, loss = calc_and_update(sess, Xc_batch, Xr_batch, Xp_batch, layers, learning_rate)
                print("Iteration ", iteration, " Training Loss: ", loss)
                break

            if epoch % 10 == 0:
                batch_index = random.randint(0, int(len(Xc_test) / batch_size) - 1)
                lo, hi = batch_index * batch_size, (batch_index + 1) * batch_size
                Xc_batch_t = Xc_test[lo:hi]
                Xr_batch_t = Xr_test[lo:hi]
                Xp_batch_t = Xp_test[lo:hi]
                loss_t = calc_loss(sess, Xc_batch_t, Xr_batch_t, Xp_batch_t, layers)
                print("Epoch ", epoch, "Test Loss: ", loss_t)
            if loss_t < min_test_loss:
                print("Record Minimum Loss:", loss_t)
                min_test_loss = loss_t

                print("Dumping upgraded model")
                newfilename = 'model.tfc'
                fout = open(newfilename, 'wb')
                pickle.dump(layers, fout)
                fout.close()

if __name__ == '__main__':
    train()