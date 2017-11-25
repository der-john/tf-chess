# Common imports
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import random
import itertools
import scipy.sparse
import h5py
import math
import time


def convert_board(raw_board):
    # Convert input into a 12 * 64 list
    conv_board = []
    flattened_board = raw_board.flatten()
    for sq in flattened_board:
        for piece in [1,2,3,4,5,6, 8,9,10,11,12,13]:
            conv_board.append(np.float64(sq == piece))
    return conv_board

def load_data(dir='game-files/'):
    for fn in os.listdir(dir):
        if not fn.endswith('.hdf5'):
            continue

        fn = os.path.join(dir, fn)
        try:
            yield h5py.File(fn, 'r')
        except:
            print('could not read', fn)

def get_data(series=['x', 'xr', 'xp']):
    print("getting data...")
    data = [[] for s in series]
    for f in load_data():
        print("loading", f)
        try:
            for i, s in enumerate(series):
                for raw_board in f[s].value:
                    converted_board = convert_board(raw_board)
                    data[i].append(converted_board)
        except:
            raise
            print('failed reading from', f)
    print("done getting data")

    test_size = len(data[0]) // 5
    print('Splitting', len(data[0]), 'entries into train/test set')
    data = train_test_split(*data, test_size=test_size)

    print(len(data[0]), 'train set', len(data[1]), 'test set')

    return data

def train(training_op, loss, X):
    Xc_train, Xc_test, Xr_train, Xr_test, Xp_train, Xp_test = get_data()

    n_epochs = 40
    batch_size = 50
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(Xc_train) // batch_size):
                batch_index = random.randint(0, len(Xc_train) // batch_size - 1)
                lo, hi = batch_index * batch_size, (batch_index + 1) * batch_size
                X_batch = np.array([Xc_train[lo:hi], Xp_train[lo:hi], Xr_train[lo:hi]])
                sess.run(training_op, feed_dict={X: X_batch})
                cost = sess.run(loss, feed_dict={X: X_batch})
                print("Iteration: ", iteration, " Training Loss: ", cost)

            X_test = np.array([Xc_test, Xp_test, Xr_test])
            test_cost = sess.run(loss, feed_dict={X: X_test})
            print("Epoch: ", epoch, " Test Loss: ", test_cost)

        print("Dumping the model")
        weights = [v.eval(session=sess) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
        biases = [v.eval(session=sess) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('biases:0')]
        fout = open('model.tfc', 'wb')
        pickle.dump((weights, biases), fout)
        fout.close()

def main():
    # The following NN contains ? units.
    n_inputs = 64*12
    n_hidden1 = 4096
    n_hidden2 = 2048
    n_hidden3 = 1024
    n_outputs = 1

    X = tf.placeholder(tf.float32, shape=(3, None, n_inputs), name="X")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                                  activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                                  activation=tf.nn.relu)
        hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3",
                                  activation=tf.nn.relu)
        out = tf.layers.dense(hidden3, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        kappa = 10
        fXc = out[0]
        fXr = out[1]
        fXp = out[2]
        loss = tf.reduce_mean(-tf.log(tf.sigmoid(fXc - fXr)) - kappa * tf.log(tf.sigmoid(fXc + fXp)) - kappa * tf.log(tf.sigmoid(-fXc - fXp)))

    learning_rate = 0.3
    momentum = 0.9

    with tf.name_scope("train"):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        training_op = optimizer.minimize(loss)

    train(training_op, loss, X)

if __name__ == '__main__':
    main()
