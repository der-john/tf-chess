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


n_inputs = 64*12  # 'deep-pink setup'
n_hidden1 = 1024
n_hidden2 = 1024
n_outputs = 1
params = {
    'h1_ws': [],
    'h1_bs': [],
    'h2_ws': [],
    'h2_bs': [],
    'out_ws': [],
    'out_bs': []
}

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
                    # pieces.append((x_s <= piece and x_s >= piece).astype(theano.config.floatX))
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

def nesterov_update(loss, params, learning_rate, momentum):
    updates = []
    gradients = tf.gradients(loss, params)
    for tensor in params:
        try:
            sess.run(tensor.initializer)
        except:
            # Print no need to initialize: not a variable
            continue
    # Convert momentum into a matrix
    if type(momentum) is not tf.Tensor:
        momentum = momentum * tf.ones(shape=(params[0].shape), dtype=tf.float64)
    # Convert the learning rate into a matrix
    if type(learning_rate) is not tf.Tensor:
        learning_rate = learning_rate * tf.ones(shape=(params[0].shape), dtype=tf.float64)
    # Build the momentums from the gradients and params
    for param_i, gradient_i in zip(params, gradients):
        # Note that zip gives a tuple version of an iterable)
        if gradient_i is None:
            gradient_i = 0
        momentum_param = tf.zeros(shape=(momentum.shape), dtype=tf.float64)
        velocity = momentum * momentum_param - learning_rate * gradient_i
        weight = momentum * velocity - learning_rate * gradient_i
        if param_i.shape == (n_inputs, n_hidden1):
            weight = weight + param_i
        else:
            weight = tf.pad(weight, [[0, n_hidden1 - n_inputs], [0, 0]]) + param_i
        updates.append((param_i, weight))
        updates.append((momentum_param, velocity))
    return updates

def dnn_layer(input_tensor, n_input_units, n_output_units, name, activation_fn=None, weight_params=None, bias_params=None):
    with tf.variable_scope(name):

        if len(weight_params) > 0:
            weights = tf.Variable(weight_params, name='weights', dtype=tf.float64)
        else:
            weights = tf.Variable(
                tf.truncated_normal(shape=(n_input_units, n_output_units), mean=0.0, stddev=0.1, dtype=tf.float64),
                name='weights')

        if len(bias_params) > 0:
            biases = tf.Variable(bias_params, name='biases', dtype=tf.float64)
        else:
            biases = tf.Variable(tf.zeros(shape=(n_output_units), dtype=tf.float64),
                                 name='biases')

        act = tf.matmul(input_tensor, weights) + biases

        if activation_fn is not None:
            act = activation_fn(act)

    return act

X = tf.placeholder(tf.float64, shape=(None, n_inputs), name="X")
fXc = tf.placeholder(tf.float64, shape=(None), name="f_X_c")
fXr = tf.placeholder(tf.float64, shape=(None), name="f_X_r")
fXp = tf.placeholder(tf.float64, shape=(None), name="f_X_p")

with tf.name_scope("dnn"):
    hidden1 = dnn_layer(input_tensor=X, 
                   n_input_units=n_inputs, 
                   n_output_units=n_hidden1, 
                   name='hidden1',
                   weight_params=params['h1_ws'], 
                   bias_params=params['h1_bs'],
                   activation_fn=tf.nn.relu)
    hidden2 = dnn_layer(input_tensor=hidden1, 
                   n_input_units=n_hidden1, 
                   n_output_units=n_hidden2, 
                   name='hidden2',
                   weight_params=params['h2_ws'], 
                   bias_params=params['h2_bs'],
                   activation_fn=tf.nn.relu)
    out = dnn_layer(input_tensor=hidden2, 
                   n_input_units=n_hidden2, 
                   n_output_units=n_outputs, 
                   name='outputs',
                   weight_params=params['out_ws'], 
                   bias_params=params['out_bs'],
                   activation_fn=tf.nn.relu)
'''
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    out = tf.layers.dense(hidden2, n_outputs, name="outputs")
'''

with tf.name_scope("loss"):
    kappa = 10
    loss = tf.reduce_sum(-tf.log(tf.sigmoid(fXc - fXr)) - kappa * tf.log(tf.sigmoid(fXc + fXp)) - kappa * tf.log(tf.sigmoid(-fXc - fXp)))


'''
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
'''

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 1
batch_size = 50
learning_rate = 0.01

def train():
    Xc_train, Xc_test, Xr_train, Xr_test, Xp_train, Xp_test = get_data(['x', 'xr', 'xp'])

    best_test_loss = float('inf')
    base_learning_rate = 0.03
    t0 = time.time()

    batch_index = random.randint(0, int(len(Xc_train) / batch_size) - 1)
    lo, hi = batch_index * batch_size, (batch_index + 1) * batch_size

    with tf.Session() as sess:
        init.run()
        gr = tf.get_default_graph()
        for epoch in range(n_epochs):
            for iteration in range(len(Xc_train) // batch_size):
                Xc_batch = Xc_train[lo:hi]
                Xr_batch = Xr_train[lo:hi]
                Xp_batch = Xp_train[lo:hi]
                f_X_c = sess.run(out, feed_dict={X: Xc_batch})
                f_X_r = sess.run(out, feed_dict={X: Xr_batch})
                f_X_p = sess.run(out, feed_dict={X: Xp_batch})
                # sess.run(training_op, feed_dict={fXc: f_X_c, fXr: f_X_r, fXp: f_X_p})
                loss_f = sess.run(loss, feed_dict={fXc: f_X_c, fXr: f_X_r, fXp: f_X_p})
                print("Iteration ", iteration, " Loss: ", loss_f)
                # h1_ws has the dimensions (768, 1024), h1_bs has dimensions (1024)
                params['h1_ws'] = gr.get_tensor_by_name('dnn/hidden1/weights:0').eval()
                params['h1_bs'] = gr.get_tensor_by_name('dnn/hidden1/biases:0').eval()
                # h2_ws has the dimensions (1024, 1024), h2_bs has dimensions (1024)
                params['h2_ws'] = gr.get_tensor_by_name('dnn/hidden2/weights:0').eval()
                params['h2_bs']  = gr.get_tensor_by_name('dnn/hidden2/biases:0').eval()
                # out_ws has the dimensions (1024), out_bs has dimensions ()
                params['out_ws'] = gr.get_tensor_by_name('dnn/outputs/weights:0').eval()
                params['out_bs'] = gr.get_tensor_by_name('dnn/outputs/biases:0').eval()

                Ws_s = [params['h1_ws'], params['h2_ws'], params['out_ws']]
                bs_s = [params['h1_bs'], params['h2_bs'], params['out_bs']]
                # Ws_s + bs_s has length 6

                updates = nesterov_update(loss, Ws_s + bs_s, learning_rate, 0.9)
                # updated = sess.run(self.outputs + self.updates, feed_dict=feed_dict)
                print(len(updates))
                input("That was len'updates'")
            # acc_train = sess.run(loss, feed_dict={Xc: Xc_train, Xr: Xr_train, Xp: Xp_train})
            # acc_test = sess.run(loss, feed_dict={Xc: Xc_test, Xr: Xr_test, Xp: Xp_test})
            # print(epoch, "Training cost:", acc_train, "Test cost:", acc_test)

        save_path = saver.save(sess, "./my_model_final.ckpt")

if __name__ == '__main__':
    train()