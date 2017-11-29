import tensorflow as tf

def get_nn_for_training():
    n_inputs = 64*12
    n_hidden1 = 4096
    n_hidden2 = 2048
    n_hidden3 = 1024
    n_outputs = 1

    X = tf.placeholder(tf.float32, shape=(3, None, n_inputs), name="X")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.relu)
        out = tf.layers.dense(hidden3, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        kappa = 10
        fXc = out[0]
        fXr = out[1]
        fXp = out[2]
        loss = tf.reduce_mean(-tf.log(tf.sigmoid(fXc - fXr)) - kappa * tf.log(tf.sigmoid(fXc + fXp)) - kappa * tf.log(tf.sigmoid(-fXc - fXp)))

    learning_rate = 0.03
    momentum = 0.9

    with tf.name_scope("train"):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        training_op = optimizer.minimize(loss)

    return training_op, loss, X

def neuron_layer(X, W, b, name, activation=None):
    with tf.name_scope(name):
        X = tf.cast(X, dtype=tf.float64)
        W = tf.cast(W, dtype=tf.float64)
        b = tf.cast(b, dtype=tf.float64)
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

def use_nn_for_play(sess, weights, biases, X_arg):
    W1, W2, W3, W4 = weights
    b1, b2, b3, b4 = biases
    X = tf.placeholder(tf.float32, shape=(None, 768), name="X")

    Z1 = neuron_layer(X, W1, b1, name="hidden1", activation=tf.nn.relu)
    Z2 = neuron_layer(Z1, W2, b2, name="hidden2", activation=tf.nn.relu)
    Z3 = neuron_layer(Z2, W3, b3, name="hidden3", activation=tf.nn.relu)
    out = neuron_layer(Z3, W4, b4, name="outputs")
    return sess.run(out, feed_dict={X: X_arg})