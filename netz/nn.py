import tensorflow as tf

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