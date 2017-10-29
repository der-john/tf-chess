import tensorflow as tf

x1_data = [
	0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, \
	0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, \
	0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625
	]
x2_data = [
	-0.4, -0.4, -0.4, -0.2, -0.2, -0.2, 0.1, 0.1, 0.1, \
	-0.4, -0.4, -0.4, -0.2, -0.2, -0.2, 0.1, 0.1, 0.1, \
	-0.4, -0.4, -0.4, -0.2, -0.2, -0.2, 0.1, 0.1, 0.1
	]
x3_data = [
	-0.4, -0.2, 0.1, -0.4, -0.2, 0.1, -0.4, -0.2, 0.1, \
	-0.4, -0.2, 0.1, -0.4, -0.2, 0.1, -0.4, -0.2, 0.1, \
	-0.4, -0.2, 0.1, -0.4, -0.2, 0.1, -0.4, -0.2, 0.1
	]
y_data = [
	0.06, 0.11, 0.16, 0.04, 0.10, 0.16, 0.01, 0.09, 0.16, \
	0.07, 0.15, 0.20, 0.05, 0.14, 0.20, 0.02, 0.13, 0.20, \
	0.12, 0.18, 0.21, 0.08, 0.17, 0.21, 0.03, 0.19, 0.21
	]

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W1 * x1_data + W2 * x2_data  + W3 * x3_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2000):
    sess.run(train)
    if step % 250 == 0:
        curr_W1, curr_W2, curr_W3, curr_b, curr_loss = sess.run([W1, W2, W3, b, cost])
        print("W1: %s W2: %s W3: %s b: %s loss: %s"%(curr_W1, curr_W2, curr_W3, curr_b, curr_loss))