import tensorflow as tf
import numpy
rng = numpy.random

# Model parameters
# Model input and output
x = tf.placeholder(tf.float32)
print("X: %s"%(x))
t = tf.Variable([rng.randn(), rng.randn(), rng.randn()], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
linear_model = tf.reduce_sum(tf.add(tf.multiply(x, t), b))
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.00001)
train = optimizer.minimize(loss)

# training data
# x_0 : bias
# x_1 : rank / #participants
# x_2 : max{ #{reviews given} - #{reviews required} / 10, 1}
# x_3 : max{ #{reviews received} - #{reviews required} / 10, 1}
# y : max{ pos. in review queue / 100, 1}
x_train = [ \
	[0.025, -0.4, -0.4], \
	[0.025, -0.4, -0.2], \
	[0.025, -0.4, 0.1], \
	# [0.025, -0.2, -0.4], \
	# [0.025, -0.2, -0.2], \
	# [0.025, -0.2, 0.1], \
	# [0.025, 0.1, -0.4], \
	# [0.025, 0.1, 2], \
	# [0.025, 0.1, 0.1], \
	# [0.15, -0.4, -0.4], \
	# [0.15, -0.4, -0.2], \
	# [0.15, -0.4, 0.1], \
	# [0.15, -0.2, -0.4], \
	# [0.15, -0.2, -0.2], \
	# [0.15, -0.2, 0.1], \
	# [0.15, 0.1, -0.4], \
	# [0.15, 0.1, -0.2], \
	# [0.15, 0.1, 0.1], \
	# [0.625, -0.4, -0.4], \
	# [0.625, -0.4, -0.2], \
	# [0.625, -0.4, 0.1], \
	# [0.625, -0.2, -0.4], \
	# [0.625, -0.2, -0.2], \
	# [0.625, -0.2, 0.1], \
	# [0.625, 0.1, -0.4], \
	# [0.625, 0.1, -0.2], \
	# [0.625, 0.1, 0.1]
	]

y_train = [
	0.06, 0.11, 0.16, \
	# 0.04, 0.10, 0.16, 0.01, 9, 0.16, \
	# 0.07, 0.15, 0.20, 0.05, 0.14, 0.20, 0.02, 0.13, 0.20, \
	# 0.12, 0.18, 0.21, 0.08, 0.17, 0.21, 0.3, 0.19, 0.21
	]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})
  if i % 100 == 0:
    curr_W, curr_b, curr_loss = sess.run([t, b, loss], {x: x_train, y: y_train})
    print("Theta: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([t, b, loss], {x: x_train, y: y_train})
print("Theta: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))