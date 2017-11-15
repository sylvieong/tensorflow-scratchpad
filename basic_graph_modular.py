# testing in Python 3.5.2, tf.__version__ '1.2.0', np.__version__ '1.12.1'

import tensorflow as tf
import numpy as np

# function that implements matmul, add, relu
# takes in input x and returns the output from relu
def basic_graph_module(x, input_dim, output_dim):
    b = tf.Variable(tf.zeros((output_dim,)))  # dim: Do 
    W = tf.Variable(tf.random_uniform((input_dim, output_dim),-1, 1)) # dim: Dx, Do
    h = tf.nn.relu(tf.matmul(x, W) + b) # dim (should be): N,Do
    return h
    

# module0, input_dim 784, output_dim 200, batch size 100
x = tf.placeholder(tf.float32, (100, 784))  # dim: N, Dx
h0 = basic_graph_module(x, 784, 200)

# module1, input_dim 200, output_dim 120
h1 = basic_graph_module(h0, 200, 120)

# module2, input_dim 120, output_dim 10
h2 = basic_graph_module(h1, 120, 10)


prediction = tf.nn.softmax(h2)
label = tf.placeholder(tf.float32,[100, 10])
cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# checking defined nodes
#print(b)
#print(W)
print(x)
print(h0)
print(h1)
print(prediction)
print(label)
print(cross_entropy)
print(train_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# x: random samples from a uniform distribution over [0, 1)
# label: randomly set one of 10 indexes of each row to 1, the rest is zero
batch_x = np.random.rand(100, 784)
batch_label = np.zeros((100,10))
# for each row randomly get an integer index [0,9) and set label_at_run_time[row, index] to 1.0
for row in range(100):
	col_to_set_to_one = np.random.randint(0,10)
	batch_label[row, col_to_set_to_one] = 1.0
print(batch_label)


#sess.run(h, prediction, cross_entropy, {x: x_at_run_time, label: label_at_run_time})

for i in range(10):
	print(i)
	print(sess.run(h0, feed_dict={x: batch_x, label: batch_label}))
	print(sess.run(h1, feed_dict={x: batch_x, label: batch_label}))
	print(sess.run(h1, feed_dict={x: batch_x, label: batch_label}))
	print(sess.run(prediction, feed_dict={x: batch_x, label: batch_label}))
	sess.run(train_step, feed_dict={x: batch_x, label: batch_label})

