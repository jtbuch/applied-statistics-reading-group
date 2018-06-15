#!/usr/bin/python27
# import tensorflow and the mnist dataset
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# set batch size
batchSz=100
# create placeholder variables for input
img=tf.placeholder(tf.float32, [batchSz,784])
ans = tf.placeholder(tf.float32, [batchSz, 10])
# define the first layer
U = tf.Variable(tf.random_normal([784,784], stddev=.1))
bU = tf.Variable(tf.random_normal([784], stddev=.1))
l1Output = tf.matmul(img,U)+bU
l1Output=tf.nn.relu(l1Output)
# define the second layer
V = tf.Variable(tf.random_normal([784,10], stddev=.1))
bV = tf.Variable(tf.random_normal([10], stddev=.1))
# run softmax
prbs=tf.nn.softmax(tf.matmul(l1Output,V)+bV)
# define loss function
xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))
# set learning rate and loss function
train = tf.train.GradientDescentOptimizer(0.5).minimize(xEnt)
# count the number of correct classifications
numCorrect= tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))
# crate new session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#-------------------------------------------------
# feed the data into the model batch by batch 1000 times
for i in range(1000):
    imgs, anss = mnist.train.next_batch(batchSz)
    sess.run(train, feed_dict={img: imgs, ans: anss})
# test accuracy
sumAcc=0
for i in range(1000):
    imgs, anss= mnist.test.next_batch(batchSz)
    sumAcc+=sess.run(accuracy, feed_dict={img: imgs, ans: anss})
print "Test Accuracy: %r" % (sumAcc/1000)
