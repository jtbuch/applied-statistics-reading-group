#!/usr/bin/python27
# import tensorflow
import tensorflow as tf
# The real TensorFlow code is in C. But it has various "wrappers", the most complete of which is Python. The Python code
# one writes is transformed into a "graph" in C (called "session" or "sess" for short) and the process of writing code 
# in TensorFlow is composed of the two steps of defining that graph and then asking for specific parts of it to be run.
# While the implicit C code is incredibly fast, one can take advantage of the parallelizing power of GPUs that this language
# is designed to work with to increase the speed even further. Let me know if you need helping compiling TensorFlow from
# source to use GPU. 

# import the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# One-hot transforms the integer labels (e.g. 1) into one-hot vectors that are more convenient to train with in TensorFlow
# (e.g. [1,0,0,0,0,0,0,0,0,0])

# we now define the different elements of our graph, the network itself

# set batch size: determines the number of training examples that will be used as input for one update of the parameters
batchSz=100
# create placeholder variables for input
# Placeholders are different than parameters. They are empty data containers of the right data type that are used for
# passing in input or passing out results. It's always a good idea to explicitly set the type to float32. The second argument
# determines the size of the container. Since we have 784 pixels per image and 100 images in a batch, our input is 100x784.
img=tf.placeholder(tf.float32, [batchSz,784])
ans = tf.placeholder(tf.float32, [batchSz, 10])
# define the first layer
# these are the weights. Variables are by default trainable. Set trainable=False if they should not be included in backpropagation
# Here I'm just using random normal distribution with low standard deviation to kickstart the weights
U = tf.Variable(tf.random_normal([784,784], stddev=.1),trainable=True) 
bU = tf.Variable(tf.random_normal([784], stddev=.1),trainable=True) # these are the biases
l1Output = tf.matmul(img,U)+bU # matmul just multiplies the two matrices
l1Output=tf.nn.relu(l1Output) # relu sets negative activations to zero. It has been shown that this greatly helps ff net perf.
# define the second layer: Ten nodes because of the ten digits. This is the output layer.
V = tf.Variable(tf.random_normal([784,10], stddev=.1),trainable=True) # using matrix multiplication, boil down the 784 activations into 10
bV = tf.Variable(tf.random_normal([10], stddev=.1),trainable=True) # add biases
# run softmax
prbs=tf.nn.softmax(tf.matmul(l1Output,V)+bV) # so I'm calculating the product of the output of layer one and layer two weights,
# adding layer two biases, and then applying softmax to the activations of the output to determine which option to choose.
# softmax gives the closest approximation to the underlying probability distribution that created activations (information theory)
# define loss function
# here we're using cross entropy loss. It essentially adds up how much uncertainty there is in our distribution of outputs
# and how far it is from the ideal (true) distribution. To calculate it (there are helper functions that do that, by the way)
# I am first picking out the probability of the right answer in our output by multiplying the one-hot answer vector by the
# log of the output probability. Logging is done to prevent underflow problems with small probabilities (them turning into
# zero). reduce_sum computes the sum across the dimension of interest. Here, dimension 1, rows, i.e. turn the one-hot vector
# into a single number. We want to see how much uncertainty overall we had in our batch. So reduce_mean takes the mean of the
# one hundred log-probability values
# this function has changed a bit since I used it.
xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))
# set learning rate and loss function
train = tf.train.GradientDescentOptimizer(0.05).minimize(xEnt) # determine which optimizer to use, learning rate, and the
# function to minimize
# count the number of correct classifications
numCorrect= tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1)) # this checks to see if the max argument in the one-hot answer vector
# matches the max argument in the output probabilities. If so, the model has succeeded
accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32)) # cast transforms a data type into another. numCorrect is integer,
# so if we don't transform it into floats, the mean calculated will also be a smaller than one integer, that is 0.

# crate new session and initialize variables
sess = tf.Session() # creates the graph associated with the elements above. Note the capital S
sess.run(tf.global_variables_initializer()) # initializes the variables (assigns random normal initial values as defined
#-------------------------------------------------
# feed the data into the model batch by batch 1000 times
for i in range(1000):
    imgs, anss = mnist.train.next_batch(batchSz) # asks for the input and desired output to be stored in imgs and anss 
    # for the next batch
    sess.run(train, feed_dict={img: imgs, ans: anss}) # this calls for running the "train" function defined above.
    # TensorFlow will automatically calculate variables that are needed and ignore the rest. But the input-output placeholders 
    # defined above need to be filled with the value in imgs and anss for the training to happen. feed_dict associates these
    # with one-another and feeds them into the network. Always be careful about the data-types and shapes of these
# test accuracy
sumAcc=0
for i in range(1000):
    imgs, anss= mnist.test.next_batch(batchSz)
    sumAcc+=sess.run(accuracy, feed_dict={img: imgs, ans: anss}) # similar syntax to training above. However, since "accuracy"
    # does not require updating variables to calculate, no training happens in this part. Which means we get to test our
    # performance on not-previously-seen samples. Note that sumAcc needs to be defined as the output. That one is the Python
    # object we can print. Accuracy itself is a TensorFlow object that cannot be directly printed out.
print "Test Accuracy: %r" % (sumAcc/100) 
# there is a bug somewhere, possibly in the non-Python code this transforms into, causing us to lose one decimal point in 
# the calculation in some trials. Neither I, nor any of the TAs could figure it out: the code seems fine as far as we could tell.
