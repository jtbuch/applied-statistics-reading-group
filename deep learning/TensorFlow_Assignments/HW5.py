#!/usr/bin/python27
### CSCI2470 - Assignment 5 - Babak Hemmatian - bhemmati - BannerID:B01190949
# import the required modules
import tensorflow as tf
import numpy as np
from sys import argv
import time
from math import floor
### set hyperparameters
batchSz = 50 # number of parallel batches
windowSz = 20 # window size
embedSz = 128 # embedding size
hiddenSz = 512 # number of units in the recurrent layer
print "Embedding size = " + str(embedSz)
print "Recurrent layer size = " + str(hiddenSz)
# timer
print "start time: " + time.strftime('%l:%M%p')
### read the training data
# initialize the vocabulary dict
V = dict()
## read the training file
train_file = open(argv[1],"r")
# create a vocabulary based on training data and rewrite the set as a sequence of indices
indexed_train = []
for line in train_file:
    for word in line.split():
        if word in V.keys():
            indexed_train.append(V[word])
        else:
            V[word] = len(V)
            indexed_train.append(V[word])
train_file.close()
print "Finished indexing training data at " + time.strftime('%l:%M%p')
## Read the development set file
dev_file = open(argv[2],"r")
# read the dev set and turn it into indices
indexed_dev = []
for line in dev_file:
    for word in line.split():
        if word in V.keys():
            indexed_dev.append(V[word])
        else:
            indexed_dev.append(0)
dev_file.close()
print "Finished indexing development data at " + time.strftime('%l:%M%p')
### set up the computation graph
## create placeholders for input, output
inpt = tf.placeholder(tf.int32, shape=[None,None])
answr = tf.placeholder(tf.int32, shape=[None,None])
## set up the variables
# initial embeddings
E = tf.Variable(tf.random_normal([len(V), embedSz], stddev = 0.1))
# look up the embeddings
embed = tf.nn.embedding_lookup(E, inpt)
## define the recurrent layer (Gated Recurrent Unit)
rnn= tf.contrib.rnn.GRUCell(hiddenSz)
initialState = rnn.zero_state(batchSz, tf.float32)
output, nextState = tf.nn.dynamic_rnn(rnn, embed,initial_state=initialState)
## create weights and biases for a feedforward layer
W = tf.Variable(tf.random_normal([hiddenSz,len(V)], stddev=0.1))
b = tf.Variable(tf.random_normal([len(V)], stddev=0.1))
## calculate loss
# calculate logits
logits = tf.tensordot(output,W,[[2],[0]])+b
# calculate cross-entropy loss
xEnt = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=answr,weights=tf.ones([batchSz,windowSz],tf.float32))
loss = tf.reduce_mean(xEnt)
## training with AdamOptimizer
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
### training the network
##create the session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
Loss = 0
state = sess.run(initialState)
## train
# number of windows and length of each batch for the training set
batchLen = floor(float((len(indexed_train)+1))/float(batchSz))
batchLen = int(batchLen)
number_of_windows = floor(float(batchLen)/float(windowSz))
number_of_windows = int(number_of_windows)
# feed the input in and update the weights
new_input = np.asarray(indexed_train[0:batchSz*batchLen])
new_input = np.reshape(new_input,(batchSz,batchLen))
for i in range(number_of_windows):
    inputs = np.zeros([batchSz,windowSz])
    answers = np.zeros([batchSz,windowSz])
    for j in range(batchSz):
        inputs[j,:] = new_input[j,i*windowSz:(i+1)*windowSz]
        answers[j,:] = new_input[j,i*windowSz+1:(i+1)*windowSz+1]
    _,outputs,next,Losses = sess.run([train,output,nextState,loss],feed_dict={inpt:inputs,answr:answers})
    state = next
    Loss+=Losses
## calculate training set perplexity
train_perplexity = np.exp(Loss/number_of_windows)
print "Perplexity on the training set:" + str(train_perplexity)
### test the network
# number of windows and batch length for the development set
batchLen = floor(float((len(indexed_dev)+1))/float(batchSz))
batchLen = int(batchLen)
number_of_windows = floor(float(batchLen)/float(windowSz))
number_of_windows = int(number_of_windows)
# feed the input in and calculate loss
new_input = np.asarray(indexed_dev[0:batchSz*batchLen])
new_input = np.reshape(new_input,(batchSz,batchLen))
Devloss = 0
for i in range(number_of_windows):
    inputs = np.zeros([batchSz,windowSz])
    answers = np.zeros([batchSz,windowSz])
    for j in range(batchSz):
        inputs[j,:] = new_input[j,i*windowSz:(i+1)*windowSz]
        answers[j,:] = new_input[j,i*windowSz+1:(i+1)*windowSz+1]
    DevLoss = sess.run(loss,feed_dict={inpt:inputs,answr:answers})
    Devloss+=DevLoss
## calculate development set perplexity
dev_perplexity = np.exp(Devloss/number_of_windows)
print "Perplexity on the development set:" + str(dev_perplexity)
# timer
print "Finishing time:" + time.strftime('%l:%M%p')
