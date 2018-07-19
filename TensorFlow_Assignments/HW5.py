#!/usr/bin/python27
### CSCI2470 - Assignment 5 - Babak Hemmatian - bhemmati - BannerID:B01190949
# import the required modules
import tensorflow as tf
import numpy as np
from sys import argv # so that python can use command line arguments as file addresses
import time # to keep track of performance
from math import floor
### set hyperparameters
batchSz = 50 # number of parallel batches: break the corpus into this many pieces
# and learn the weights on them in parallel. Affects how often the weights get
# updated
windowSz = 20 # window size: How many steps back in time we're going.
# tensorflow only does non-overlapping windows unfortunately (see notebook)
embedSz = 128 # embedding size: How many weights characterize a word's meaning
hiddenSz = 512 # number of units in the recurrent layer that capture the gist
# of the sequence
print "Embedding size = " + str(embedSz)
print "Recurrent layer size = " + str(hiddenSz)
# timer
print "start time: " + time.strftime('%l:%M%p')
# remember: The goal is to predict what the next word in each sequence is
### read the training data
# initialize the vocabulary dict
V = dict()
## read the training file
train_file = open(argv[1],"r")
# create a vocabulary of integers based on training data and rewrite the set as
# a sequence of indices
indexed_train = []
for line in train_file:
    for word in line.split():
        if word in V.keys():
            indexed_train.append(V[word])
        else:
            V[word] = len(V)
            indexed_train.append(V[word])
train_file.close()
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
### set up the computation graph
## create placeholders for input, output
# shapes have to be None, since we're not sure how long the longest sequence
# in the data is. Shorter sequences get padded with *STOP* special characters
# the integer for *STOP* is zero for convenience
inpt = tf.placeholder(tf.int32, shape=[None,None])
answr = tf.placeholder(tf.int32, shape=[None,None])
## set up the variables
# initial embeddings: Each word is represented by an embedSz # of rand floats
E = tf.Variable(tf.random_normal([len(V), embedSz], stddev = 0.1))
# look up the embeddings: Connects these float representations with the integer
# representations we created of the data earlier
embed = tf.nn.embedding_lookup(E, inpt)
## define the recurrent layer (Gated Recurrent Unit)
rnn= tf.contrib.rnn.GRUCell(hiddenSz)
initialState = rnn.zero_state(batchSz, tf.float32)
output, nextState = tf.nn.dynamic_rnn(rnn, embed,initial_state=initialState)
# output is a combination of the hidden state and input. nextState is the
# updated version of the hidden state that is fed into the model for the next
# batch
## create weights and biases for a feedforward layer: The feedforward layer
# turns the weights of the hidden recurrent layer into predictions about
# the specific word that comes next
W = tf.Variable(tf.random_normal([hiddenSz,len(V)], stddev=0.1))
b = tf.Variable(tf.random_normal([len(V)], stddev=0.1))
## calculate loss
# calculate logits. We're picking out the weights at the last step of the look-
# back and multiplying them by "output", which is the transformed input. Not
# really different from the feedforward case
logits = tf.tensordot(output,W,[[2],[0]])+b
# calculate cross-entropy loss for the entire sequence based on how much
# probability the softmax model assigned to each next word for each part of the
# sequence
xEnt = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=answr,weights=tf.ones([batchSz,windowSz],tf.float32))
loss = tf.reduce_mean(xEnt) # total loss is the mean over all batches.
## training with AdamOptimizer (an improved version of gradient descent that
# , if I'm not mistaken, uses higher order moments as well, but is fast)
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
### training the network
##create the session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
Loss = 0
state = sess.run(initialState) # initializing the initial hidden state of zeros
## train
# number of windows and length of each batch for the training set
# break the corpus into batches
batchLen = floor(float((len(indexed_train)+1))/float(batchSz))
batchLen = int(batchLen)
# break each batch into non-overlapping windows
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
        # answers are the inputs shifted by one: the next words
    _,outputs,next,Losses = sess.run([train,output,nextState,loss],feed_dict={inpt:inputs,answr:answers})
    state = next
    Loss+=Losses
## calculate training set perplexity (average per-word loss)
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
new_input = np.reshape(new_input,(batchSz,batchLen)) # creating one loooong
# vector of inputs now that our parallel learning is done
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
