# import the required modules
import tensorflow as tf
import numpy as np
from sys import argv
import time
# timer
print "start time:" + time.strftime('%l:%M%p')
# set hyperparameter
embedSz = 400
# create placeholders for input, output and dropout rate
inpt = tf.placeholder(tf.int32, shape=[None])
inpt2 = tf.placeholder(tf.int32, shape=[None])
answr = tf.placeholder(tf.int32, shape=[None])
keepP = tf.placeholder(tf.float32)
# initialize the vocabulary dict
V = dict()
# read the training set
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
# Open development set
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
# create initial embeddings
E = tf.Variable(tf.random_normal([len(V), embedSz], stddev = 0.1))
# look up the embeddings
embed = tf.nn.embedding_lookup(E, inpt)
embed2 = tf.nn.embedding_lookup(E, inpt2)
both = tf.concat([embed,embed2],1)
# create a feedforward layer
# U = tf.Variable(tf.random_normal([embedSz*2,500], stddev=.1))
# bU = tf.Variable(tf.random_normal([500], stddev=.1))
# l2Output = tf.nn.relu(tf.matmul(both,U)+bU)
# create the layer that produces the logits
W = tf.Variable(tf.random_normal([2*embedSz,len(V)], stddev= 0.1))
b = tf.Variable(tf.random_normal([len(V)], stddev=0.1))
# logits
logits = tf.nn.relu(tf.matmul(both,W)+b)
# l2Output = tf.nn.dropout(logits,keepP)
# calculate loss and perplexity
xEnt = tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=answr)
batch_loss = tf.reduce_mean(xEnt)
# training with AdamOptimizer
train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(batch_loss)
# create the session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# turn the training data into word triples and train the NN with batches of size 20
loss = 0
number_of_batches = 0
for i in range(0, len(indexed_train)-2,20):
    input_1=[]
    input_2=[]
    labels=[]
    if (len(indexed_train)-2)-i>20:
        for j in range(i,i+20):
            input_1.append(indexed_train[j])
            input_2.append(indexed_train[j+1])
            labels.append(indexed_train[j+2])
    else:
        for j in range(i,len(indexed_train)-2):
            input_1.append(indexed_train[j])
            input_2.append(indexed_train[j+1])
            labels.append(indexed_train[j+2])
    input_1 = np.asarray(input_1)
    input_2 = np.asarray(input_2)
    labels = np.asarray(labels)
    _,Loss=sess.run([train,batch_loss],feed_dict={inpt:input_1,inpt2:input_2,answr:labels,keepP:0.9})
    number_of_batches+=1
    loss += Loss
perplexity = np.exp(loss/number_of_batches)
print "Perplexity on the training set:" + str(perplexity)
# turn the development data into word triples
dev_input_1=[]
dev_input_2=[]
dev_labels=[]
for i in range(0, len(indexed_dev)-2):
    dev_input_1.append(indexed_dev[i])
    dev_input_2.append(indexed_dev[i+1])
    dev_labels.append(indexed_dev[i+2])
# dev_input_1 = np.asarray(dev_input_1)
# dev_input_2 = np.asarray(dev_input_2)
# dev_labels = np.asarray(dev_labels)
# test the model on the dev set
Losses=sess.run(batch_loss,feed_dict={inpt:dev_input_1,inpt2:dev_input_2,answr:dev_labels,keepP:1})
dev_perplexity = np.exp(Losses)
print "Perplexity on the development set:" + str(dev_perplexity)
# timer
print "Finishing time:" + time.strftime('%l:%M%p')
