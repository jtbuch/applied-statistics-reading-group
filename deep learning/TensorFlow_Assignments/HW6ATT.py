#!/usr/bin/python27
### CSCI2470 - Assignment 6 - Babak Hemmatian - bhemmati - BannerID:B01190949
# import the required modules
import tensorflow as tf
import numpy as np
import sys
import io
import time
from math import floor
### set hyperparameters
MaxFLen = 13 # maximum length of sentences in French
MaxELen = 13 # maximum length of sentences in English
bSz = 20 # number of parallel batches
embedSz = 30 # embedding size
rnnSz = 64 # number of units in the recurrent layer
keepPrb = 0.5 # dropout rate
print "Embedding size = " + str(embedSz)
print "Recurrent layer size = " + str(rnnSz)
print "Dropout rate = " + str(1-keepPrb)
# timer
print "Start time: " + time.strftime('%l:%M%p')
### read the training data
# initialize the vocabulary dictionaries
Vf = {"STOP":0} # French
Ve = {"STOP":0} # English
## read the data files and separate them into sentences
train_F_file = io.open(sys.argv[1],"r",encoding='latin-1')
train_F_sent = []
for line in train_F_file:
    train_F_sent.append(line)
train_F_file.close()
# create a vocabulary based on French training data and rewrite the set as a sequence of indices
indexed_F_train = np.zeros((len(train_F_sent),MaxFLen))
for counter,sentence in enumerate(train_F_sent):
    for ind,word in enumerate(sentence.split()):
        if word in Vf.keys():
            indexed_F_train[counter][ind] = Vf[word]
        else:
            Vf[word] = len(Vf)
            indexed_F_train[counter][ind] = Vf[word]
vfSz = len(Vf) # size of the French vocabulary
print "Finished reading French training data at "+time.strftime('%l:%M%p')
## read the English training file
train_E_file = io.open(sys.argv[2],"r",encoding='latin-1')
train_E_sent = []
for line in train_E_file:
    train_E_sent.append(line)
train_E_file.close()
# create a vocabulary based on English training data and rewrite the set as a sequence of indices
indexed_E_train = np.zeros((len(train_E_sent),MaxELen))
weights_E_train = np.zeros((len(train_E_sent),MaxELen))
weights_E_train[:,0] = 1
for counter,sentence in enumerate(train_E_sent):
    for ind,word in enumerate(sentence.split()):
        if word in Ve.keys():
            indexed_E_train[counter,ind+1] = Ve[word]
        else:
            Ve[word] = len(Ve)
            indexed_E_train[counter,ind+1] = Ve[word]
        weights_E_train[counter,ind+1] = 1
print "Finished reading English training data at "+time.strftime('%l:%M%p')
# matrix for answers
goal = np.zeros((indexed_E_train.shape[0],MaxELen))
goal[:,:MaxELen-1] = indexed_E_train[:,1:]
veSz = len(Ve) # size of the English vocabulary
### set up the computation graph
## create placeholders for input, output and loss weights
encIn = tf.placeholder(tf.int32, shape=[None,None])
decIn = tf.placeholder(tf.int32, shape=[None,None])
ans = tf.placeholder(tf.int32, shape=[None,None])
weights = tf.placeholder(tf.float32,shape=[None,None])
## set up the encoding RNN
with tf.variable_scope("enc"):
    F = tf.Variable(tf.random_normal((vfSz,embedSz),stddev=.1))
    embs = tf.nn.embedding_lookup(F, encIn)
    embs = tf.nn.dropout(embs, keepPrb)
    cell = tf.contrib.rnn.GRUCell(rnnSz)
    initState = cell.zero_state(bSz, tf.float32)
    encOut, encState = tf.nn.dynamic_rnn(cell, embs,
                                           initial_state=initState)
## add pseudoattention
attw = tf.Variable(tf.random_normal((MaxFLen,MaxELen),stddev=.1))
TencOut = tf.transpose(encOut,[0,2,1]) # transpose of encoder output
decIT = tf.tensordot(TencOut,attw,[[2],[0]]) # transpose of weighted decoder input state
decI = tf.transpose(decIT,[0,2,1]) # weighted decoder input state
## set up the decoding RNN
with tf.variable_scope("dec"):
    E = tf.Variable(tf.random_normal((veSz,embedSz),stddev=.1))
    embs = tf.nn.embedding_lookup(E, decIn)
    embs = tf.concat([embs,decI],2)
    embs = tf.nn.dropout(embs, keepPrb)
    cell = tf.contrib.rnn.GRUCell(rnnSz)
    decOut,_ = tf.nn.dynamic_rnn(cell, embs, initial_state=encState)
## Turn the predictions into logits and calculate the loss
W = tf.Variable(tf.random_normal([rnnSz,veSz],stddev=.1))
b = tf.Variable(tf.random_normal([veSz],stddev=.1))
logits = tf.tensordot(decOut,W,axes=[[2],[0]])+b
loss = tf.contrib.seq2seq.sequence_loss(logits, ans,weights)
predictions = tf.argmax(logits,2)
## training with AdamOptimizer
train = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
### training the network
##create the session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
state = sess.run(initState)
## train
TotalWords = 0
TotalCorr = 0
for i in range(0,len(train_F_sent),bSz):
    Encinputs = np.zeros((bSz,MaxFLen))
    Decinputs = np.zeros((bSz,MaxELen))
    answers = np.zeros((bSz,MaxELen))
    weight = np.zeros((bSz,MaxELen))
    for j in range(bSz):
        if j+i == len(train_F_sent):
            break
        Encinputs[j,:] = indexed_F_train[i+j,:]
        Decinputs[j,:] = indexed_E_train[i+j,:]
        answers[j,:] = goal[i+j,:]
        weight[j,:] = weights_E_train[i+j,:]
    _ = sess.run(train,feed_dict={encIn:Encinputs,decIn:Decinputs,ans:answers,weights:weight}) # add predictions as an output and uncomment the next part to get performance on the training set
#     for j in range(bSz):
#         for i in range(MaxELen):
#             if answers[j][i]==0:
#                 break
#             TotalWords+=1
#             if Preds[j][i]==answers[j][i]:
#                 TotalCorr+=1
# TrainPerf = float(TotalCorr)/float(TotalWords)
# print "Percentage correctly predicted on the training set: " + str(TrainPerf)
print "Training finished at "+time.strftime('%l:%M%p')
### Read the test set data
## Read the French test set file
test_F_file = io.open(sys.argv[3],"r",encoding='latin-1')
test_F_sent = []
for line in test_F_file:
    test_F_sent.append(line)
test_F_file.close()
# read the French test set and turn it into indices
indexed_F_test = np.zeros((len(test_F_sent),MaxFLen))
for counter,sentence in enumerate(test_F_sent):
    for ind,word in enumerate(sentence.split()):
        if word in Vf.keys():
            indexed_F_test[counter][ind] = Vf[word]
        else:
            indexed_F_test[counter][ind] = Vf["*UNK*"]
## Read the English test set file
test_E_file = io.open(sys.argv[4],"r",encoding='latin-1')
test_E_sent = []
for line in test_E_file:
    test_E_sent.append(line)
test_E_file.close()
# read the English test set and turn it into indices
indexed_E_test = np.zeros((len(test_E_sent),MaxELen))
for counter,sentence in enumerate(test_E_sent):
    for ind,word in enumerate(sentence.split()):
        if word in Ve.keys():
            indexed_E_test[counter,ind+1] = Ve[word]
        else:
            indexed_E_test[counter,ind+1] = Ve["*UNK*"]
test_E_file.close()
# matrix for answers
goal_test = np.zeros((len(test_E_sent),MaxELen))
goal_test[:,:MaxELen-1] = indexed_E_test[:,1:]
### Test the model
TotalWordsTest = 0
TotalCorrTest = 0
for i in range(0,len(test_F_sent),bSz):
    Encinputs = np.zeros((bSz,MaxFLen))
    Decinputs = np.zeros((bSz,MaxELen))
    answers = np.zeros((bSz,MaxELen))
    for j in range(bSz):
        if i+j == len(test_F_sent):
            break
        Encinputs[j,:] = indexed_F_test[i+j,:]
        Decinputs[j,:] = indexed_E_test[i+j,:]
        answers[j,:] = goal_test[i+j,:]
    Predicts = sess.run(predictions,feed_dict={encIn:Encinputs,decIn:Decinputs,ans:answers})
    for j in range(bSz):
        for i in range(MaxELen):
            if answers[j][i]==0:
                break
            TotalWordsTest+=1
            if Predicts[j][i]==answers[j][i]:
                TotalCorrTest+=1
TestPerf = float(TotalCorrTest)/float(TotalWordsTest)
print "Percentage correctly predicted on the test set: " + str(TestPerf)
# timer
print "Finished at " + time.strftime('%l:%M%p')
