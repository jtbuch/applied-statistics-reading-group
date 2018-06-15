# python27
### HW1 - CSCI2470 - Babak Hemmatian - Banner ID: B01190949
# import the required modules
import numpy as np
import gzip
import sys
# load the training dataset
with open(sys.argv[1], 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    bytestream.seek(16)
    images = np.frombuffer(bytestream.read(),np.uint8)
    # define training set as the first 10000 examples
    train_images = [np.divide(float(x),255) for x in images[0:10000*784]]
    # add bias as a feature with value 1 to all training examples
    train_images = np.c_[np.ones(10000),np.reshape(train_images,(10000,784))]
# load the training labels
with open(sys.argv[2], 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    bytestream.seek(8)
    train_labels = np.frombuffer(bytestream.read(10000),np.uint8)
# load the test dataset
with open(sys.argv[3], 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    bytestream.seek(16)
    test_images = np.frombuffer(bytestream.read(),np.uint8)
    test_images = [np.divide(float(x),255) for x in test_images]
    test_images = np.reshape(test_images,(10000,784))
	 # add bias as a feature with value 1 to all test examples
    test_images = np.c_[np.ones(10000),np.reshape(test_images,(10000,784))]
# load the test labels
with open(sys.argv[4], 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    bytestream.seek(8)
    test_labels = np.frombuffer(bytestream.read(),np.uint8)
# set the learning rate
rate = 0.5
# initialize required arrays
l_a = np.zeros(len(train_images))
l_test = np.zeros(len(test_images))
# initialize the weights for the ten neural units
curr_weights = np.random.normal(0,1,(785,10))
delta_weights = np.zeros((785,10))
### update the weights
for x in range(0,len(train_images)):
    ## forward pass
    # calculate logits
    prod = np.dot(train_images[x],curr_weights)
    # softmax
    expon = [np.exp(z) for z in prod]
    sum_expon = sum(expon)
    prob = np.divide(expon,sum_expon)
    # loss
    l_a[x] = -np.log(prob[train_labels[x]])
    ## backward pass
    for j in range(0,10):
        # update values for weights
        if j==train_labels[x]:
            delta_weights[:,j] = - rate * np.transpose(train_images[x,:]) * -(1-prob[j])
        else:
            delta_weights[:,j] = - rate * np.transpose(train_images[x,:]) * (prob[j])
    # update the weights
    curr_weights = np.add(curr_weights,delta_weights)
# calculate overall loss over the training set
l_train_all = sum(l_a)
### evaluate performance on the test set
# set counter for correct classifications
corr = float(0)
for y in range(0,len(test_images)):
    # calculate logits
    prod = np.dot(test_images[y],curr_weights)
    # softmax
    expon = [np.exp(x) for x in prod]
    sum_expon = sum(expon)
    prob = np.divide(expon,sum_expon)
    # loss
    l_test[y] = -np.log(prob[test_labels[y]])
    # update the counter for correct classifications
    if np.argmax(prob) == test_labels[y]:
        corr+=float(1)
# calculate overall loss in the current development set
l_test_all = sum(l_test)
# calculate the percentage of digits in the test set correctly classified
perc_correct = float(corr/10000)
print "Percent correctly classified in the test set:" + str(perc_correct)
