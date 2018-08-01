#!/usr/bin/python27
### CSCI2470 - Assignment 7 - Babak Hemmatian - bhemmati - BannerID:B01190949
### import the required modules
import tensorflow as tf
import gym
import time
import numpy as np
### set hyperparameters
gamma = 0.9999 # discount factor
learning_rate = 0.005 # used to adjust the gradient
trials = 3 # how many times the model is trained from scratch
episodes = 1000 # number of games in each trial
steps = 300 # max number of steps in each game
### Define the variables
## forward-pass
# Placeholder for the states in current episode
state = tf.placeholder(shape=[None,4], dtype = tf.float32)
# first feedforward layer
W = tf.Variable(tf.random_uniform(shape=[4,8], dtype = tf.float32))
b1 = tf.Variable(tf.random_uniform(shape=[8], dtype = tf.float32))
hidden = tf.nn.relu(tf.matmul(state,W) + b1)
# second feedforward layer
O = tf.Variable(tf.random_uniform(shape=[8,2], dtype = tf.float32))
b2 = tf.Variable(tf.random_uniform(shape=[2], dtype = tf.float32))
output = tf.nn.softmax(tf.matmul(hidden,O) + b2)
## backward-pass
# placeholders for rewards and actions in current episode
rewards = tf.placeholder(shape=[None],dtype = tf.float32)
actions = tf.placeholder(shape=[None],dtype = tf.int32)
# calculate loss based on the actions taken in current episode
indices = tf.range(0, tf.shape(output)[0]) * 2 + actions
actProbs = tf.gather(tf.reshape(output, [-1]),indices)
loss = -tf.reduce_mean(tf.log(actProbs) * rewards)
### train the network
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainOp = optimizer.minimize(loss)
## create the tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
## create the environment and initialize arrays to store rewards
game = gym.make('CartPole-v0')
totRs = np.zeros([trials,episodes]) # to store each episode's reward value (number of steps, i.e. i)
AvgtotRs = np.zeros([trials]) # to store the average reward value for the last 100 episodes of each trial
## train the model
# timer
print "Started training at "+time.strftime('%l:%M%p')
for j in range(trials):
    sess.run(init) # re-initialize the weights for each trial
    for i in range(episodes):
        st = game.reset() # reset for each new game
        disRs = np.empty([steps]) # initialize array for discounted returns
        hist = np.empty([steps,6]) # initialize array for current game's history
        for t in range(steps): # run for [steps] timesteps or until done, whichever is first
            # input the current state and receive action recommendations from the network
            st = np.asarray([st])
            nActs = sess.run(output,feed_dict={state: st})
            # choose action based on a distribution given by the network
            nAct = np.random.choice(2, p=nActs[0])
            # take the action
            st1,rwd,dn,_ = game.step(nAct)
            # render the environment (with delay) to see a visual repr of the action
            # game.render()
            # time.sleep(0.1)
            # store the state, action, outcome and reward in history
            hist[t,:4] = st
            hist[t,4] = nAct
            hist[t,5] = rwd
            # change the state
            st = st1
            if dn: # if the game is over
                # calculate the discounted reward for all the state-action pairs that happened
                for k in range(t):
                    disRs[k] = 0
                    for w in range(t-1-k):
                        disRs[k] = disRs[k] + (gamma ** w) * hist[k+1,5]
                # train the model based on discounted rewards in the current episode
                sess.run(trainOp,feed_dict={state:hist[:t,:4],rewards:disRs[:t],actions:hist[:t,4]})
                # record the collected reward
                totRs[j,i] += t
                break
        # print "Episode finished after %s timesteps" % (str(t))
        # record and report average per episode reward in the last 100 episodes of each trial
        if i!= 0 and (i+1) % 1000 == 0:
            AvgtotRs[j] = np.mean(totRs[j,i-99:1000])
            print "Average per episode reward in the last 100 episodes of trial %s: " % (j+1) + str(AvgtotRs[j])
print "Finished training at "+time.strftime('%l:%M%p')
print "Average per episode reward in the last 100 episodes of the three trials: "
print np.mean(AvgtotRs)
