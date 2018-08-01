#!/usr/bin/python27
### CSCI2470 - Assignment 8 - Babak Hemmatian - bhemmati - BannerID:B01190949
### import the required modules
import tensorflow as tf
import gym
import time
import numpy as np
### set hyperparameters
gamma = 0.99 # discount factor
learning_rate = 0.001 # used to adjust the gradient
trials = 3 # how many times the model is trained from scratch
episodes = 1000 # number of games in each trial
steps = 500 # max number of steps in each game
### Define the variables
## forward-pass
with tf.device('/cpu:0'):
    # Placeholder for the states in current episode
    state = tf.placeholder(shape=[None,4], dtype = tf.float32)
    ## layers for estimating discounted future reward
    # first feedforward layer
    W = tf.Variable(tf.random_uniform(shape=[4,256], dtype = tf.float32))
    b1 = tf.Variable(tf.random_uniform(shape=[256], dtype = tf.float32))
    l1Output = tf.nn.relu(tf.matmul(state,W) + b1)
    # second feedforward layer
    W2 = tf.Variable(tf.random_uniform(shape=[256,2], dtype = tf.float32))
    b2 = tf.Variable(tf.random_uniform(shape=[2], dtype = tf.float32))
    l2Output = tf.nn.softmax(tf.matmul(l1Output,W2) + b2)
    ## layers for extimating V
    # first feedforward layer
    V1 = tf.Variable(tf.random_normal([2,512], dtype = tf.float32, stddev = .1))
    vb1 = tf.Variable(tf.random_normal([512], dtype = tf.float32, stddev = 0.1))
    v1Out = tf.nn.relu(tf.matmul(l2Output,V1) + vb1)
    # second feedforward layer
    V2 = tf.Variable(tf.random_normal([512,1], dtype = tf.float32, stddev = 0.1))
    vb2 = tf.Variable(tf.random_normal([1], dtype = tf.float32, stddev = 0.1))
    vOut = tf.matmul(v1Out,V2) + vb2
    ## backward-pass
    # placeholders for rewards and actions in current episode
    rewards = tf.placeholder(shape=[None],dtype = tf.float32)
    actions = tf.placeholder(shape=[None],dtype = tf.int32)
    advantage = tf.placeholder(shape=[None],dtype = tf.float32)
    # calculate loss based on the actions taken in current episode
    indices = tf.range(0, tf.shape(l2Output)[0]) * 2 + actions
    actProbs = tf.gather(tf.reshape(l2Output, [-1]),indices)
    # calcualte loss over V
    vLoss = tf.reduce_mean(tf.square(rewards-vOut))
    loss = -tf.reduce_mean(tf.log(actProbs) * advantage)
    # sum the two losses
    loss = loss + 0.25*vLoss
    ### train the network
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainOp = optimizer.minimize(loss)
    ## create the tensorflow session
    init = tf.global_variables_initializer()
    sess = tf.Session()
## create the environment and initialize arrays to store rewards
game = gym.make('CartPole-v1')
totRs = np.zeros([trials,episodes]) # to store each episode's reward value (number of steps, i.e. i)
AvgtotRs = np.zeros([trials]) # to store the average reward value for the last 100 episodes of each trial
## train the model
# timer
print "Started training at "+time.strftime('%l:%M%p')
for j in range(trials):
    tf.reset_default_graph()
    sess.run(init) # re-initialize the weights for each trial
    for i in range(episodes):
        st = game.reset() # reset for each new game
        disRs = np.empty([steps]) # initialize array for discounted returns
        hist = np.empty([steps,7]) # initialize array for current game's history
        Adv = np.empty([steps]) # initialize array for advantage estimates
        for t in range(steps): # run for [steps] timesteps or until done, whichever is first
            # input the current state and receive action recommendations from the network
            st = np.asarray([st])
            nActs,v_est = sess.run([l2Output,vOut],feed_dict={state: st})
            # choose action based on a distribution given by the network
            nAct = np.random.choice(2, p=nActs[0])
            # take the action
            st1,rwd,dn,_ = game.step(nAct)
            # render the environment (with delay) to see a visual repr of the action
            # game.render()
            # time.sleep(0.1)
            # store the state, action, outcome and reward in history
            hist[t,:4] = st # state
            hist[t,4] = nAct # chosen action
            hist[t,5] = rwd # reward 0 or 1
            hist[t,6] = v_est # the output of the value network
            # change the state
            st = st1
            if dn: # if the game has ended
                # calculate the discounted reward for all the state-action pairs that happened
                for k in range(t+1):
                    disRs[k] = 0
                    for w in range(t+1-k):
                        disRs[k] = disRs[k] + (gamma ** w) * hist[w+k,5]
                    Adv[k] = disRs[k] - hist[k,6] # calculate the advantage
                # train the model based on discounted rewards in the current episode
                sess.run(trainOp,feed_dict={state:hist[:t,:4],rewards:disRs[:t],advantage:Adv[:t],actions:hist[:t,4]})
                # record the collected reward
                totRs[j,i] += t
                break
        # print "Episode finished after %s timesteps" % (str(t))
        # record and report average per episode reward in the last 100 episodes of each trial
        if i!= 0 and (i+1) % 1000 == 0:
            AvgtotRs[j] = np.mean(totRs[j,i-99:1000])
print "Finished training at "+time.strftime('%l:%M%p')
print "Average per episode reward in the last 100 episodes of the best trial: "
print max(AvgtotRs)
