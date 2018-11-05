import tensorflow as tf

tf.reset_default_graph() #clears graph from last time

INPUT_SZ = 2
STATE_SZ = 3

WINDOW_SZ = 20
BATCH_SZ = 100

batch_in = tf.placeholder(tf.float32, shape=[BATCH_SZ, WINDOW_SZ, INPUT_SZ]) 
prev_state = tf.placeholder(tf.float32, shape=[BATCH_SZ, STATE_SZ]) 

#wieghts for read and update. These are our masks.
#### (why are these combined and not W_r and U_r independantly???)
Weight_r = tf.Variable(initial_value=tf.random_normal(dtype=tf.float32, shape=[INPUT_SZ+STATE_SZ, STATE_SZ])) 
Weight_z = tf.Variable(initial_value=tf.random_normal(dtype=tf.float32, shape=[INPUT_SZ+STATE_SZ, STATE_SZ])) 
Bias_r = tf.Variable(initial_value=tf.random_normal(dtype=tf.float32, shape=[STATE_SZ])) 
Bias_z = tf.Variable(initial_value=tf.random_normal(dtype=tf.float32, shape=[STATE_SZ])) 


#weights for creating new state
Weight_Proposal = tf.Variable(initial_value=tf.random_normal(dtype=tf.float32, shape=[INPUT_SZ+STATE_SZ, STATE_SZ])) 
Bias_Proposal = tf.Variable(initial_value=tf.random_normal(dtype=tf.float32, shape=[STATE_SZ])) 


#define recurrent part of network. If you use Tensorflows LSTM later on, this for loop will be done for you
states = [prev_state] #a list of state tensors
for i in xrange(WINDOW_SZ):   
    curr_batch_input = batch_in[:,i,:] #get the input for time step i across all batches

    #concatenate with previous state along row axis
    cur_state = states[-1]
    concat_last_state_and_input = tf.concat([cur_state, curr_batch_input], 1)

    #now we have our masks
    r = tf.sigmoid(tf.matmul(concat_last_state_and_input, Weight_r) + Bias_r)
    z = tf.sigmoid(tf.matmul(concat_last_state_and_input, Weight_z) + Bias_z)

    #"*"" is element-wise
    current_state_masked = cur_state*r
    proposal_state_input = tf.concat([current_state_masked, curr_batch_input], 1)
    proposal_state = tf.tanh(tf.matmul(proposal_state_input, Weight_Proposal) + Bias_Proposal)

    new_state = cur_state*z + proposal_state*(1-z) #using "1" in tf will popilate a tensor of all 1's of whatever size you need
    
    states.append(new_state)
