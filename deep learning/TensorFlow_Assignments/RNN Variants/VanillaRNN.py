
import glob, os, string, collections, math

import tensorflow as tf
import numpy as np

#supress annoying wanring
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.reset_default_graph() #clears graph from last run (if you want to run again)... 
                         #HOWEVER, it pick up training where it left off, unless you delete the "tensorboard_logs" dir
                         #And the model_files dir (in your current working directory)


WINDOW_SZ = 20
BATCH_SZ = 100
EMBED_SZ = 30
LEARN_RATE = 1e-4
VOCAB_SZ = 8000
SAVE_FREQ = 10 #save every 10 steps
#only one epoch, so no epoch var

BATCH_CAP = 5 #cap number of bathches so lab doesnt take too long

#for simplicity, the size of our state will be the size of the y we want out.
STATE_SZ = VOCAB_SZ

MODEL_DIR = os.path.join(os.getcwd(), "model_files")
MODEL_PATH = os.path.join(MODEL_DIR, "model")
SUMMARY_DIR = os.path.join(os.getcwd(), "tensorboard_logs")
CORPUS_PATH = os.path.join(os.getcwd(), "corpus.txt") #define path str. (works on any OS)


class VanillaRNN:
    def __init__(self):
        # define session and grpah
        self.sess = tf.Session()
        self.defineGraph()
        self.saver = tf.train.Saver()

        #check if the model exists already
        checkPoint = tf.train.get_checkpoint_state(MODEL_DIR)
        modelExists = checkPoint and checkPoint.model_checkpoint_path

        #if it exists, load weights. Otherwise, init all weights.
        if modelExists:
            self.saver.restore(self.sess, checkPoint.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def defineGraph(self):
        #for plotting / visualization
        self.train_writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)

        #keep a variable representing the step we are on. make sure it is not trainable.
        self.glob_step = tf.Variable(0, name="global_step", trainable=False)

        self.initializer = tf.random_normal_initializer(stddev=0.1) #init vars from a noraml distribution

        self.batch_in = tf.placeholder(tf.int64, shape=[BATCH_SZ, WINDOW_SZ]) 

        self.prev_state = tf.placeholder(tf.float32, shape=[BATCH_SZ, STATE_SZ]) 
        
        self.embeddings = tf.Variable(tf.random_uniform([VOCAB_SZ, EMBED_SZ],-1.0,1.0))

        #should have shape=[BATCH_SZ, WINDOW_SZ, EMBED_SZ]
        self.embedded_batch_in = tf.nn.embedding_lookup(self.embeddings, self.batch_in)
        
        self.labels = tf.placeholder(tf.int32, shape=[BATCH_SZ, WINDOW_SZ])


        # variables bellow are not stored in "self" because you will never need to ask sess for them. very hidden states.
        # use "get vairiable" so i can give them names


        #weights need to go from [x_t + s_t-1] to vector of size [STATE_SZ]
        weights = tf.get_variable("W_IN", dtype=tf.float32, shape=[EMBED_SZ+STATE_SZ, STATE_SZ], initializer=self.initializer)
        biases = tf.get_variable("B_IN", dtype=tf.float32, shape=[STATE_SZ], initializer=self.initializer)


        #define recurrent part of network. If you use Tensorflows LSTM later on, this for loop will be done for you
        states = [self.prev_state] #a list of state tensors
        for i in xrange(WINDOW_SZ):   
            curr_batch_input = self.embedded_batch_in[:,i,:] #get the input for time step i across all batches
            #concatenate with previous state along row axis
            cur_state = states[-1]
            concat_last_state_and_input = tf.concat([cur_state, curr_batch_input], 1)

            new_state = tf.matmul(concat_last_state_and_input, weights) + biases
            states.append(new_state)

        #out_states should have shape [batch_sz, "window_sz", state_sz] = [batch_sz, window_sz, vocab_sz] in our case
        self.out_states = tf.stack(states[1:], axis=1) #turn list of tensors into one tensor. ignore first state fed in.

        #soft max along our "window_sz" dimension
        self.y_probabilities = tf.nn.softmax(self.out_states, dim=1)

        #loss is average cross entropy across all words at all timesetos
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out_states, labels=self.labels))
        
        #keep a log of our loss function so we can visualize with Tensoarboard
        tf.summary.scalar("loss", self.loss)

        self.trainOp = tf.train.AdamOptimizer(learning_rate=LEARN_RATE).minimize(self.loss, global_step=self.glob_step)

        #necessary step to save all summary logs for visualize with Tensoarboard
        self.mergedSummaries = tf.summary.merge_all()

    def save(self):
        self.saver.save(self.sess, MODEL_PATH, global_step = self.glob_step)

    def train(self, batchesData, batchesLabels):
        previousState = np.zeros([BATCH_SZ, STATE_SZ]) #one state for each batch

        for step in xrange(len(batchesData)):
            print "\nstep", step, "out of", len(batchesData)-1

            #not yet caught up to where model has already trained:
            if step < tf.train.global_step(self.sess, self.glob_step):
                print "already done"
                continue

            batchData = batchesData[step]
            batchLabels = batchesLabels[step]


            feedDict = {self.batch_in: batchData, self.labels: batchLabels, self.prev_state: previousState}
            sessArgs = [self.out_states, self.y_probabilities, self.loss, self.mergedSummaries, self.trainOp]

            # RUN #
            hiddenStates, probs, lossReturned, summary, _ = self.sess.run(sessArgs, feed_dict=feedDict)

            #we are striding by window size, so use last state for next time
            previousState = hiddenStates[:,-1,:] #take all states across batches from last step in window

            print "loss -", lossReturned

            predictedLabels = np.argmax(probs, axis=2)
            numCorrect = np.sum(predictedLabels == batchLabels)
            numInBatch = np.prod(np.array(batchLabels).shape)

            print "perplexity -", math.e**lossReturned

            print "accuracy -", float(numCorrect)/float(numInBatch)

            #for visualizatoin. write your summary logs. 
            self.train_writer.add_summary(summary, step) 

            #save your model every once and a while
            if step%SAVE_FREQ==0:
                self.save()



#helper func:

#return same string with only letters left
def filterOnlyLetters(strIn):
    letterSet=set(string.letters)
    return filter(lambda char: char in letterSet, strIn)


def tokenizer(strIn):
    """
    splits a string into lowercase words with only letters (no punctoination or numbers)
    Args: 
        strIn (string): the sentence
    Returns:
        (list of str): a list of the words wihout punctination and all lower case

    """
    spaceSeperatedFragments = strIn.strip().split() #seperate on white space
    words = map(filterOnlyLetters, spaceSeperatedFragments) #remove punctuation and numbers
    words = filter(lambda word: word!="", words) #filter empty words
    wordsLowerCase = [w.lower() for w in words]

    return wordsLowerCase


#main. read in data then train
if __name__ == "__main__":
    #read in data into batches then train...
    

    #read in corpus as words then process to number IDs.
    corpusWords = [] #turn corpus into a list of words
    
    with open(CORPUS_PATH, "r") as corpusFile:
        lines = corpusFile.readlines()

    for line in lines:
        corpusWords.extend(tokenizer(line))

    #enforce VOCAB_SZ only keeping VOCAB_SZ-1 most common words. 
    #(We will add the word *UNK* to replace all of these)
    wordCounts = collections.Counter(corpusWords)
    commonWordCountTuples = wordCounts.most_common(VOCAB_SZ-1)
    commonWords = zip(*commonWordCountTuples)[0]
    allowedWords = set(list(commonWords)+["*UNK*"])

    corpusWordIDs = [] #corpus of numbers representing words
    wordToIDs = {}
    nextIDToUse = 0

    for word in corpusWords:
        word = word if word in allowedWords else "*UNK*" #if not allowed, make "*UNK*"
        if word not in wordToIDs: #if word not in our dictionary, buy it valid, give it ID
            wordToIDs[word] = nextIDToUse
            nextIDToUse += 1
        corpusWordIDs.append(wordToIDs[word]) #record next ID



    # #split corpus Ids into windows
    windowData = []
    windowLabels = []
    for i in xrange(0, len(corpusWordIDs)-WINDOW_SZ-1, WINDOW_SZ):
        data = corpusWordIDs[i : i+WINDOW_SZ]
        labels = corpusWordIDs[i+1 : i+WINDOW_SZ+1]
        windowData.append(data)
        windowLabels.append(labels)

    #split windows into batches that opperate in parallel
    batchesData = []
    batchesLabels = []
    numBatches = len(windowData)/BATCH_SZ # number of batches that will fit. also length of one sequence (in windows).
    for batchIndex in xrange(0, numBatches):
        batchData = [windowData[i*numBatches + batchIndex] for i in xrange(BATCH_SZ)]
        batchLabel = [windowLabels[i*numBatches + batchIndex] for i in xrange(BATCH_SZ)]
        batchesData.append(batchData)
        batchesLabels.append(batchLabel)

    #cap the number of batches so lab doesnt take took long
    batchesData = batchesData[:BATCH_CAP]
    batchesLabels = batchesLabels[:BATCH_CAP]


    #INIT neural net
    rnnModel = VanillaRNN()
    #Train neural net
    rnnModel.train(batchesData, batchesLabels)




