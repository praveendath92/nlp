import numpy as np
from os import listdir
from os.path import isfile, join
import re
from random import randint
import tensorflow as tf
import progressbar

"""
Sentiment analyses.
"""
class Sentiment:
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    maxSeqLength = 250
    totalSize = 30000
    batchSize = 24
    lstmUnits = 64
    numClasses = 2
    iterations = 100000
    numDimensions = 300

    def read_vocabulary(self, wordListFile="data/wordsList.npy", wordVectorFile="data/wordVectors.npy"):
        wordsList = np.load(wordListFile)
        print('Loaded the word list!')
        wordsList = wordsList.tolist()  # Originally loaded as numpy array
        self.wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
        self.wordVectors = np.load(wordVectorFile)
        print('Loaded the word vectors!')

    """
        Read reviews from file and integerize them. Saves them in given ids_file.
    """
    def read_reviews_from_file(self, positiveDir="data/positiveReviews/", negativeDir="data/negativeReviews/",
                               ids_file="data/idsMatrix.npy"):
        positiveFiles = [positiveDir + f for f in listdir(positiveDir) if
                         isfile(join(positiveDir, f))]
        negativeFiles = [negativeDir + f for f in listdir(negativeDir) if
                         isfile(join(negativeDir, f))]
        posProgBar = progressbar.ProgressBar(max_value=len(positiveFiles))
        negProgBar = progressbar.ProgressBar(max_value=len(negativeFiles))

        self.ids = np.zeros((len(positiveFiles)+len(negativeFiles), self.maxSeqLength), dtype='int32')
        print('Reading Positive files ...')
        fileCounter = 0
        for i,pf in enumerate(positiveFiles):
            with open(pf, "r", encoding="UTF-8") as f:
                indexCounter = 0
                line = f.readline()
                cleanedLine = self.cleanup_sentence(line)
                split = cleanedLine.split()
                for word in split:
                    try:
                        self.ids[fileCounter][indexCounter] = self.wordsList.index(word)
                    except ValueError:
                        self.ids[fileCounter][indexCounter] = 399999  # Vector for unkown words
                    indexCounter = indexCounter + 1
                    if indexCounter >= self.maxSeqLength:
                        break
                fileCounter = fileCounter + 1
            posProgBar.update(i)

        for i,nf in enumerate(negativeFiles):
            with open(nf, "r", encoding="UTF-8") as f:
                indexCounter = 0
                line = f.readline()
                cleanedLine = self.cleanup_sentence(line)
                split = cleanedLine.split()
                for word in split:
                    try:
                        self.ids[fileCounter][indexCounter] = self.wordsList.index(word)
                    except ValueError:
                        self.ids[fileCounter][indexCounter] = 399999  # Vector for unkown words
                    indexCounter = indexCounter + 1
                    if indexCounter >= self.maxSeqLength:
                        break
                fileCounter = fileCounter + 1
            negProgBar.update(i)

        if ids_file:
            np.save(ids_file, self.ids)

    """
        Read reviews from a numpy matrix file.
    """
    def read_reviews_from_cache(self, np_matrix_file="data/idsMatrix.npy"):
        self.ids = np.load(np_matrix_file)


    """
        Init tensorflow.
    """
    def init_tf(self):
        tf.reset_default_graph()

        self.labels = tf.placeholder(tf.float32, [None, self.numClasses])
        self.input_data = tf.placeholder(tf.int32, [None, self.maxSeqLength])
        self.lstmKeepProb = tf.placeholder(tf.float32)

        data = tf.nn.embedding_lookup(self.wordVectors, self.input_data)

        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=self.lstmKeepProb)
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        self.prediction = (tf.matmul(last, weight) + bias)

        correctPred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(senti.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)

    """
    Init tf network for training.
    """
    def init_tf_train(self):
        self.init_tf()

    """
    Init tf network for testing.
    """
    def init_tf_test(self):
        self.init_tf()


    def train_batch(self, iterations=iterations, save_path="data/models/pretrained_lstm.ckpt"):
        self.init_tf_train()
        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

        bar = progressbar.ProgressBar(max_value=iterations)
        print("Training ...")
        for i in range(iterations):
            # Next Batch of reviews
            nextBatch, nextBatchLabels = self.getTrainBatch(i)
            self.sess.run(self.optimizer, {self.input_data: nextBatch, self.labels: nextBatchLabels, self.lstmKeepProb:0.75})

            # Write summary to Tensorboard
            if (i % 5 == 0):
                bar.update(i)

            # Save the network every 10,000 training iterations
            if (i % 1000 == 0 and i != 0):
                saver.save(self.sess, save_path, global_step=i)


    def train_single(self, sentence, expected_output, iterations=10, save_path="data/models/pretrained_lstm.ckpt"):
        self.init_tf_train()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        print("Training on single example...")
        sentence_vec = self.sentence_to_vec(sentence)
        for i in range(iterations):
            self.sess.run(self.optimizer, {self.input_data: sentence_vec, self.labels: [expected_output], self.lstmKeepProb:0.75})

    def save_model(self, global_step, save_path="data/models/"):
        saver = tf.train.Saver(max_to_keep=5)
        saver.save(self.sess, save_path, global_step=global_step)

    def load_model(self, model_path="data/models/"):
        self.init_tf_test()
        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(model_path))


    def test_batch(self, iterations = 10):
        for i in range(iterations):
            nextBatch, nextBatchLabels = self.getTestBatch()
            print("Accuracy for this batch:",
                  (self.sess.run(self.accuracy, {self.input_data: nextBatch, self.labels: nextBatchLabels, self.lstmKeepProb: 1.0})) * 100)

    def sentence_to_vec(self, sentence):
        cleanedLine = self.cleanup_sentence(sentence)
        split = cleanedLine.split()
        sentence_vec = np.zeros((1, self.maxSeqLength), dtype='int32')
        indexCounter = 0
        for word in split:
            try:
                sentence_vec[0][indexCounter] = self.wordsList.index(word)
            except ValueError:
                sentence_vec[0][indexCounter] = 399999  # Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter >= self.maxSeqLength:
                break
        return sentence_vec



    def test_single(self, sentence):
        sentence_vec = self.sentence_to_vec(sentence)
        print("Prediction:", self.sess.run(tf.argmax(self.prediction, 1), {self.input_data: sentence_vec, self.lstmKeepProb:1.0}))

    """
        Cleanup sentence.
    """
    def cleanup_sentence(self, sentence):
        sentence = sentence.lower().replace("<br />", " ")
        return re.sub(self.strip_special_chars, "", sentence.lower())

    def getTrainBatch(self, iteration, batchSize=batchSize):
        labels = []
        arr = np.zeros([batchSize, self.maxSeqLength])
        for i in range(batchSize):
            ind = iteration + i
            if (ind % 2 == 0):
                num = 1 + int(ind/2) % 11499
                labels.append([1, 0])
            else:
                num = 13500 + (int(ind/2) % 11499)
                labels.append([0, 1])
            arr[i] = self.ids[num - 1:num]
        return arr, labels

    def getTestBatch(self, batchSize=batchSize):
        labels = []
        arr = np.zeros([batchSize, self.maxSeqLength])
        for i in range(batchSize):
            num = randint(11499, 13499)
            if (num <= 12499):
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            arr[i] = self.ids[num - 1:num]
        return arr, labels


senti = Sentiment()
senti.read_vocabulary()
#senti.read_reviews_from_file()
senti.read_reviews_from_cache()
senti.train_batch()
senti.load_model()
senti.test_batch()
senti.test_single("I am not good")
senti.test_single("I am very good. I am amazing really. This should be positive right?")