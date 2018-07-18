import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import roc_auc_score

class NeuralNetwork:

    def __init__(self, inputLen, outputLen, hiddenLayers = []):
        tf.reset_default_graph()
        self.winit = tf.glorot_uniform_initializer()
        self.binit = tf.constant_initializer(0.0)

        self.hiddenLayers = hiddenLayers

        self.inputs = tf.placeholder(tf.float64, [None, inputLen])
        self.labels = tf.placeholder(tf.float64, [None, outputLen])
        self.batchSize = tf.placeholder(tf.float64, [])

        self.learningRate = tf.placeholder_with_default(np.float64(0.01),  [])
        self.regularization = tf.placeholder_with_default(np.float64(0.1),  [])
        self.keepProb = tf.placeholder_with_default(np.float64(1.0),  [])

        lastOutputs = tf.nn.dropout(self.inputs, self.keepProb)
        lastSize = inputLen
        i = 0
        weigths = []
        for l in hiddenLayers:
            scope = "hidden_" + str(i)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                w = tf.get_variable(
                    name = "w",
                    shape = [lastSize, l],
                    dtype = tf.float64,
                    initializer = self.winit)
                b = tf.get_variable(
                    name = "b",
                    shape = [l],
                    dtype = tf.float64,
                    initializer = self.binit)
                lastOutputs = tf.sigmoid(tf.matmul(lastOutputs, w) + b)
                #lastOutputs = tf.nn.relu(tf.matmul(lastOutputs, w) + b)
                lastOutputs = tf.nn.dropout(lastOutputs, self.keepProb)
                lastSize = l
                weigths.append(w)
            i += 1
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            w = tf.get_variable(
                name = "w",
                shape = [lastSize, outputLen],
                dtype = tf.float64,
                initializer = self.winit)
            b = tf.get_variable(
                name = "b",
                shape = [outputLen],
                dtype = tf.float64,
                initializer = self.binit)
            weigths.append(w)

        self.predictions = tf.matmul(lastOutputs, w) + b
        self.cost = tf.losses.softmax_cross_entropy(self.labels, self.predictions, reduction=tf.losses.Reduction.MEAN)
        self.squaredWeigth = tf.add_n([tf.reduce_sum(tf.square(w)) for w in weigths])
        self.ncost = tf.cast(self.cost, tf.float64) + (self.regularization / (self.batchSize * 2.0)) * self.squaredWeigth 

        self.optim = tf.train.AdamOptimizer(self.learningRate).minimize(self.ncost)


        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
    
    def reinit(self):
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

    def train(self, data, epoch, regularization = 0.01, learningRate = 0.01, keepProb = 1.0):
        cost = 0
        for i in tqdm(range(epoch)):
            batchInput, batchOutput = data.trainBatch()
            c = self.session.run(self.cost, feed_dict={
                self.batchSize: len(batchOutput),
                self.inputs: batchInput,
                self.labels: batchOutput
            })
            _ = self.session.run(self.optim, feed_dict={
                self.batchSize: len(batchOutput),
                self.inputs: batchInput,
                self.labels: batchOutput,
                self.learningRate:learningRate,
                self.regularization:regularization,
                self.keepProb:keepProb
            })
            cost += c
        cost = cost / epoch
        print("cost : " + str(cost))
    
    def getCost(self, data):
        batchInput, batchOutput = data.trainBatch()
        trainCost = self.session.run(self.cost, feed_dict={
            self.batchSize: len(batchOutput),
            self.inputs: batchInput,
            self.labels: batchOutput
        })
        batchInput, batchOutput = data.testBatch()
        testCost = self.session.run(self.cost, feed_dict={
            self.batchSize: len(batchOutput),
            self.inputs: batchInput,
            self.labels: batchOutput
        })
        return trainCost, testCost

    def plotCostByReg(self, data, epoch = 5000, learningRate = 0.01, minReg = 0.00, maxReg = 1.0, step = 2):
        trainCost = []
        testCost = []
        xticks = []
        reg = minReg
        while reg < maxReg:
            print("reg " + str(reg) + "/" + str(maxReg))
            self.reinit()
            self.train(data, epoch, reg, learningRate, 1.0)
            trainC, testC = self.getCost(data)
            trainCost.append(trainC)
            testCost.append(testC)
            xticks.append(reg)
            reg *= step
        self.plotLines([trainCost, testCost], ["train cost", "test cost"], xticks, "evolution of the train cost and test cost depending on the regularization factor")

    def check(self, data):
        batchInput, batchOutput = data.testBatch()
        p, cost = self.session.run([self.predictions, self.ncost], feed_dict={
            self.batchSize: len(batchOutput),
            self.inputs: batchInput,
            self.labels: batchOutput,
            self.regularization: 0.0,
            self.keepProb: 1.0
        })
        expected = batchOutput.argmax(axis=1)
        predicted = p.argmax(axis=1)
        ok = np.sum(expected == predicted)
        percent = int((ok / len(predicted)) * 100)
        print("accuracy : " + str(ok) + "/" + str(len(predicted)) + " (" + str(percent) + "%)")
    
    def plotLines(self, arrays, labels, xticks, title):
        fig = plt.figure(figsize=(15, 8))
        lines = []
        for i in range(len(arrays)):
            lines.append(plt.plot(arrays[i], label=labels[i]))

        plt.xticks(np.arange(len(xticks)), xticks)
        #ax = plt.gca()
        #ax.tick_params(axis = 'x', which = 'major', labelsize = 6)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.title(title)
        plt.show()
