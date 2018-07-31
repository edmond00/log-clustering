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

        self.saver = tf.train.Saver()


        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

    def save(self, savename):
        self.saver.save(self.session, savename)

    def restore(self, savename):
        self.saver.restore(self.session, savename)
    
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
            xticks.append(round(reg, 5))
            reg *= step
        self.plotLines([trainCost, testCost], ["train cost", "test cost"], xticks, "evolution of the train cost and test cost depending on the regularization factor")

    
    def predict(self, batchInput):
        p= self.session.run(self.predictions, feed_dict={
            self.batchSize: len(batchInput),
            self.inputs: batchInput,
            self.regularization: 0.0,
            self.keepProb: 1.0
        })
        return p

    def check(self, data):
        batchInput, batchOutput = data.testBatch()
        p = self.predict(batchInput)
        expected = batchOutput.argmax(axis=1)
        predicted = p.argmax(axis=1)
        ok = np.sum(expected == predicted)
        percent = int((ok / len(predicted)) * 100)
        print("accuracy : " + str(ok) + "/" + str(len(predicted)) + " (" + str(percent) + "%)")
    
    def printResult(self, data, log, intent = None):
        batchInput = np.array([data.logToVec(log, intent)])
        p = self.predict(batchInput)[0]
        predictedLabel = p.argmax()
        print("")
        print("'" + log + "' predicted as " + data.clusters[predictedLabel+1].name)
        print("")
        print("details :")
        for i in range(len(p)):
            print(data.clusters[i+1].name + " : " + str(p[i]))


    def writeCsvLine(self, f, data, ilog, clusterName, trainOrTest):
        batchInput = np.array([data.logToVec(data.logs[ilog], data.logIntents[ilog])])
        p = self.predict(batchInput)[0]
        predictedLabel = p.argmax()
        labelName = data.clusters[predictedLabel+1].name

        csvLine = []
        csvLine.append('"' + data.logs[ilog] + '"')
        csvLine.append(data.logIntents[ilog])
        csvLine.append(labelName)
        csvLine.append(clusterName)
        csvLine.append(trainOrTest)
        f.write(",".join(csvLine))
        f.write("\n")

    def saveResultInFile(self, data, filename):
        f = open(filename, "w")
        f.write('\ufeff' + "input,intents,predicted,expected,used for\n")
        for key,cluster in data.clusters.items():
            print(cluster.name + " : write test set")
            for ilog in tqdm(cluster.test):
                self.writeCsvLine(f, data, ilog, cluster.name, "test")
            print(cluster.name + " : write train set")
            for ilog in tqdm(cluster.train):
                self.writeCsvLine(f, data, ilog, cluster.name, "train")
            

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

    def specificResult(self, data, categoryName):
        positiveIndex = data.getCategoryIndex(categoryName)
        batchInput, batchOutput = data.testBatch()
        p = self.predict(batchInput)

        labels = batchOutput.argmax(axis=1)
        predictions = p.argmax(axis=1)

        truePositives = len(predictions[(predictions == labels) & (labels == positiveIndex)])
        trueNegatives = len(predictions[(predictions != positiveIndex) & (labels != positiveIndex)])
        falsePositives = len(predictions[(predictions != labels) & (predictions == positiveIndex)])
        falseNegatives = len(predictions[(labels == positiveIndex) & (predictions != positiveIndex)])
        truePositivesPercent = truePositives / len(labels[labels == positiveIndex])
        trueNegativesPercent = trueNegatives / len(labels[labels != positiveIndex])
        falsePositivesPercent = falsePositives / len(labels[labels != positiveIndex])
        falseNegativesPercent = falseNegatives / len(labels[labels == positiveIndex])

        precision = 0
        recall = 0
        f1 = 0
        if truePositives > 0:
            precision = truePositives / (truePositives+falsePositives)
            recall = truePositives / (truePositives+falseNegatives)
            f1 = 2 * (recall*precision) / (recall+precision)

        print("result for " + categoryName)
        print("true positives : " + str(int(truePositivesPercent*100)) + "%")
        print("false positives : " + str(int(falsePositivesPercent*100)) + "%")
        print("true negatives : " + str(int(trueNegativesPercent*100)) + "%")
        print("false negatives : " + str(int(falseNegativesPercent*100)) + "%")
        print("precision : " + str(int(precision*100)) + "%")
        print("recall : " + str(int(recall*100)) + "%")
        print("f1 score : " + str(int(f1*100)) + "%")

        return {
            "truePositives": truePositives,
            "falsePositives": falsePositives,
            "trueNegatives": trueNegatives,
            "falseNegatives": falseNegatives,
            "truePositivesPercent": truePositivesPercent,
            "falsePositivesPercent": falsePositivesPercent,
            "trueNegativesPercent": trueNegativesPercent,
            "falseNegativesPercent": falseNegativesPercent,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
