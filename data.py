import random
import re
import numpy as np
from sklearn.manifold import TSNE
import tinysegmenter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

numberCluster = 8

class Cluster:

    def __init__(self, numberId, batchSize, name):
        self.numberId = numberId
        self.batchSize = batchSize
        self.label = np.repeat(0, numberCluster)
        self.label[numberId-1] = 1
        self.data = []
        self.test = []
        self.train = []

    def addData(self, index):
        self.data.append(index)

    def splitTrainTest(self, testPercent):
        self.test = np.random.choice(self.data, int(len(self.data)*testPercent), replace=False)
        self.train = np.array([i for i in self.data if i not in self.test])

    def trainBatch(self, data):
        inputsIndex = np.random.choice(self.train, self.batchSize, replace=True)
        inputs = np.take(data.frequencyVectors, inputsIndex, axis=0)
        outputs = np.repeat([self.label], self.batchSize, axis=0)
        return inputs, outputs

    def testBatch(self, data):
        inputsIndex = np.random.choice(self.test, self.batchSize, replace=True)
        inputs = np.take(data.frequencyVectors, inputsIndex, axis=0)
        outputs = np.repeat([self.label], self.batchSize, axis=0)
        return inputs, outputs

class Data:

    def addCluster(self, numberId, batchSize, name):
        self.clusters[numberId] = Cluster(numberId, batchSize, name)

    def __init__(self):

        self.clusters = {}
        self.addCluster(1, 10, "OK")
        self.addCluster(2, 10, "NG")
        self.addCluster(3, 10, "検閲")
        self.addCluster(4, 10, "広告")
        self.addCluster(5, 10, "対応不要会話")
        self.addCluster(6, 10, "特殊文字")
        self.addCluster(7, 10, "コード入力系")
        self.addCluster(8, 10, "入力ミス")

        f = open("logs", "r")
        line = f.readline()
        self.logs = []
        index = 0
        while line:
            splits = line.split("\t")
            self.logs.append(splits[0])
            label = int(splits[1])
            self.clusters[label].addData(index)
            line = f.readline()
            index += 1

        for key,cluster in self.clusters.items():
            cluster.splitTrainTest(0.25)

        self.segmenter = tinysegmenter.TinySegmenter()
        self.vectorizer = TfidfVectorizer(min_df=0.0001, tokenizer=self.tokenize)
        #self.vectorizer = TfidfVectorizer(min_df=0.001, tokenizer=self.tokenize)
        print("vectorize logs...")
        vectors = self.vectorizer.fit_transform(self.logs)
        self.frequencyVectors = vectors.toarray()

    def tokenize(self, log):
        log = re.sub(r"([\041-\100\133-\140\173-\176])", r" \1 ", log)
        return [s.strip() for s in self.segmenter.tokenize(log) if len(s.strip()) > 0]

    def getTokens(self):
        return sorted(zip(self.vectorizer.idf_, self.vectorizer.get_feature_names()))
    
    def printFrequency(self, log):
        vector = self.vectorizer.transform([log]).toarray()[0]
        tokens = self.vectorizer.get_feature_names()
        for i in range(len(vector)):
            if vector[i] > 0.0:
                print(tokens[i] + " : " + str(vector[i]))
    
    def logToVec(self, log):
        vector = self.vectorizer.transform([log]).toarray()[0]
        return vector
        
    def trainBatch(self):
        return self.batch(testing=False)
        
    def testBatch(self):
        return self.batch(testing=True)

    def batch(self, testing):
        inputs = []
        outputs = []
        for key, cluster in self.clusters.items():
            if testing:
                i,o = cluster.testBatch(self)
            else:
                i,o = cluster.trainBatch(self)
            inputs.append(i)
            outputs.append(o)
        inputs = np.concatenate(inputs, axis=0)
        outputs = np.concatenate(outputs, axis=0)
        ri = np.arange(len(inputs))
        np.random.shuffle(ri)
        inputs = inputs[ri]
        outputs = outputs[ri]
        return inputs, outputs

    def inputLen(self):
        return len(self.vectorizer.get_feature_names())

    def outputLen(self):
        return len(self.clusters)

