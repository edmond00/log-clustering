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
from os import listdir
from os.path import isfile, join
import csv

numberCluster = 9

class ParserError(Exception): pass

class Cluster:

    def __init__(self, numberId, batchSize, testBatchSize, name):
        self.numberId = numberId
        self.name = name
        self.batchSize = batchSize
        self.testBatchSize = testBatchSize
        self.label = np.repeat(0, numberCluster)
        self.label[numberId-1] = 1
        self.data = []
        self.test = []
        self.train = []

    def addData(self, index):
        self.data.append(index)

    def printOccurences(self):
        print(self.name + " : " + str(len(self.data)))

    def splitTrainTest(self, testPercent):
        if len(self.data) == 0:
            return
        self.test = np.random.choice(self.data, int(len(self.data)*testPercent), replace=False)
        self.train = np.array([i for i in self.data if i not in self.test])

    def trainBatch(self, data):
        if len(self.train) == 0:
            return 
        inputsIndex = np.random.choice(self.train, self.batchSize, replace=True)
        inputs = np.take(data.frequencyVectors, inputsIndex, axis=0)
        inputs = np.concatenate([inputs, data.intentOneHotVector(inputsIndex)], axis=1)
        outputs = np.repeat([self.label], self.batchSize, axis=0)
        return inputs, outputs

    def testBatch(self, data):
        if len(self.train) == 0:
            return
        inputsIndex = np.random.choice(self.test, self.testBatchSize, replace=True)
        inputs = np.take(data.frequencyVectors, inputsIndex, axis=0)
        inputs = np.concatenate([inputs, data.intentOneHotVector(inputsIndex)], axis=1)
        outputs = np.repeat([self.label], self.testBatchSize, axis=0)
        return inputs, outputs

class Data:

    def __init__(self):

        self.clusters = {}
        self.addCluster(1, 10, 2000, "OK")
        self.addCluster(2, 10, 200, "NG")
        self.addCluster(3, 10, 200, "検閲")
        self.addCluster(4, 10, 200, "広告")
        self.addCluster(5, 10, 200, "対応不要会話")
        self.addCluster(6, 10, 200, "特殊文字")
        self.addCluster(7, 10, 200, "コード入力系")
        self.addCluster(8, 10, 200, "入力ミス")
        self.addCluster(9, 10, 200, "インナーテスト")
        self.logs = []
        self.logIntents = []
        self.allIntents = []

        print("read data...")

        self.getDataFromDirectory("newLogs", 7, 8, 13)
        self.getDataFromDirectory("oldLogs", 1, 3, 6)
        self.printOccurences()

        for key,cluster in self.clusters.items():
            cluster.splitTrainTest(0.25)

        print("vectorize logs...")

        self.segmenter = tinysegmenter.TinySegmenter()
        self.vectorizer = TfidfVectorizer(min_df=0.0001, tokenizer=self.tokenize)
        #self.vectorizer = TfidfVectorizer(min_df=0.001, tokenizer=self.tokenize)
        vectors = self.vectorizer.fit_transform(self.logs)
        self.frequencyVectors = vectors.toarray()

    def intentOneHotVector(self, indexs):
        result = np.zeros([len(indexs), len(self.allIntents)])
        for i in range(len(indexs)):
            intent = self.logIntents[indexs[i]]
            if intent is not None:
                j = self.allIntents.index(intent)
                result[i][j] = 1.0
        return result
                

    def getCategoryId(self, categoryName):
        for key,cluster in self.clusters.items():
            if cluster.name == categoryName:
                return cluster.numberId 
        raise ParserError("WARNING : can not find cluster id for name '" + categoryName + '"')
        

    def getAllFiles(self, dirname):
        return [f for f in listdir(dirname) if isfile(join(dirname, f))]


    def parseLine(self, line, inputIndex, categoryIndex, intentIndex = None):
        splits = list(csv.reader([line], delimiter=','))[0]
        try:
            if intentIndex is None:
                return splits[inputIndex], self.getCategoryId(splits[categoryIndex]), None
            else:
                return splits[inputIndex], self.getCategoryId(splits[categoryIndex]), splits[intentIndex]
        except IndexError as e:
            raise ParserError("WARNING : index error when parsing '" + line + '"')


    def getDataFromFile(self, dirname, filename, inputIndex, categoryIndex, intentIndex = None):
        f = open(dirname + "/" + filename, "r", encoding='utf-8')
        line = f.readline()
        first = True
        while line:
            if first:
                first = False
            else:
                try:
                    inputString, categoryId, intent = self.parseLine(line, inputIndex, categoryIndex, intentIndex)
                    self.clusters[categoryId].addData(len(self.logs))
                    self.logs.append(inputString)
                    self.logIntents.append(intent)
                    if intent is not None and intent not in self.allIntents:
                        self.allIntents.append(intent)
                except ParserError as e:
                    print(e)
            line = f.readline()

    def getDataFromDirectory(self, dirname, inputIndex, categoryIndex, intentIndex = None):
        allfiles = self.getAllFiles(dirname)
        print("get data from " + dirname)
        for onefile in tqdm(allfiles):
               self.getDataFromFile(dirname, onefile, inputIndex, categoryIndex, intentIndex)


    def addCluster(self, numberId, batchSize, testBatchSize, name):
        self.clusters[numberId] = Cluster(numberId, batchSize, testBatchSize, name)

    def printOccurences(self):
        for key,cluster in self.clusters.items():
            cluster.printOccurences()

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
    
    def logToVec(self, log, intent = None):
        ohv = np.zeros(len(self.allIntents))
        if intent is not None and intent in self.allIntents:
            j = self.allIntents.index(intent)
            ohv[j] = 1.0
        vector = np.concatenate([self.vectorizer.transform([log]).toarray()[0], ohv])
        return vector
        
    def trainBatch(self):
        return self.batch(testing=False)
        
    def testBatch(self):
        return self.batch(testing=True)

    def batch(self, testing):
        inputs = []
        outputs = []
        for key, cluster in self.clusters.items():
            if len(cluster.data) > 0:
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
        return len(self.vectorizer.get_feature_names()) + len(self.allIntents)

    def outputLen(self):
        return len(self.clusters)

    def getCategoryIndex(self, categoryName):
        for key, cluster in self.clusters.items():
            if cluster.name == categoryName:
                return key -1


