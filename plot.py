import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams


class Plot:

    def __init__(self, data, nn):
        self.data = data
        self.nn = nn

    def addMean(self, l):
        l.append(sum(l) / len(l))

    def plotResultBarChart(self):
        
        
        tp = []
        fp = []
        tn = []
        fn = []
        precisions = []
        recalls = []
        f1 = []
        category = []

        for key,cluster in self.data.clusters.items():
            if len(cluster.test) > 0:
                category.append(cluster.name)
                results = self.nn.specificResult(self.data, cluster.name)
                tp.append(results["truePositivesPercent"])
                tn.append(results["trueNegativesPercent"])
                fp.append(results["falsePositivesPercent"])
                fn.append(results["falseNegativesPercent"])
                precisions.append(results["precision"])
                recalls.append(results["recall"])
                f1.append(results["f1"])
        category.append("平均")
        self.addMean(tp)
        self.addMean(tn)
        self.addMean(fp)
        self.addMean(fn)
        self.addMean(precisions)
        self.addMean(recalls)
        self.addMean(f1)

        n_groups = len(f1)
        
        fig, ax = plt.subplots()

        font = FontProperties(fname='./Osaka.ttc', size=10)
        rcParams['font.family'] = font.get_name()

        bar_width = 0.05
        opacity = 0.8
        
        index = np.arange(n_groups)
        
        w = 0
        ax.bar(index+w, tp, bar_width,
            alpha=opacity, color='lime',
            label='true positives')
        w += bar_width
        ax.bar(index+w, tn, bar_width,
            alpha=opacity, color='green',
            label='true negatives')
        w += bar_width
        ax.bar(index+w, fp, bar_width,
            alpha=opacity, color='red',
            label='false positives')
        w += bar_width
        ax.bar(index+w, fn, bar_width,
            alpha=opacity, color='orangered',
            label='false negatives')
        w += bar_width*2
        ax.bar(index+w, precisions, bar_width,
            alpha=opacity, color='blue',
            label='precision')
        w += bar_width
        ax.bar(index+w, recalls, bar_width,
            alpha=opacity, color='mediumslateblue',
            label='recall')
        w += bar_width
        ax.bar(index+w, f1, bar_width,
            alpha=opacity, color='cyan',
            label='f1 score')
        
        ax.set_xlabel('Group')
        ax.set_ylabel('Score')
        ax.set_title('log clustering results on testing dataset')
        ax.set_xticks(index + bar_width * 4)
        ax.set_xticklabels(category)
        ax.legend()
        
        fig.tight_layout()
        plt.show()

