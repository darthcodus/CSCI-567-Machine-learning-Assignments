import math
import pickle

import pandas as pd

class GaussianBayesClassifier:
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.u_kj = {}
        self.sig_kj = {}
        self.x_jk = {} # per class sigma(xi)
        self.n = 0 # total sample count
        self.nc = {} # Per class sample count
        self.classes = None

    def train(self, train_df):
        """
        Train on a pandas dataframe. Assumes the first n-1 columsn and features, and the last column is the class.
        :param train_df: The pandas dataframe to train on
        """
        self.n = len(train_df)
        cols = list(train_df.columns.values)
        self.classes = train_df.iloc[:, -1].unique()
        if self.verbose:
            print("Initializing Gaussian bayes classifier with:")
            print("Training features (first 5 rows): ", train_df.iloc[:5])
            print("Unique classes:", self.classes)
        for idx, cat in enumerate(self.classes):
            k = cat
            cat_col = train_df[train_df[cols[-1]] == cat]
            self.nc[cat] = (len(cat_col))
            for j in range(0, len(cat_col.iloc[0]) - 1):
                if k not in self.u_kj:
                    self.u_kj[k] = []
                if k not in self.sig_kj:
                    self.sig_kj[k] = []
                self.u_kj[k].append(cat_col.iloc[:, j].mean())
                self.sig_kj[k].append(train_df.iloc[:, j].std(ddof=0)) # Making std independent of category gives better accuracy, why?

    def writeToFile(self, fname):
        pickle.dump(self, open(fname, "wb"))

    @staticmethod
    def loadFromFile(fname):
        return pickle.load(open(fname, "rb"))

    def _getScores(self, feature_vector):
        scores = {}
        for category in self.classes:
            scores[category] = math.log(float(self.nc[category])/self.n)
            for j, xj in enumerate(feature_vector):
                if self.sig_kj[category][j] == 0:
                    continue
                scores[category] -= math.pow(( (float(xj)-self.u_kj[category][j])/self.sig_kj[category][j] ), 2)
        return scores

    def classify(self, df):
        classes = []
        for row in df.values:
            scores = self._getScores(row)
            classes.append( max(scores.keys(), key=(lambda x: scores[x])) )
        return classes
