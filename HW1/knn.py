import logging

import numpy as np
import pandas as pd

class KNN:
    def __init__(self, classes, train_df, k, distance = None, verbose = False):
        logging.debug('Creating KNN...')
        self.verbose = verbose
        self.points = train_df
        self.classes = classes
        self.k = k
        self.distance = lambda a,b : np.linalg.norm(a-b, ord = 2)
        if self.distance is not None:
            self.distance = distance
        # self.cache = []

    def classify(self, point, leave_one_out = False):
        dist_class_pairs = [[None,self.classes[idx]] for idx in xrange(0,len(self.points))]
        for idx, row in enumerate(self.points):
            dist_class_pairs[idx][0] = self.distance(row, point)
        dist_class_pairs.sort(key = lambda pair: pair[0])
        counts = {}
        for idx in range(0, min(self.k, len(dist_class_pairs))):
            dist, cat = dist_class_pairs[idx]
            if self.verbose:
                print("Point %d: Distance: " + dist + ", class: " + cat )
            counts[cat] += 1
        return max(counts, key=lambda x: counts[x])
