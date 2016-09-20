import logging

import numpy as np
import pandas as pd

class KNN:
    def __init__(self, classes, train_df, k, distance = None, verbose = False):
        logging.debug('Creating KNN...')
        self.verbose = verbose
        self.training_data = {"points": train_df, "classes": classes}
        self.k = k
        self.distance = lambda a,b : np.linalg.norm(a-b, ord = 2)
        if self.distance is not None:
            self.distance = distance
        # self.cache = []

    def classify(self, point, leave_one_out = False):
        dist_class_pairs = [[None,self.training_data["classes"][idx]] for idx in xrange(0,len(self.training_data["points"]))]
        distances = self.training_data["points"].apply(lambda training_point: self.distance(training_point, point), axis = 1)
        # print(distances)
        # TODO: directly use pandas for selection and stuff too
        #for idx, row in enumerate(self.training_data["points"].values):
        for idx in range(0, len(distances)):
            dist = distances.iloc[idx] #self.distance(row, point)
            # print(self.distance(self.training_data["points"].values[idx], point) - dist)
            dist_class_pairs[idx][0] = dist
        dist_class_pairs.sort(key = lambda pair: pair[0])

        counts = {}
        total_count = 0
        for idx in range(0, len(dist_class_pairs)):
            dist, cat = dist_class_pairs[idx]
            if leave_one_out and dist == 0.0:
                continue
            total_count += 1
            if cat not in counts:
                counts[cat] = 0
            counts[cat] += 1
            if total_count == self.k:
                break
        return max(counts, key=lambda x: counts[x])
