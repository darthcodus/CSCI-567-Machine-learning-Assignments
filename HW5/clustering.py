import numpy as np
import pandas as pd

class KMeans(object):
    def __init__(self, num_clusters, df):
        self.num_clusters = num_clusters
        self.df = df

        self.uk = []
        maxvals = df.max(axis=0).values
        minvals = df.min(axis=0).values
        for i in range(0, num_clusters):
            self.uk.append([])
            for f in range(0, len(df.columns)):
                self.uk[i].append(np.random.uniform(minvals[f], maxvals[f]))

    def _get_j(self, cluster_assigments):
        j = 0.0
        for n, point in enumerate(self.df.values):
            for k in self.num_clusters:
                if cluster_assigments[n] == k:
                    j += np.linalg.det(np.subtract(point, self.uk[k]))
