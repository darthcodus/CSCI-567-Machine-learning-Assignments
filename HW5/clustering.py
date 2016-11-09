import numpy as np
import pandas as pd

class KMeans(object):
    def __init__(self, num_clusters, df):
        assert num_clusters > 1 and df is not None
        self.num_clusters = num_clusters
        self.df = df

        self.centroids = []
        point_gen = self._get_random_point_in_input_range()
        for i in range(0, num_clusters):
            self.centroids.append(next(point_gen))

        """
        import random
        for i in range(0, num_clusters):
            pidx = random.randint(0, len(df) - 1)
            self.centroids[i] = df.values[pidx]
        """

        self.cluster_assignments = None

    def _get_random_point_in_input_range(self, minvals=None, maxvals=None):
        if maxvals is None:
            maxvals = self.df.max(axis=0).values
        if minvals is None:
            minvals = self.df.min(axis=0).values
        while True:
            point = []
            for f in range(0, len(self.df.columns)):
                point.append(np.random.uniform(minvals[f], maxvals[f]))
            yield point

    def _get_diff_l2_norm(self, v1, v2):
        return np.linalg.norm(np.subtract(v1, v2))

    def _get_j(self, centroids, cluster_assigments):
        j = 0.0
        for n, point in enumerate(self.df.values):
            for k in range(0, self.num_clusters):
                if cluster_assigments[n] == k:
                    j += self._get_diff_l2_norm(point, centroids[k])
        return j

    def _get_cluster_assignments(self, centroids):
        cluster_assignments = [-1] * len(self.df)
        for i, point in enumerate(self.df.values):
            mind = self._get_diff_l2_norm(point, centroids[0])
            cluster_assignments[i] = 0
            for j, uk in enumerate(centroids):
                if j == 0:
                    continue
                d = self._get_diff_l2_norm(point, uk)
                if d < mind:
                    mind = d
                    cluster_assignments[i] = j
        assert -1 not in cluster_assignments
        return cluster_assignments

    def _get_updated_centroids(self, cluster_assignments):
        centroids = []
        for cluster in range(0, self.num_clusters):
            centroids.append(np.zeros(len(self.centroids[0])))
            count = 0
            for i, point in enumerate(self.df.values):
                if cluster_assignments[i] == cluster:
                    count += 1
                    centroids[cluster] = np.add(centroids[cluster], point)
            if count is not 0:
                centroids[cluster] = centroids[cluster]/count
            else:
                # randomly initializing again
                centroids[cluster] = next(self._get_random_point_in_input_range())
        assert not np.isnan(np.min(centroids))
        return centroids

    def cluster(self):
        if self.cluster_assignments is not None:
            return self.centroids, self.cluster_assignments
        old_assignments = [-1]*len(self.df)
        i = 0
        print("Iteration: ", end="")
        while True:
            print("%d..." % i, end="")
            i +=1
            self.cluster_assignments = self._get_cluster_assignments(self.centroids)
            self.centroids = self._get_updated_centroids(self.cluster_assignments)
            j = self._get_j(self.centroids, self.cluster_assignments)
            #print("J: %f" % j)
            #print("Assignments: ")
            #print(self.cluster_assignments)
            if np.array_equal(self.cluster_assignments, old_assignments):
                print("\nFinished clustering. Final j:%f" % j)
                print("Assignments: ")
                print(self.cluster_assignments)
                assert len(self.centroids) == self.num_clusters
                return self.centroids, self.cluster_assignments
            old_assignments = self.cluster_assignments
