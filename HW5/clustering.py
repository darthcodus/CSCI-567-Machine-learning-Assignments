import numpy as np
import pandas as pd

import utils

class KMeans(object):
    def __init__(self, num_clusters, df):
        assert num_clusters > 1 and df is not None
        self.num_clusters = num_clusters
        self.df = df
        self.cluster_assignments = None
        self.centroids = utils.get_n_points_in_range(num_clusters, len(self.df.columns), self.df.min(axis=0).values, self.df.max(axis=0).values)
        """
        import random
        for i in range(0, num_clusters):
            pidx = random.randint(0, len(df) - 1)
            self.centroids[i] = df.values[pidx]
        """

    def _get_diff_norm(self, v1, v2, order=2):
        return np.linalg.norm(np.subtract(v1, v2), ord=order)

    def _get_j(self, centroids, cluster_assigments):
        j = 0.0
        for n, point in enumerate(self.df.values):
            for k in range(0, self.num_clusters):
                if cluster_assigments[n] == k:
                    j += self._get_diff_norm(point, centroids[k])
        return j

    def _get_cluster_assignments(self, centroids):
        cluster_assignments = [-1] * len(self.df)
        for i, point in enumerate(self.df.values):
            mind = self._get_diff_norm(point, centroids[0])
            cluster_assignments[i] = 0
            for j, uk in enumerate(centroids):
                if j == 0:
                    continue
                d = self._get_diff_norm(point, uk)
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
                centroids[cluster] = utils.get_random_point_in_range(len(self.df.columns), self.df.min(axis=0).values, self.df.max(axis=0).values)
        assert not np.isnan(np.min(centroids))
        return centroids

    def cluster(self, stepwise_plots=False):
        if self.cluster_assignments is not None:
            return self.centroids, self.cluster_assignments
        old_assignments = [-1]*len(self.df)
        i = 0
        print("Iteration: ", end="")
        cluster_colors = utils.get_n_points_in_range(self.num_clusters, 3, (0,0,0), (1,1,1))
        while True:
            print("%d..." % i, end="")
            i +=1
            self.cluster_assignments = self._get_cluster_assignments(self.centroids)
            self.centroids = self._get_updated_centroids(self.cluster_assignments)
            j = self._get_j(self.centroids, self.cluster_assignments)

            if stepwise_plots:
                import matplotlib.pyplot as plt
                f, axs = plt.subplots(1)
                self.df.plot(kind='scatter', x='f1', y='f2', c=[cluster_colors[i] for i in self.cluster_assignments], ax=axs)
                for k in range(0, len(self.centroids)):
                    axs.scatter(self.centroids[k][0], self.centroids[k][1], c=cluster_colors[k], marker='x', s=200)
                plt.show()

            if np.array_equal(self.cluster_assignments, old_assignments):
                print("\nFinished clustering. Final j:%f" % j)
                print("Assignments: ")
                print(self.cluster_assignments)
                assert len(self.centroids) == self.num_clusters
                return self.centroids, self.cluster_assignments
            old_assignments = self.cluster_assignments
