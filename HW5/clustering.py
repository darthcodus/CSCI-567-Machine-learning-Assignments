from __future__ import print_function
import math

import numpy as np
import pandas as pd

import utils

class KMeans(object):
    def __init__(self, num_clusters, df, verbose=True):
        assert num_clusters > 1 and df is not None
        self.num_clusters = num_clusters
        self.df = df
        self.cluster_assignments = None
        self.centroids = None
        self._initialize_centroids()
        self.verbose = verbose
        """
        import random
        for i in range(0, num_clusters):
            pidx = random.randint(0, len(df) - 1)
            self.centroids[i] = df.values[pidx]
        """

    def _initialize_centroids(self):
        self.centroids = utils.get_n_points_in_range(self.num_clusters, len(self.df.columns), self.df.min(axis=0).values, self.df.max(axis=0).values)

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
        if self.verbose:
            print("Iteration: ", end="")
        cluster_colors = utils.get_n_points_in_range(self.num_clusters, 3, (0,0,0), (1,1,1))
        while True:
            if self.verbose:
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
                if self.verbose:
                    print("\nFinished clustering. Final j:%f" % j)
                    print("Assignments: ")
                    print(self.cluster_assignments)
                assert len(self.centroids) == self.num_clusters
                return self.centroids, self.cluster_assignments
            old_assignments = self.cluster_assignments


class KernelKMeans(object):
    def __init__(self, num_clusters, df, kernel_func=None, kernel_mat=None):
        if (kernel_mat is not None and kernel_mat is not None) or (kernel_func is None and kernel_mat is None):
            raise RuntimeError("Must specify only one of kernel function or kernel matrix")

        assert num_clusters > 1 and df is not None
        self.num_clusters = num_clusters
        self.df = df
        #self.cluster_assignments = None
        #self.centroids = None

        if kernel_func is not None:
            self.kernel_mat = self._get_kernel_matrix(df, kernel_func)
        else:
            self.kernel_mat = kernel_mat
        #super(KernelKMeans, self).__init__(num_clusters=num_clusters, df=df)

    def _get_kernel_matrix(self, df, kernel_func):
        kernel_mat = []
        for i, xi in enumerate(df.values):
            kernel_mat.append([])
            for j, xj in enumerate(df.values):
                kernel_mat[i].append(kernel_func(xi, xj))
        return np.array(kernel_mat)

    def _get_d(self):
        d = np.zeros((len(self.df), self.num_clusters))
        for j in range(self.num_clusters):
            gamma = np.array(self.cluster_assignments) == j #*len(self.df)
            nk = np.sum(gamma)
            assert not nk == 0
            d[:, j] += np.sum(np.outer(gamma, gamma) * self.kernel_mat / nk**2) - 2 * np.sum(gamma * self.kernel_mat, axis=1) / nk
        return d

    def cluster(self):
        self.cluster_assignments = []
        for i in range(0, len(self.df)):
            self.cluster_assignments.append(np.random.randint(0, self.num_clusters))
        i = 1
        print("Iteration: ", end="")
        while True:
            print("%d..." % i, end="")
            i += 1
            d = self._get_d()
            old_assignments = self.cluster_assignments
            self.cluster_assignments = d.argmin(axis=1)
            if np.array_equal(self.cluster_assignments, old_assignments):
                print("\nFinished clustering.")# Final d", d)
                print("Assignments: ")
                print(self.cluster_assignments)
                return self.cluster_assignments


class GMM(object):
    def __init__(self, num_clusters, df):
        assert num_clusters > 1 and df is not None
        self.num_clusters = num_clusters
        self.df = df
        self.centroids = None
        self.cluster_assignments = []
        for i in range(0, len(self.df)):
            self.cluster_assignments.append(np.random.randint(0, self.num_clusters))
        d = len(self.df.columns)
        centroids, labels = (KMeans(df=self.df, num_clusters=self.num_clusters, verbose=False)).cluster()
        self.vars = []
        for i in range(0, self.num_clusters):
            self.vars.append(np.identity(d))
        self.means = centroids
        self.gamma = None

    def get_gamma(self):
        if self.gamma is not None:
            return self.gamma
        self.gamma = []
        for i in range(0, len(self.df)):
            self.gamma.append([])
            for j in range(0, self.num_clusters):
                self.gamma[i].append(self.get_gamma_ij(i, j))
        return self.gamma

    def _update_predictions(self):
        #wjnew = []
        #for i in range(0, self.num_clusters):
        #    wjnew.append(np.sum(np.array(self.cluster_assignments) == i))
        gamma = self.get_gamma()
        mnew = []
        for i in range(0, self.num_clusters):
            mnew.append(np.zeros(len(self.df.columns)))
            for j in range(0, len(self.df)):
                mnew[i] = np.add(mnew[i], gamma[j][i]*self.df.values[j])
            mnew[i] *= 1/self.n_j(i)

        vnew = []
        for j in range(0, self.num_clusters):
            vnew.append(np.zeros( (len(self.df.columns), len(self.df.columns)) ) )
            for i in range(0, len(self.df)):
                v = np.subtract(self.df.values[i], mnew[j])[:, np.newaxis]
                #print(v)
                vnew[j] = np.add(vnew[j], gamma[i][j] * np.dot(v, v.T))
            vnew[j] /= self.n_j(j)

        #self.w = wjnew
        self.means = mnew
        self.vars = vnew
        self.gamma = None

    def get_gamma_ij(self, i, j):
        dnm = 0
        for l in range(0, self.num_clusters):
            dnm += utils.multivariate_gaussian(self.df.values[i], self.means[l], self.vars[l])
        return utils.multivariate_gaussian(self.df.values[i], self.means[j], self.vars[j]) / dnm

    def n_j(self, j):
        nj = 0
        for i in range(0, len(self.df)):
            nj += self.get_gamma()[i][j]
        return nj

    def get_L(self):
        n = len(self.df)
        l = 0
        for i in range(0, n):
            t = 0
            for j in range(0, self.num_clusters):
                t += utils.multivariate_gaussian(self.df.values[i], self.means[j], self.vars[j]) # *self.w[j
            l += math.log(t)
        l /= n
        return l

    def cluster(self):
        i = 0
        self.l = []
        gammas = []
        means = []
        covars = []
        while i < 5:
            i+=1
            print("Iteration: %d" % i)
            self._update_predictions()
            gammas.append(self.get_gamma())
            means.append(self.means)
            covars.append(self.vars)
            self.l.append(self.get_L())
            print("L new: %f" % self.l[-1])
        bestgammaidx = self.l.index(max(self.l))
        print("Means:")
        print(means[bestgammaidx])
        print("Covariances:")
        print(covars[bestgammaidx])
        return np.array(gammas[bestgammaidx]).argmax(axis=1)
