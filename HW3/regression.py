import numpy as np
import pandas as pd

class LinearRegression(object):
    def __init__(self, lambda_val = 0, kernel = None):
        self.w = None
        self.kernel = kernel
        self.lambda_val = lambda_val

    def kernelify(self, feature_mat):
        phi = []
        if self.kernel is not None:
            for i in range(0, len(feature_mat)):
                phi.append(self.kernel(feature_mat[i]))
        else:
            phi = feature_mat
        return phi

    def train(self, train_features, train_targets):
        x_aug = np.ones((len(train_features), len(train_features[0]) + 1))
        x_aug[:,1:] = train_features

        for i, row in enumerate(x_aug):
            for j, entry in enumerate(row):
                assert not np.isnan(entry)

        phi = self.kernelify(x_aug)
        k = np.dot(phi, np.transpose(phi))
        x_aug_t = np.transpose(phi)
        lambdai = self.lambdaval*np.identity(len(k))
        # w_map = \phi_T (K + \lambda I )^{-1}  y
        self.w = np.dot( np.dot(np.transpose(phi), np.linalg.inv(np.add(k, lambdai))), train_targets)

    def predict(self, x):
        assert self.w is not None
        return np.dot(np.transpose(self.w), np.append(np.ones((1,)), x))
