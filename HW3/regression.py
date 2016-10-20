import numpy as np
import pandas as pd

class KernelizedLinearRegression(object):
    def __init__(self, kernel, lambda_val = 0):
        self.w = None
        self.kernel = kernel
        self.lambda_val = lambda_val

    def kernelify(self, feature_mat):
        phi = []
        for i in range(0, len(feature_mat)):
            phi.append(self.kernel(feature_mat[i]))
        return phi

    def train(self, train_features, train_targets):
        #print(train_features)
        #x_aug = np.ones((len(train_features), len(train_features[0]) + 1))
        #x_aug[:,1:] = train_features
        x_aug = train_features

        phi = self.kernelify(x_aug)
        #print(phi)
        k = np.dot(phi, np.transpose(phi))
        #print(k)
        x_aug_t = np.transpose(phi)
        lambdai = self.lambda_val*np.identity(len(k))
        # w_map = \phi_T (K + \lambda I )^{-1}  y
        self.w = np.dot( np.dot(np.transpose(phi), np.linalg.inv(np.add(k, lambdai))), train_targets)

    def predict(self, x):
        assert self.w is not None
        return np.dot(np.transpose(self.w), self.kernel(x))
