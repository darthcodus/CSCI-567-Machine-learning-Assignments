import numpy as np
import pandas as pd

class LinearRegression(object):
    def __init__(self, df_train_features, df_train_targets):
        x_aug = np.ones((len(df_train_features), len(df_train_features.iloc[0]) + 1))
        x_aug[:,1:] = df_train_features.values
        """
        or, can do: np.append(np.ones(len(df_train_features), 1), df_train_features.values, axis=1)
        """
        for i, row in enumerate(x_aug):
            for j, entry in enumerate(row):
                print(i,j)
                assert not np.isnan(entry)

        x_aug_t = np.transpose(x_aug)
        self.w = np.dot(np.dot((np.linalg.inv(np.dot(x_aug_t,x_aug))), x_aug_t), df_train_targets.values)

    def predict(self, x):
        return np.dot(np.transpose(self.w),np.append(np.ones((1,)), x))

class RidgeRegression(object):
    def __init__(self, df_train_features, df_train_targets):
        raise NotImplementedError()