from __future__ import print_function
from contextlib import contextmanager
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datanormalizer import *
from model_evaluator import ModelEvaluator
from plots import Histogrammer
from regression import LinearRegression, RidgeRegression

"""
    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
                 by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's
"""

DEBUG = False
OUTPUT_FOLDER = "."

@contextmanager
def open_output_file(fname):
    with open(os.path.join(OUTPUT_FOLDER, "%s.txt" % fname), 'w') as f:
        yield f

def main():
    #from sklearn.datasets import load_boston
    #boston = load_boston()
    #print(boston.data.shape)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if DEBUG else logging.INFO)
    dataFile = "data/housing.data"

    col_names = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"]
    train_df = pd.read_csv(dataFile, names = col_names, delim_whitespace = True)
    test_df = train_df.iloc[::7, :]
    train_df.drop(train_df.index[::7], inplace=True)

    train_df_features = train_df.iloc[:, :-1]
    train_df_targets = train_df.iloc[:, -1]
    test_df_features = test_df.iloc[:, :-1]
    test_df_targets = test_df.iloc[:, -1]

    # Data analysis
    print("Data analysis:")
    print("No. of attributes: ", len(train_df.iloc[0]))
    print("No. of features usable for classifcation: ", len(train_df.iloc[0])-1)
    print("Size of training data: ", len(train_df))
    print("Size of testing data: ", len(test_df))
    print("Histogram of attributes will be shown at the end of generating all results")

    print("\nPearson correlations:")
    target_col = col_names[-1]
    for col in col_names:
        if col.lower() == 'chas': # categorical. Also, see dtypes
            continue
        print("Correlation of %s with target(%s): %f" % (col, target_col, train_df[[col, target_col]].corr(method='pearson').iloc[0,1]))

    normalizer = DataFrameStdNormalizer(train_df_features)
    train_df_features_normalized = normalizer.get_normalized_data(train_df_features)
    test_df_features_normalized = normalizer.get_normalized_data(test_df_features)

    print("Linear regression")
    regmodel = LinearRegression()
    eval = ModelEvaluator(regmodel)
    regmodel.train(train_df_features_normalized, train_df_targets)
    trainingError = eval.mean_squared_error(train_df_features_normalized, train_df_targets)
    print("Mean squared error on training data: %f" % trainingError)
    print("Mean squared error on test data: %f" % eval.mean_squared_error(test_df_features_normalized, test_df_targets))

    for lambdaval in (0.01, 0.1, 1.0):
        regmodel = RidgeRegression(lambdaval)
        eval = ModelEvaluator(regmodel)
        regmodel.train(train_df_features_normalized, train_df_targets)
        trainingError = eval.mean_squared_error(train_df_features_normalized, train_df_targets)
        meansquarederror = eval.mean_squared_error(test_df_features_normalized, test_df_targets)
        print("Mean squared error on training data: %f" % trainingError)
        print("Mean squared error for ridge regression with lambda %f: %f" % (lambdaval, meansquarederror))

    # 3.2 training ~20-22, testing: ~28
    # 3.3 training ~23-25, testing ~30-33
    lambdaval = 10
    while lambdaval > 0:
        # cross validation
        for i in range(0,10):
            chunksize = len(train_df)/10
            test_df_cv = None
            train_df_cv_targets = None
            test_df_cv = None
            test_df_cv_targets = None
            test_df_cv = train_df_features_normalized.iloc[i*chunksize:i*chunksize+chunksize]
            test_df_cv_targets = train_df_targets.iloc[i*chunksize:i*chunksize+chunksize]
            train_df_cv = train_df_features_normalized.drop(train_df_features_normalized.index[i*chunksize:i*chunksize+chunksize])
            train_df_cv_targets = train_df_targets.drop(train_df_targets.index[i*chunksize:i*chunksize+chunksize])
            regmodel = RidgeRegression(lambdaval)
            regmodel.train(train_df_cv, train_df_cv_targets)
            eval = ModelEvaluator(regmodel)
            cv_meansqaurederror = eval.mean_squared_error(test_df_cv, test_df_cv_targets)
            test_meansquarederror = eval.mean_squared_error(test_df_features_normalized, test_df_targets)
            print("Ridge regression model with lambda = %f" % lambdaval)
            print("Cross validation test set error = %f" % cv_meansqaurederror)
            print("Testing set error for lambda = %f" % meansquarederror)
            lambdaval /= 10

    return
    print("\n******************************** Showing histogram of attributes********************************")
    Histogrammer.plot_histgram_of_features(train_df, 3, 5)
    print("\nClose window to terminate")
    #plt.show(block=False) #.draw()
    #plt.pause(0.001)
    #raw_input("Press enter to continue")
    plt.show()
    return

if __name__ == "__main__":
    main()
