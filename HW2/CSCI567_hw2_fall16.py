from __future__ import print_function
from contextlib import contextmanager
import heapq
import itertools
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

    print("\n*********************Linear Regression*******************")
    regmodel = LinearRegression()
    eval = ModelEvaluator(regmodel)
    regmodel.train(train_df_features_normalized, train_df_targets)
    trainingError = eval.mean_squared_error(train_df_features_normalized, train_df_targets)
    print("Mean squared error on training data: %f" % trainingError)
    print("Mean squared error on test data: %f" % eval.mean_squared_error(test_df_features_normalized, test_df_targets))

    print("\n***********Ridge regression with lambda 0.01m 0.1, 1.0***************")
    for lambdaval in (0.01, 0.1, 1.0):
        regmodel = RidgeRegression(lambdaval)
        eval = ModelEvaluator(regmodel)
        regmodel.train(train_df_features_normalized, train_df_targets)
        trainingError = eval.mean_squared_error(train_df_features_normalized, train_df_targets)
        testingError = eval.mean_squared_error(test_df_features_normalized, test_df_targets)
        print("Ridge regression model with lambda = %f" % lambdaval)
        print("Mean squared error on training data = %f" % trainingError)
        print("Mean squared error on test data = %f" % testingError)
        print("")

    print("\n*********************Cross Validation*******************")
    lambdaval = float(10.0)

    # Shuffle data
    shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))
    shuffled_train_df_features = train_df.iloc[:, :-1]
    shuffled_train_df_targets = train_df.iloc[:, -1]
    shuffled_train_df_features_normalized = (DataFrameStdNormalizer(shuffled_train_df_features)).get_normalized_data(shuffled_train_df_features)

    lambda_error_map = {}
    for i in range(0,6):
        lambdaval = float(10.0) / (10**i)
        # cross validation
        mean_cv_error = 0
        regmodel = RidgeRegression(lambdaval)
        eval = ModelEvaluator(regmodel)
        for i in range(0,10):
            chunksize = len(train_df)/10
            test_df_cv = None
            train_df_cv_targets = None
            test_df_cv = None
            test_df_cv_targets = None
            test_df_cv = shuffled_train_df_features_normalized.iloc[i*chunksize:i*chunksize+chunksize]
            test_df_cv_targets = shuffled_train_df_targets.iloc[i*chunksize:i*chunksize+chunksize]

            train_df_cv = shuffled_train_df_features_normalized.drop(shuffled_train_df_features_normalized.index[i*chunksize:i*chunksize+chunksize])
            train_df_cv_targets = shuffled_train_df_targets.drop(shuffled_train_df_targets.index[i*chunksize:i*chunksize+chunksize])
            regmodel.train(train_df_cv, train_df_cv_targets)
            #print(eval.mean_squared_error(test_df_cv, test_df_cv_targets))
            mean_cv_error += eval.mean_squared_error(test_df_cv, test_df_cv_targets)
        mean_cv_error /= 10
        print("MSE for lambda %f = %f" % (lambdaval, mean_cv_error))
        lambda_error_map[lambdaval] = mean_cv_error

    lambdabest = min(lambda_error_map, key=lambda_error_map.get)
    print("Lowest MSE for lambda = %f" % lambdabest)
    regmodel = RidgeRegression(lambdabest)
    regmodel.train(train_df_features_normalized, train_df_targets)
    eval = ModelEvaluator(regmodel)
    test_meansquarederror = eval.mean_squared_error(test_df_features_normalized, test_df_targets)
    print("Test error for model with lambda %f = %f" % (lambdabest, test_meansquarederror))
    print("")

    print("\n*********************Feature Selection*******************")
    print("*********************i. Max correlation*******************")
    target_col = col_names[-1]
    corr = {}
    for col in col_names:
        if col.lower() == 'chas': # categorical. Also, see dtypes
            continue
        corr[col] = abs(train_df[[col, target_col]].corr(method='pearson').iloc[0,1])
    maxcorrcols = heapq.nlargest(5, corr, key=corr.get)[1:]
    print("Selecting the following coluns with max correlation: ")
    print(maxcorrcols)
    train_df_features_normalized_maxcorr = train_df_features[maxcorrcols]
    regmodel = LinearRegression()
    regmodel.train(train_df_features_normalized_maxcorr, train_df_targets)
    eval = ModelEvaluator(regmodel)
    trainingError = eval.mean_squared_error(train_df_features[maxcorrcols], train_df_targets)
    print("Mean squared error on training data: %f" % trainingError)
    print("Mean squared error on test data: %f" % eval.mean_squared_error(test_df_features[maxcorrcols], test_df_targets))

    print("*******************ii. Max correlation with residue*****************")
    residue = train_df_targets.copy(deep=True)
    cols = []
    regmodel = LinearRegression()
    eval = ModelEvaluator(regmodel)
    for i in range(0, 4):
        corr = {}
        for col in col_names:
            if col.lower() in ('medv', 'chas') or col in cols: # categorical. Also, see dtypes
                continue
            # corr[col] = train_df[[col]].corrwith(residue).iloc[0]
            corr[col] = abs(pd.concat([train_df[[col]], residue], axis = 1).corr(method='pearson').iloc[0,1])
        maxcorrcol = max(corr, key=corr.get)
        cols.append(maxcorrcol)
        print("Taking cols: %s" % maxcorrcol)
        regmodel.train(train_df_features[cols], train_df_targets)
        for i in range(0,len(residue)):
            residue.at[residue.index[i]] = train_df_targets.iloc[i] - regmodel.predict(train_df_features[cols].iloc[i])
        #trainingError = eval.mean_squared_error(train_df_features_normalized, train_df_targets)
        #print("Mean squared error on training data: %f" % trainingError)
        #print(cols)
        print("Mean squared error on train data: %f" % eval.mean_squared_error(train_df_features[cols], train_df_targets))
    print("Mean squared error on test data: %f" % eval.mean_squared_error(test_df_features[cols], test_df_targets))

    print("*********************iii. All 4 feature combinations*******************")
    bestcols = None
    besttrainmse = 999999
    regmodel = LinearRegression()
    eval = ModelEvaluator(regmodel)
    for cols in list(list(x) for x in itertools.combinations(train_df_features_normalized.columns, 4)):
        regmodel.train(train_df_features_normalized[cols], train_df_targets)
        mse_train = eval.mean_squared_error(train_df_features_normalized[cols], train_df_targets)
        #print("Mean squared error on train data: %f" % )
        #print("Mean squared error on test data: %f" % eval.mean_squared_error(train_df_features_normalized[cols], train_df_targets))
        if mse_train < besttrainmse:
            bestcols = cols
            besttrainmse = mse_train
    print("Best training MSE = %f for columns:" % besttrainmse)
    print(bestcols)
    regmodel.train(train_df_features_normalized[bestcols], train_df_targets)
    print("Testing MSE of this model: %f" % eval.mean_squared_error(test_df_features_normalized[cols], test_df_targets))

    print("\n*********************Feature Expansion*******************")
    df_train_featuregen = train_df_features_normalized.copy(deep=True)
    df_test_featuregen = test_df_features_normalized.copy(deep=True)
    #i = 0
    for cols in list(list(x) for x in itertools.combinations(train_df_features_normalized.columns, 2)) + [[col,col] for col in train_df_features_normalized.columns]:
        #i += 1
        #print("Gen %d: %s" % (i,cols[0]+cols[1]))
        #df_train_featuregen[cols[0]+cols[1]] = df_train_featuregen.apply(lambda x: [[x[cols[0]], x[cols[1]]]], axis=1)
        df_train_featuregen[cols[0]+cols[1]] = df_train_featuregen[cols[0]]*df_train_featuregen[cols[1]]
        df_test_featuregen[cols[0]+cols[1]] = df_test_featuregen[cols[0]]*df_test_featuregen[cols[1]]
        #df_test_featuregen[cols[0]+cols[1]] = df_test_featuregen.apply(lambda x: [[x[cols[0]], x[cols[1]]]], axis=1)
    regmodel = LinearRegression()
    regmodel.train(df_train_featuregen, train_df_targets)
    eval = ModelEvaluator(regmodel)
    trainingError = eval.mean_squared_error(df_train_featuregen, train_df_targets)
    print("Mean squared error on training data: %f" % trainingError)
    print("Mean squared error on test data: %f" % eval.mean_squared_error(df_test_featuregen, test_df_targets))

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
