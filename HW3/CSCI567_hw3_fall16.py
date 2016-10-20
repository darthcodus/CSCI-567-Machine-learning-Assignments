from __future__ import print_function
from contextlib import contextmanager
import heapq
import itertools
import logging
import math
import os
import sys

from model_evaluator import ModelEvaluator
from regression import LinearRegression

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal, uniform
import pandas as pd
from scipy.io import loadmat
from svmutil import *

DEBUG = False
OUTPUT_FOLDER = "."

@contextmanager
def open_output_file(fname):
    with open(os.path.join(OUTPUT_FOLDER, "%s.txt" % fname), 'w') as f:
        yield f

def generate_dataset(mean, variance, fx, lower = -1, upper = 1, num_samples = 10):
    dataset = []
    for i in range(0, range(num_samples)):
        xi = uniform(-1, 1)
        dataset.append(xi, fx(xi))
    return dataset

def generate_fx_dataset(num_samples = 10):
    fx = lambda xi : 2*xi*xi + normal(0, math.sqrt(0.1))
    return generate_dataset(0, 0.1)

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if DEBUG else logging.INFO)

    print("********************* a) *********************")
    # a)
    for i in range(0, 100):
        datasets = generate_fx_dataset(10)
        regmodel = LinearRegression()
        eval = ModelEvaluator(regmodel)
        regmodel.train(train_df_features_normalized, train_df_targets)
        trainingError = eval.mean_squared_error(train_df_features_normalized, train_df_targets)
        print("Mean squared error on training data: %f" % trainingError)
        print("Mean squared error on test data: %f" % eval.mean_squared_error(test_df_features_normalized, test_df_targets))
    train_data = loadmat('data/phishing-train.mat')
    test_data = loadmat('data/phishing-test.mat')
    train_features = train_data['features']
    train_labels = train_data['labels']
    test_features = train_data['features']
    test_labels = train_data['labels']
    """
    Histogrammer.plot_histgram_of_features(train_df, 3, 5)
    print("\nClose window to continue")
    plt.show()
    """

if __name__ == "__main__":
    main()
