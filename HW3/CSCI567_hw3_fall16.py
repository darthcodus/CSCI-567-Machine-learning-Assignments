from __future__ import print_function
from contextlib import contextmanager
import heapq
import itertools
import logging
import math
import os
import sys

from model_evaluator import ModelEvaluator
from regression import KernelizedLinearRegression

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
    for i in range(0, num_samples):
        xi = uniform(-1, 1)
        dataset.append( (xi, fx(xi)) )
    return dataset

def generate_fx_dataset(num_samples = 10):
    fx = lambda xi : 2*xi*xi + normal(0, math.sqrt(0.1))
    return generate_dataset(0, 0.1, fx)

def plot_histgram(df, rows, cols, plot_name, number_of_bins = 10):
    if rows*cols < len(df):
        raise RuntimeError("Not enough subplots for all columns")
    f, axs = plt.subplots(rows, cols)
    for i in range(0, len(df)):
        ax = axs[int(i/cols)][i%cols]
        #df.iloc[0].hist(alpha=.5, bins=number_of_bins, ax=ax)
        bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 10000]
        rwidths = [1] * (len(bins) - 1)
        rwidths[-1] = 1/(float(bins[-1])/50)
        #ax.hist(df.iloc[0].values, bins=bins, rwidth = rwidths)
        data = df.iloc[i].values
        ax.hist(data, bins=range(0, int(1000) + 50, 50))
        ax.title.set_text("%s-%d" % (plot_name, i + 1))
    for i in range(len(df),rows*cols):
        f.delaxes(axs[int(i/cols)][i%cols])
    return f


def pretty_print_header(s):
    print(('  ' + s + '  ').center(50, '*'))


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if DEBUG else logging.INFO)

    g_x_list = []
    g_x_list.append(lambda xi: 1) #g1
    g_x_list.append(lambda xi: [1]) #g2
    g_x_list.append(lambda xi: [1, xi]) #g3
    g_x_list.append(lambda xi: [1, xi, xi*xi]) #g4
    g_x_list.append(lambda xi: [1, xi, xi*xi, xi*xi*xi]) #g5
    g_x_list.append(lambda xi: [1, xi, xi*xi, xi*xi*xi, xi*xi*xi*xi]) #g6

    pretty_print_header("Generating datasets")
    datasets_10 = []
    datasets_100 = []
    for i in range(0, 100):
        datasets_10.append(generate_fx_dataset(10))
        datasets_100.append(generate_fx_dataset(100))

    pretty_print_header("(a)")
    mses_g = []
    for g_x in g_x_list[2:]:
        regmodel = KernelizedLinearRegression(g_x, 0)
        eval = ModelEvaluator(regmodel)
        meanerror = []
        mses = []
        for i in range(0, 100):
            regmodel.train([x[0] for x in datasets_10[i]], [x[1] for x in datasets_10[i]])
            meansquarederror_dataseti = eval.mean_squared_error([x[0] for x in datasets_10[i]], [x[1] for x in datasets_10[i]])
            mses.append(meansquarederror_dataseti)
        mses_g.append(mses)
    #print(mses_g)
    #print(np.subtract(mses_g[0],mses_g[1]))
    print(mses_g[0])
    print(mses_g[1])
    plot_histgram(pd.DataFrame(mses_g), 2, 3, "g", 30)
    plt.show()

    return
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
