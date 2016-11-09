from __future__ import print_function
from contextlib import contextmanager
import heapq
import itertools
import logging
import math
import os
import sys
import time

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

def plot_histgram(df, rows, cols, plot_name, title):
    if rows*cols < len(df):
        raise RuntimeError("Not enough subplots for all columns")
    f, axs = plt.subplots(rows, cols)
    f.suptitle(title)
    for i in range(0, len(df)):
        ax = axs[int(i/cols)][i%cols]
        #df.iloc[0].hist(alpha=.5, bins=number_of_bins, ax=ax)
        bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        rwidths = [1] * (len(bins) - 1)
        rwidths[-1] = 1/(float(bins[-1])/50)
        #ax.hist(df.iloc[0].values, bins=bins, rwidth = rwidths)
        data = df.iloc[i].values
        ax.hist(data, bins=range(0, int(1000) + 50, 50))
        ax.title.set_text("%s%d" % (plot_name, i + 1))
    for i in range(len(df),rows*cols):
        f.delaxes(axs[int(i/cols)][i%cols])
    return f


def pretty_print_header(s):
    print(('  ' + s + '  ').center(50, '*'))


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if DEBUG else logging.INFO)

    g_x_list = []
    g_x_list.append(lambda xi: [1]) #g1
    g_x_list.append(lambda xi: [1]) #g2
    g_x_list.append(lambda xi: [1, xi]) #g3
    g_x_list.append(lambda xi: [1, xi, xi*xi]) #g4
    g_x_list.append(lambda xi: [1, xi, xi*xi, xi*xi*xi]) #g5
    g_x_list.append(lambda xi: [1, xi, xi*xi, xi*xi*xi, xi*xi*xi*xi]) #g6

    pretty_print_header("Bias Variance Trade-off")
    pretty_print_header("Generating datasets")
    datasets_10 = []
    datasets_100 = []
    for i in range(0, 100):
        datasets_10.append(generate_fx_dataset(10))
        datasets_100.append(generate_fx_dataset(100))

    pretty_print_header("(a) and (b)")
    for datasets, title in ((datasets_10, "Part (a)"), (datasets_100, "Part (b)")):
        avg_ws = []
        mses_g = []
        # g_1
        mses_g1 = []
        for i in range(0, 100):
            mses_g1.append(np.mean([(x[1] - 1)**2 for x in datasets[i]]))
        mses_g.append(mses_g1)
        avg_ws.append([1])

        # g_2
        mses_g2 = []
        avg_w2 = 0
        for i in range(0, len(datasets)):
            w_0 = sum([x[1] for x in datasets[i]]) / len(datasets)
            mses_g2.append(np.mean([(x[1] - w_0)**2 for x in datasets[i]]))
            avg_w2 += w_0
        mses_g.append(mses_g2)
        avg_w2 /= len(datasets)
        avg_ws.append([avg_w2])

        # g3-6
        for g_x in g_x_list[2:]:
            wmean = [0]*len(g_x(0))
            regmodel = KernelizedLinearRegression(g_x, 0)
            eval = ModelEvaluator(regmodel)
            meanerror = []
            mses = []
            for i in range(0, len(datasets)):
                regmodel.train([x[0] for x in datasets[i]], [x[1] for x in datasets[i]])
                meansquarederror_dataseti = eval.mean_squared_error([x[0] for x in datasets[i]], [x[1] for x in datasets[i]])
                mses.append(meansquarederror_dataseti)
                wmean = np.add(wmean, regmodel.w)
            mses_g.append(mses)
            wmean /= len(datasets)
            avg_ws.append(wmean)


        #print(mses_g)
        #print(np.subtract(mses_g[0],mses_g[1]))
        # print(mses_g[0])
        # print(mses_g[1])
        plot_histgram(pd.DataFrame(mses_g), 2, 3, "g", title)
    plt.show()

    pretty_print_header("Linear and Kernel SVM")
    train_data = loadmat('data/phishing-train.mat')
    test_data = loadmat('data/phishing-test.mat')
    train_features = train_data['features']
    train_labels = train_data['label']
    test_features = test_data['features']
    test_labels = test_data['label']

    train_df_features = pd.DataFrame(train_data['features'])
    train_df_labels = pd.DataFrame(train_data['label'])
    test_df_features = pd.DataFrame(test_data['features'])
    test_df_labels = pd.DataFrame(test_data['label'])

    categoricals_feature_columns = [1, 6, 7, 13, 14, 15, 25, 28]
    other_feature_columns = sorted(list(set(range(0,len(train_df_features.columns))) - set(categoricals_feature_columns)))

    df = train_df_features
    df_cat_train = pd.get_dummies(df[categoricals_feature_columns].applymap(str))
    df_others = df[other_feature_columns]
    train_df_features = df_others.join(df_cat_train.applymap(float))

    df = test_df_features
    df_cat_test = pd.get_dummies(df[categoricals_feature_columns].applymap(str))
    df_others = df[other_feature_columns]
    test_df_features = df_others.join(df_cat_test[df_cat_train.columns])

    # Feed to svmutil
    start = time.clock()
    crange = range(-6, 3)
    for c in crange:
        print("Evaluating svm for c=4^%d"%c)
        m = svm_train(train_labels.tolist()[0], train_df_features.values.tolist(), '-c %f -v 3' % math.pow(4,c))
    print("Average training time=%fs" % ((time.clock() - start)/len(crange)))

    crange = range(-3,8)
    degrees = (1,2,3)
    start = time.clock()
    for c in crange:
        for degree in degrees:
            print("Evaluating svm for c=4^%d"%c)
            print("Degree = %d" % degree)
            m = svm_train(train_labels.tolist()[0], train_df_features.values.tolist(), '-t %d -c %f -v 3 -d %d' % (1, math.pow(4,c), degree))
    print("Average training time=%fs" % ((time.clock() - start)/(len(crange)*len(degrees))))

    gammedegrees = range(-7, -1)
    start = time.clock()
    for c in crange:
        for degree in gammedegrees:
            print("Evaluating svm for c=4^%d"%c)
            print("Gamme = %f" % math.pow(4,degree))
            m = svm_train(train_labels.tolist()[0], train_df_features.values.tolist(), '-t %d -c %f -v 3 -g %f' % (2, math.pow(4,c), math.pow(4,degree)))
    print("Average training time=%fs" % ((time.clock() - start)/(len(crange)*len(degrees))))
    #m = svm_load_model('heart_scale.model')
    #p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
    #ACC, MSE, SCC = evaluations(y, p_label)

    #m = svm_train(train_labels.tolist()[0], train_df_features.values.tolist(), '-t %d -c %f -v 3 -g %f' % (2, math.pow(4,c), math.pow(4,degree)))
    m = svm_train(train_labels.tolist()[0], train_df_features.values.tolist(), '-t %d -c %f -g %f' % (2, math.pow(4,2), math.pow(4,-2)))
    p_labs, p_acc, p_vals = svm_predict(test_labels.tolist()[0], test_df_features.values.tolist(), m)
    print(p_acc)
    print("Best performing model: rbf-kernel svm with c=4^2, gamma = -2")
if __name__ == "__main__":
    main()
