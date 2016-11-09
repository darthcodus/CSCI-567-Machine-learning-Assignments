from __future__ import print_function
from contextlib import contextmanager
import heapq
import itertools
import logging
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal, uniform
import pandas as pd
from scipy.io import loadmat

from clustering import KMeans

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

def pretty_print_header(s):
    print(('  ' + s + '  ').center(50, '*'))


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if DEBUG else logging.INFO)
    col_names = ["f1", "f2"]
    df_blob = pd.read_csv("data/hw5_blob.csv", names=col_names)
    df_circle = pd.read_csv("data/hw5_circle.csv", names=col_names)

    #df_circle.plot(kind='scatter', x='f1', y='f2')
    #df_blob.plot(kind='scatter', x='f1', y='f2');

    get_color = lambda x: ((1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,1,1))[x]
    f, axs = plt.subplots(2, 3)
    for i, (dataset_name, dataset_dataframe) in enumerate((("hw5_circle", df_circle), ("hw5_blob", df_blob))):
        print("Running K-Means for dataset: %s" % dataset_name)
        for j, num_clusters in enumerate((2, 3, 5)):
            print("Number of clusters: %d" % num_clusters)
            c = KMeans(num_clusters=num_clusters, df=df_circle)
            centroids, cluster_assignments = c.cluster()
            ax = ax = axs[i][j]
            dataset_dataframe.plot(kind='scatter', x='f1', y='f2', c=[get_color(i) for i in cluster_assignments], ax=ax)
            ax.title.set_text("%d clusters" % num_clusters)

            for k in range(0, len(centroids)):
                ax.scatter(centroids[k][0], centroids[k][1], c=get_color(k), marker='x', s=200)

            print()
    plt.show()


if __name__ == "__main__":
    main()
