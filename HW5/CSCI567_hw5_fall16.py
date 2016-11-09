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

    df_circle.plot(kind='scatter', x='f1', y='f2')
    df_blob.plot(kind='scatter', x='f1', y='f2');

    #for num_cluster in (2, 3, 5):
    c1 = KMeans(num_clusters=2, df=df_circle)
    print()
    #plt.show()


if __name__ == "__main__":
    main()
