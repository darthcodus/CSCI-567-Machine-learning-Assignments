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

import hmm

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
    transition_probs = [ [0.7, 0.3], [0.4, 0.6] ]
    emission_probs = [[0.4, 0.2, 0.3, 0.1], [0.2, 0.4, 0.1, 0.3]]
    initial_probs = [0.6, 0.4]
    state_labels = ['s1', 's2']
    emission_labels = ['a', 'c', 'g', 't']
    model = hmm.HMM(initial_probs, transition_probs, emission_probs, state_labels, emission_labels)
    print(model.calc_prob_output_sequence([c for c in 'accgta']))
    hmm.test_hmm()

if __name__ == "__main__":
    main()
