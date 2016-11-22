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

    #hmm.test_hmm()

    transition_probs = [ [0.7, 0.3], [0.4, 0.6] ]
    emission_probs = [[0.4, 0.2, 0.3, 0.1], [0.2, 0.4, 0.1, 0.3]]
    initial_probs = [0.6, 0.4]
    state_labels = ['S1', 'S2']
    emission_labels = ['a', 'c', 'g', 't']
    model = hmm.HMM(initial_probs, transition_probs, emission_probs, state_labels, emission_labels)
    emission_seq_labels = [c for c in 'accgta']
    emission_idx_list = model._get_emission_idx_seq_from_label_seq(emission_seq_labels)
    print("O/p prob", model.calc_prob_output_sequence(emission_seq_labels))
    print(model.get_likelihood(5, 'S1', emission_seq_labels))
    assert math.isclose(model.get_likelihood(5, 'S1', emission_seq_labels), model.alpha_t_helper(5, 0, emission_idx_list)/model.calc_prob_output_sequence(emission_seq_labels))
    print(model.get_likelihood(5, 'S2', emission_seq_labels))
    print(model.get_likelihood(3, 'S1', emission_seq_labels))
    print(model.get_likelihood(3, 'S2', emission_seq_labels))
    #print(model.alpha_t_helper(5, 1, emission_idx_list))

    pretty_print_header("Viterbi algorith on ACCGTA to get most likely sequence of states:")
    print(model.get_most_likely_state_seq_from_labels(emission_seq_labels))
    """
    for i in range(0,len(emission_probs[0])):
        print("Emission: %s" % emission_labels[i])
        print(0.60206796235*emission_probs[0][i] + 0.39793203763*emission_probs[1][i])
    """

if __name__ == "__main__":
    main()
