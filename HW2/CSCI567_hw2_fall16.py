from __future__ import print_function
from contextlib import contextmanager
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datanormalizer import *
from plots import Histogrammer

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

    normalizer = DataFrameStdNormalizer(train_df)
    train_df_n = normalizer.get_normalized_data(train_df)
    test_df_n = normalizer.get_normalized_data(test_df)

    # Data analysis
    print("Data analysis:")
    print("No. of attributes: ", len(train_df.iloc[0]))
    print("No. of features usable for classifcation: ", len(train_df.iloc[0])-1)
    print("Size of training data: ", len(train_df))
    print("Size of testing data: ", len(test_df))
    Histogrammer.plot_histgram_of_features(train_df, 3, 5)
    plt.show()
    return

if __name__ == "__main__":
    main()
