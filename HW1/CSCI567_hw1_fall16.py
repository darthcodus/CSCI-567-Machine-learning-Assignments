from __future__ import print_function
from contextlib import contextmanager
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from gaussianbayes import GaussianBayesClassifier
from knn import KNN
import numpy as np

"""
1. Id number: 1 to 214
2. RI: refractive index
3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
4. Mg: Magnesium
5. Al: Aluminum
6. Si: Silicon
7. K: Potassium
8. Ca: Calcium
9. Ba: Barium
10. Fe: Iron
11. Type of glass: (class attribute)
"""
# 1. 11 attributes
    # 1,1.52101,13.64,4.49,1.10,71.78,0.06,8.75,0.00,0.00,1
# 2. one of which is id, not useful for classification, and one is the class. The rest 2-10 are attributes/features which can be used for classifcation
# 3. there are 7 classes
# 4. The class distrubution histogram (Figure 1) shows the class distribution.
    # Class 2 is the majority class. No, the distribution is not uniform

DEBUG = True
OUTPUT_FOLDER = "."

@contextmanager
def open_output_file(fname):
    with open(os.path.join(OUTPUT_FOLDER, "%s.txt" % fname), 'w') as f:
        yield f

def main(): # TODO: test with user input, confirm input with TAs
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if DEBUG else logging.INFO)
    trainingData = "data/train.txt"
    testingData = "data/test.txt"
    if len(sys.argv) > 1:
        trainingData = sys.argv[1]
        testingData = sys.argv[2]
        print("Taking %s as training data, and %s as testing data" % trainingData, testingData)
    col_names = ["index", "ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe", "type"]
    train_df = pd.read_csv(trainingData, names = col_names)
    test_df = pd.read_csv(testingData, names = col_names)

    # Data stats
    print("Data characteristics:")
    print("No. of attributes: ", len(train_df.iloc[0]))
    print("No. of features usable for classifcation: ", len(train_df.iloc[0])-2)
    print("Size of training data", len(train_df))
    print("Size of testing data", len(test_df))
    print("No. of unique classes: ", 7)
    print("Unique classes represented in training data: ", train_df['type'].unique())
    print("\t(Histogram of classes in figure 1)")
        # plot class histogram
    train_df.hist('type')
    plt.title("Figure 1: Class (glass type) Histogram")
    # /Data stats

    # Run KNN for k = 1, 3, 5, 7 and L1 & L2 norms on training (leave one out) and test sets
    for k in (1, 3, 5, 7):
        for order in (1,2): # order of the norm
            print("Running KNN of order %d on test set with L-%d norm" % (k, order))
            knn = KNN(train_df.iloc[:,-1], train_df.iloc[:, 1:-1], k, distance = lambda a,b: np.linalg.norm(a-b, ord = order))
            for title, filename, data, leave_one_out in (("TEST", "knn_%d_l%d_test" % (k, order), test_df, False), ("TRAIN", "knn_%d_l%d_train" % (k, order), train_df, True)):

                with open_output_file(filename) as f:
                    f.write("#index,predicted_class,actual_class")
                    total = 0
                    correct = 0
                    for row in data:
                        logging.debug("Classifying")
                        predicted = knn.classify(row[1:-1], leave_one_out)
                        actual = row[-1]
                        f.write("%d,%d,%d" % (row[0], predicted, actual))
                        total += 1
                        if actual == predicted:
                            correct += 1
                    accstr = "Accuracy on %s data: " % title, float(correct)/total
                    f.write(accstr)
                    print(accstr)

    # gb = GaussianBayesClassifier(train_df.iloc[:, 1], train_df.iloc[:, 1:-1], DEBUG)
    # plt.show()

if __name__ == "__main__":
    main()
