import matplotlib.pyplot as plt
import pandas as pd

class Histogrammer(object):
    @staticmethod
    def plot_histgram_of_features(df, rows, cols, number_of_bins = 10):
        if rows*cols < len(df.columns):
            raise RuntimeError("Not enough subplots for all columns")
        f, axs = plt.subplots(rows, cols)
        for i, col in enumerate(df.columns):
            ax = axs[int(i/cols)][i%cols]
            df.hist(col, alpha=.5, bins=number_of_bins, ax=ax)
            ax.title.set_text(col)
        for i in range(len(df.columns),rows*cols):
            f.delaxes(axs[int(i/cols)][i%cols])
        return f
