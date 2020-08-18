import numpy as np
import pandas as pd
from scipy.stats import pearsonr    # may need to also import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("always")


def calculate_pvalues(dfr, method_func=pearsonr):
    """
    dfr : dataframe
    method_func : function (NOT A STRING)
        - either pearsonr or spearmanr from scipy.stats
    """
    dfcols = pd.DataFrame(columns=dfr.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in dfr.columns:
        for c in dfr.columns:
            pvalues[r][c] = round(method_func(dfr[r], dfr[c])[1], 2)
    return pvalues


def get_p_val_mask(dfr, method_func=pearsonr, p_val_cutoff=0.1):
    """
    dfr : dataframe
    method_func : function (NOT A STRING)
        - either pearsonr or spearmanr from scipy.stats
    p_val_cutoff : float
        - 0 < p_val_cutoff < 1
        - other floats work in theory but would not make sense
    """
    p_vals_mtrx = calculate_pvalues(dfr, method_func=method_func)
    for c in list(p_vals_mtrx):
        p_vals_mtrx[c] = p_vals_mtrx[c].apply(
            lambda x: 0 if x >= p_val_cutoff else 1)
    return p_vals_mtrx


def plot_distributions():
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

    # plot data:
    for j in range(5):
        for i in range(5):
            if i == j:
                axes[j][i].hist(x=train_df[list(train_df)[i]], bins=20,
                                color="blue", alpha=0.75)
                axes[j][i].set_yticklabels([])
            else:
                axes[j][i].scatter(x=train_df[list(train_df)[i]],
                                   y=train_df[list(train_df)[j]],
                                   color="blue", s=2, alpha=0.6)

    # formatting: label x-axis of bottom row and y-axis of left column
    for k in range(5):
        label_text = list(train_df)[k].replace(" ", "\n")
        axes[4][k].set_xlabel(label_text)
        axes[k][0].set_ylabel(label_text, rotation=0, ha="right", va="center")

    fig.suptitle("Distribution of Data in Training Set by Feature",
                 fontsize=16, y=.93)

    plt.subplots_adjust(wspace=.5, hspace=.3)
    plt.show()
