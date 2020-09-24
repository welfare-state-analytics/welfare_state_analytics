#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.classification import confusion_matrix
from json import load
from sys import argv,exit

def plot_confusion_matrix(y_true=None, y_pred=None, cm=None, labels=None, max_size=2048, normalize=True, colors=plt.cm.Blues, title='Confusion'):
    x = get_confusion_plot(y_true, y_pred, cm, labels, max_size, normalize, colors, title)


def get_confusion_plot(y_true=None, y_pred=None, cm=None, labels=None, max_size=2048, normalize=True, colors=plt.cm.Blues, title='Confusion'):
    if type(cm) == type(None):
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    if not labels:
        labels = [ x+1 for x in range(cm.shape[0]) ]

    plt.close()

    #cm_n = cm = cm / cm.sum(axis=1)
    if normalize:
        #cm = cm / cm.sum(axis=1)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig,ax = plt.subplots(figsize=(cm.shape[0], cm.shape[1]), dpi=max_size/max(cm.shape))

    plt.xticks(range(len(labels)), labels, fontsize=14.0, fontweight='bold', rotation='45')
    plt.yticks(range(len(labels)), labels, fontsize=14.0, fontweight='bold')
    im = ax.imshow(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                   interpolation='nearest',
                   cmap=colors)

    #res = ax.imshow(np.array(cm_n),
    #                cmap=plt.cm.jet,
    #                interpolation='nearest')

    ax.xaxis.set_tick_params(width=3, length=10)
    ax.yaxis.set_tick_params(width=3, length=10)

    ax.set_xlabel('Machine', fontsize=6 + max_size/100)
    ax.set_ylabel('Man', fontsize=6 + max_size/100)
    ax.set_title(title, fontsize=6 + max_size/100)

    width, height = cm.shape

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] >= 0.01:
                ax.text(j, i, format(cm[i, j], fmt),
                        fontsize=16,
                        fontweight='bold',
                        alpha=0.25,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    return fig

if __name__ == "__main__":
    if len(argv) < 2:
        print(f'usage: {argv[0]} <matrix.json> [labels.json]')
        exit(1)

    cm = np.array(load(open(argv[1])))
    labels = load(open(argv[2])) if len(argv) > 2 else None
    fig = get_confusion_plot(cm=cm, labels=labels, max_size=2048, colors=plt.cm.Blues, normalize=True)

    fig.savefig('confusion_matrix.png', format='png')

