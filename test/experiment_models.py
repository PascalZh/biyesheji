#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.cluster import MeanShift, DBSCAN, OPTICS

sys.path.append('..')
import model
from utils import save_folders, load_folders

filename = "edge_detection_dataset"

def plot2(x, y, y_hats, fmt1='b.', fmt2='g.'):
    for i in range(len(y_hats)):
        plt.subplot(1, len(y_hats), i+1)
        plt.plot(x, y, fmt1)
        plt.plot(x, y_hats[i], fmt2)
    plt.show()


for xy in load_folders(filename, 'poly2_0'):
    print(f"xy shape: {xy.shape}")

    xy = xy[np.argsort(xy[:,0])]

    x = xy[:,0]
    y = xy[:,1]

    clustering = DBSCAN().fit(xy)
    labels = clustering.labels_
    # plt.scatter(x, y, s=4, c=labels)

    # plt.show()


# ma = model.MA(60, 1)
# y_hats = [ma(y)]
# ma = model.MA(3, 20)
# y_hats.append(ma(y))
# ma = model.MA(3, 100)
# y_hats.append(ma(y))
# plot2(x, y, y_hats)

