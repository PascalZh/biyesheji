#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detection Module.
Including point cloud processing functions, pattern extraction functions, and
others.
"""
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def extract_patterns(image):
    """Use DBSCAN or other algorithms to extract patterns from a 2d point cloud.
    Args:
        image (np.array(dtype='float', shape=(n, 2))): a 2d point cloud.

    Returns:
        np.array(dtype='float', shape=(n, 1)): labels of the patterns.
    """
    clustering = DBSCAN().fit(image)
    labels = clustering.labels_
    return labels


def distance(predict_j, observation_k):
    pass
