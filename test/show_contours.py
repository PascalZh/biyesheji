#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('../serial/contours.p', 'rb') as f:
    contours = pickle.load(f)
    for contour in contours:
        contour = contour.squeeze()
        x = contour[:,0]
        y = contour[:,1]
        x = x - np.min(x)
        y = y - np.min(y)
        w = np.max(x) - np.min(x) + 1
        h = np.max(y) - np.min(y) + 1
        im = np.zeros((w, h), dtype='uint8')
        im[x,y] = 255
        plt.imshow(im, origin='lower', cmap='gray')
        plt.show()
